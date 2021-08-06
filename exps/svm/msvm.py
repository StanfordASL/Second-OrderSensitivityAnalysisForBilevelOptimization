import sys, pdb, time, gzip, pickle

import torch, cvxpy as cp, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import line_profiler as lp

import osqp

import header

from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd
from implicit import implicit_grads_1st, implicit_grads_2nd, generate_fns
from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit.pca import visualize_landscape

from objs import LS, CE
import mnist

t2n = lambda x: np.copy(x.detach().cpu().clone().numpy().astype(np.float64))
n2t = lambda x, device=None, dtype=None: torch.as_tensor(
    x, device=device, dtype=dtype
)

spfmt = "csc"
sphcat = lambda xs: sp.hstack(xs, format=spfmt)
spvcat = lambda xs: sp.vstack(xs, format=spfmt)
spmat = lambda x: sp.csc_matrix(x)
speye = lambda n: sp.eye(n, format=spfmt)
spzeros = lambda m, n: spmat((m, n))
spdiags = lambda x: sp.diags(x, format=spfmt)
spkron = lambda a, b: sp.kron(a, b, format=spfmt)


class MSVM:
    def __init__(self, k=10, e=784):
        self.k, self.e = k, e

    def _split_W(self, V):
        W = V[: self.e * self.k].reshape((self.k, self.e)).transpose(-2, -1)
        return W

    def _split_vars(self, V, n):
        assert V.numel() == self.k * self.e + 2 * n + n * self.k
        # W = V[: self.e * self.k].reshape((self.k, self.e)).transpose(-2, -1)
        W = self._split_W(V)
        Zet = V[W.numel() : W.numel() + n]
        Lam = V[
            W.numel() + Zet.numel() : W.numel() + Zet.numel() + self.k * n + n
        ]
        return W, Zet, Lam

    def pred(self, V, Z, *params):
        assert Z.ndim == 2
        W = V[: self.e * self.k].reshape((self.k, self.e)).transpose(-2, -1)
        return Z @ W

    def _generate_problem_data(self, Z, Y, *params):
        assert Z.ndim == 2 and Y.ndim == 2
        (gam,) = params
        (n, e), k = Z.shape, Y.shape[-1]
        assert self.k == k and self.e == e
        opts = dict(device=Z.device, dtype=Z.dtype)

        A1 = torch.kron(torch.eye(k, **opts), Z)
        A2 = (
            (Y[..., None] * Z[..., None, :])
            .reshape((Y.shape[0], -1))
            .tile((k, 1))
        )

        A1_ = spmat(t2n(A1))
        A2_ = spmat(t2n(A2))
        Aa_ = sphcat([A1_ - A2_, -spkron(np.ones((k, 1)), speye(n))])
        Ab_ = sphcat([spzeros(n, e * k), -sp.eye(n)])
        A_ = spvcat([Aa_, Ab_])

        P_ = spdiags(np.concatenate([np.ones(k * e), 0.0 * np.ones(n)]))

        D = (
            (Y[None, ...].argmax(-1) == torch.arange(k, **opts)[..., None])
            .reshape(-1)
            .to(Z.dtype)
        )
        # b_ = np.concatenate([t2n(D) - 1.0, np.zeros(n)])
        # q_ = np.concatenate([np.zeros(k * e), gam * np.ones(n)])
        b = torch.cat([D - 1.0, torch.zeros(n, **opts)])
        q = torch.cat(
            # [torch.zeros(k * e, **opts), (10.0 ** gam) * torch.ones(n, **opts)]
            # [torch.zeros(k * e, **opts), gam.abs() * torch.ones(n, **opts)]
            [torch.zeros(k * e, **opts), gam * torch.ones(n, **opts)]
        )

        return P_, q, A_, b

    def solve(self, Z, Y, *params, method="cvx", verbose=False):
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        q, b = t2n(q), t2n(b)
        if method == "cvx":
            x = cp.Variable(A.shape[-1])
            obj = cp.Minimize(0.5 * cp.sum(cp.quad_form(x, P)) + q @ x)
            cstr = [A @ x <= b]
            prob = cp.Problem(obj, cstr)
            prob.solve(cp.GUROBI, verbose=verbose)
            assert prob.status in ["optimal", "optimal_inaccurate"]
            x, lam = x.value, cstr[0].dual_value
        elif method == "osqp":
            P, A = sp.csc_matrix(P), sp.csc_matrix(A)
            l = -np.infty * np.ones(A.shape[0])
            global m
            m = osqp.OSQP()
            m.setup(P=P, q=q, l=l, u=b, A=A, verbose=verbose, polish=False)
            results = m.solve()
            assert results.info.status == "solved"
            x, lam = results.x, results.y
        return n2t(np.concatenate([x, lam]), device=Z.device, dtype=Z.dtype)

    def grad(self, V, Z, Y, *params):
        assert Z.ndim == 2 and Y.ndim == 2
        opts = dict(device=Z.device, dtype=Z.dtype)
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        n = Z.shape[-2]

        x, lam = t2n(V[: self.k * self.e + n]), t2n(V[self.k * self.e + n :])
        dL = n2t(P @ x, **opts) + q + n2t(A.T @ lam, **opts)
        cstr = n2t(lam * (A @ x), **opts) - n2t(lam, **opts) * b

        # x, lam = V[: self.k * self.e + n], V[self.k * self.e + n :]
        # P, A = n2t(P.toarray(), **opts), n2t(A.toarray(), **opts)
        # dL = P @ x + q + A.T @ lam
        # cstr = lam * (A @ x) - lam * b

        return torch.cat([dL, cstr])

    def hess(self, V, Z, Y, *params):
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        q, b = t2n(q), t2n(b)
        n = Z.shape[-2]
        x, lam = t2n(V[: self.k * self.e + n]), t2n(V[self.k * self.e + n :])
        K11, K12 = P, A.T
        K21 = spdiags(lam) @ A
        K22 = spdiags(A @ x - b)
        K = sphcat([spvcat([K11, K21]), spvcat([K12, K22])])
        return K

    def Dzk_solve(self, V, Z, Y, *params, rhs=None, T=False, diag_reg=None):
        diag_reg = 1e-7
        K = self.hess(V, Z, Y, *params)
        if diag_reg is not None:
            K = K + diag_reg * speye(K.shape[-1])
        rhs = t2n(rhs)
        ret = spla.spsolve(K, rhs) if T == False else spla.spsolve(K.T, rhs)
        return n2t(ret, device=Z.device, dtype=Z.dtype)


def loss_fn(V, *params):
    global OPT, Zts, Yts
    # return torch.nn.CrossEntropyLoss()(OPT.pred(V, Zts), Yts.argmax(-1))
    Yp = OPT.pred(V, Zts)
    return (
        -torch.sum(Yts * Yp) + torch.sum(torch.logsumexp(Yp, 1))
    ) / Zts.shape[-2]
    # return CE.fval(OPT._split_W(V), Zts, Yts, 0.0)


def Hz_fn(V, *params):
    global OPT, Zts, Yts
    W = OPT._split_W(V)
    Yp_aug = OPT.pred(V, Zts)
    s = torch.softmax(Yp_aug, -1)
    Ds = -s[..., :, None] @ s[..., None, :] + torch.diag_embed(s[..., :], 0)
    H = (
        torch.einsum(
            "bijkl,bijkl,bijkl->ijkl",
            Ds[..., None, :, None, :],
            Zts[..., :, None, None, None],
            Zts[..., None, None, :, None],
        )
        / Zts.shape[-2]
    )
    H = H.transpose(-1, -2)
    H = H.transpose(1, 0)
    H = H.reshape((W.numel(), W.numel()))
    H = torch.block_diag(
        H,
        torch.zeros(
            (V.numel() - W.numel(),) * 2, device=Zts.device, dtype=Zts.dtype
        ),
    )
    # H2 = HESSIAN_DIAG(lambda z: loss_fn(z, *params), V)[0]

    # plt_fn = lambda a: plt.imshow(a)

    # plt.figure()
    # plt.title("H")
    # plt_fn(H.cpu().numpy())
    # plt.colorbar()
    # plt.tight_layout()

    # plt.figure()
    # plt.title("H2")
    # plt_fn(H2.cpu().numpy())
    # plt.colorbar()
    # plt.tight_layout()

    # plt.figure()
    # plt.title("||H2 - H||")
    # plt_fn((H2 - H).abs().cpu().numpy())
    # plt.colorbar()
    # plt.tight_layout()

    # plt.figure()
    # plt.title("H2 / H")
    # plt_fn((H2 / H).cpu().numpy())
    # plt.colorbar()
    # plt.tight_layout()

    # print("err = %9.4e" % ((H - H2).norm() / H2.norm()).cpu())

    # plt.show()
    return H


if __name__ == "__main__":
    global OPT, Zts, Yts

    device, dtype = "cpu", torch.float64
    opts = dict(device=device, dtype=dtype)
    Xtr, Ytr = mnist.train["images"], mnist.train["labels"]
    Xts, Yts = mnist.test["images"], mnist.test["labels"]

    r = np.random.randint(Xtr.shape[0], size=(200,))

    Xtr = n2t(Xtr[r, :], device=device).to(dtype)
    Ytr = torch.nn.functional.one_hot(
        n2t(Ytr[r], device=device).to(torch.long), 10
    ).to(dtype)

    Xts = n2t(Xts, device=device).to(dtype)
    Yts = torch.nn.functional.one_hot(
        n2t(Yts, device=device).to(torch.long), 10
    ).to(dtype)

    Ztr = torch.cat([Xtr[:, :1] ** 0, Xtr], -1)  # [:, :500]
    Zts = torch.cat([Xts[:, :1] ** 0, Xts], -1)  # [:, :500]

    OPT = MSVM(k=10, e=Ztr.shape[-1])

LOSSES_FNAME = "data/losses2.pkl.gz"
OPTHIST_FNAME = "data/opt_hist.pkl.gz"

if __name__ == "__main__":
    # gam = torch.as_tensor([-8.0], device=device, dtype=dtype)
    # gam = torch.as_tensor([-20.0], device=device, dtype=dtype)
    gam = torch.as_tensor([1e-2], device=device, dtype=dtype)
    params = (gam,)
    k_fn = lambda z, *params: OPT.grad(z, Ztr, Ytr, *params)
    opt_fn = lambda *params: OPT.solve(Ztr, Ytr, *params, method="cvx")
    Dzk_solve = lambda z, *params, rhs=None, T=False: OPT.Dzk_solve(
        z, Ztr, Ytr, *params, rhs=rhs, T=T
    )
    optimizations = dict(Dzk_solve_fn=Dzk_solve, Hz_fn=Hz_fn)
    f_fn, g_fn, h_fn = generate_fns(
        loss_fn, opt_fn, k_fn, optimizations=optimizations
    )
    # h_fn_ = h_fn
    # def h_fn(*params):
    #    H = h_fn_(*params)
    #    #print(torch.sort(torch.linalg.eigvals(H).real)[0][[0, -1]])
    #    print(H)
    #    return H

if False and __name__ == "__main__":

    method = "sqp"
    opt_opts = dict(verbose=True, full_output=True)
    fns = [f_fn, g_fn] if method != "sqp" else [f_fn, g_fn, h_fn]
    agd_opts = dict(max_it=100, ai=1e-1, af=1e-2)
    lbfgs_opts = dict(max_it=10, lr=1e-1)
    sqp_opts = dict(max_it=10, reg0=1e-9, ls_pts_nb=1, force_step=True)
    if method == "agd":
        minimize_fn, opt_opts = minimize_agd, dict(opt_opts, **agd_opts)
    elif method == "lbfgs":
        minimize_fn, opt_opts = minimize_lbfgs, dict(opt_opts, **lbfgs_opts)
    elif method == "sqp":
        minimize_fn, opt_opts = minimize_sqp, dict(opt_opts, **sqp_opts)

    gam, gam_hist = minimize_fn(*fns, gam, **opt_opts)
    gam_hist_losses = [loss_fn(opt_fn(gam), gam) for gam in gam_hist]

    gam, gam_hist = t2n(gam), [t2n(gam_) for gam_ in gam_hist]
    gam_hist_losses = [t2n(loss) for loss in gam_hist_losses]

    with gzip.open(OPTHIST_FNAME, "wb") as fp:
        pickle.dump((gam, gam_hist, gam_hist_losses), fp)

if True and __name__ == "__main__":
    # scan through gams ##########################
    # gams, losses, hs = torch.linspace(-9, 1, 100), []
    gams, losses, gs, hs = torch.logspace(-9, 1, 100, **opts), [], [], []
    for gam in tqdm(gams):
        V = OPT.solve(Ztr, Ytr, gam, method="cvx")
        losses.append(float(t2n(loss_fn(V, gam))))
        gs.append(g_fn(gam))
        hs.append(h_fn(gam))
        tqdm.write(
            "(%9.4e, %9.4e, %9.4e, %9.4e -> %9.4e)"
            % (float(gam), losses[-1], gs[-1], hs[-1], gs[-1] / hs[-1])
        )
    losses = torch.as_tensor(losses, **opts)
    pdb.set_trace()

    with gzip.open(LOSSES_FNAME, "wb") as fp:
        pickle.dump((t2n(gams), t2n(losses)), fp)

if True and __name__ == "__main__":
    with gzip.open(LOSSES_FNAME, "rb") as fp:
        gams, losses = pickle.load(fp)

    with gzip.open(OPTHIST_FNAME, "rb") as fp:
        gam, gam_hist, gam_hist_losses = pickle.load(fp)

    gams, gam_hist = (10.0 ** np.array(gams)), (10 ** np.array(gam_hist))
    plt.plot(gams, losses, color="C0")
    plt.plot(gam_hist, gam_hist_losses, color="C1")
    plt.scatter(gam_hist, gam_hist_losses, color="C1")
    plt.scatter(gam_hist[-1], gam_hist_losses[-1], color="black")
    plt.ylabel("$\\ell_\\operatorname{test}$")
    plt.xlabel("$\\gamma$")
    plt.title("SVM Tuning")
    plt.tight_layout()
    ax = plt.gca()

    a = plt.axes([0.3, 0.3, 0.6, 0.6])
    p1 = list(zip(*[(g, l) for (g, l) in zip(gams, losses) if g < 3e-6]))
    p2 = list(
        zip(*[(g, l) for (g, l) in zip(gam_hist, gam_hist_losses) if g < 3e-6])
    )
    plt.plot(*p1, color="C0")
    # plt.scatter(*p1, color="C0")
    plt.plot(*p2, color="C1")
    plt.scatter(*p2, color="C1")
    plt.ylabel("$\\ell_\\operatorname{test}$")
    plt.xlabel("$\\gamma$")
    plt.gca().get_xaxis().get_major_formatter().set_powerlimits((0, 0))

    # ax.add_patch(Rectangle([, 1.0], 5, 3, fc="y", lw=10))
    plt.savefig("figs/gam_optim.png", dpi=200)

    plt.show()


if False and __name__ == "__main__":
    gam = 1e-2
    t = time.perf_counter()
    V = OPT.solve(Ztr, Ytr, gam, method="osqp")
    print("Elapsed %9.4e s" % (time.perf_counter() - t))
    t = time.perf_counter()
    K = OPT.hess(V, Ztr, Ytr, gam)
    print("Elapsed %9.4e s" % (time.perf_counter() - t))

if True and __name__ == "__main__":
    acc_fn = (
        lambda V, Z, Y: (Y.argmax(-1) == OPT.pred(V, Z).argmax(-1))
        .to(dtype)
        .mean()
    )
    acc_tr, acc_ts = acc_fn(V, Ztr, Ytr), acc_fn(V, Zts, Yts)

    print("Train acc = %5.1f%%" % (acc_tr * 1e2))
    print("Test acc =  %5.1f%%" % (acc_ts * 1e2))

    # pdb.set_trace()
