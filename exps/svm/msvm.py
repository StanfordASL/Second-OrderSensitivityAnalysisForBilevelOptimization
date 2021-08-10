import sys, pdb, time, gzip, pickle
from collections import OrderedDict as odict

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
from implicit.diff import JACOBIAN, HESSIAN_DIAG, torch_grad as grad
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
            [torch.zeros(k * e, **opts), (10.0 ** gam) * torch.ones(n, **opts)]
            # [torch.zeros(k * e, **opts), gam.abs() * torch.ones(n, **opts)]
            # [torch.zeros(k * e, **opts), gam * torch.ones(n, **opts)]
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

    def Dzk_solve(self, V, Z, Y, *params, rhs=None, T=False, diag_reg=1e-5):
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
    return H


actions = [z for z in sys.argv[1:]]


# prepare data #################################################3
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

LOSSES_FNAME = "data/losses.pkl.gz"
DIAGREG_FNAME = "data/diagreg.pkl.gz"
OPTHIST_FNAME = "data/opt_hist.pkl.gz"

# prepare the functions ###########################################
if __name__ == "__main__":
    gam = torch.as_tensor([-8.0], device=device, dtype=dtype)
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


def g_alt_fn(*params):
    print("Called")
    z = OPT.solve(Ztr, Ytr, *params, method="cvx")
    k = OPT.grad(z, Ztr, Ytr, *params)
    # H_ = grad(lambda z: OPT.grad(z, Ztr, Ytr, *params), verbose=True)(z)
    H = torch.as_tensor(OPT.hess(z, Ztr, Ytr, *params).toarray(), **opts)
    Dg = JACOBIAN(lambda z: loss_fn(z, *params), z)
    J = JACOBIAN(lambda *params: OPT.grad(z, Ztr, Ytr, *params), *params)
    v = -torch.linalg.solve(H.T, Dg)
    g = v @ J
    return g


if "check" in actions:
    g = g_alt_fn(*params)
    g_ = g_fn(*params)
    pdb.set_trace()

# optimize starting from a bad guess ##############################
if "optimize" in actions:
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

# scan through gamma values #######################################
if "scan" in actions:
    gams, losses, gs, hs = torch.linspace(-9, 1, 100, **opts), [], [], []
    gs_alt = []
    for gam in tqdm(gams):
        V = OPT.solve(Ztr, Ytr, gam, method="cvx")
        losses.append(float(t2n(loss_fn(V, gam))))
        gs.append(g_fn(gam))
        # gs_alt.append(g_alt_fn(gam))
        hs.append(h_fn(gam))
        tqdm.write(
            "(%9.4e, %9.4e, %9.4e, %9.4e -> %9.4e)"
            % (float(gam), losses[-1], gs[-1], hs[-1], gs[-1] / hs[-1])
        )
    losses, gs, hs = [torch.as_tensor(z, **opts) for z in [losses, gs, hs]]
    gams, losses, gs, hs = [np.array(t2n(z)) for z in [gams, losses, gs, hs]]

    # gs_alt = torch.as_tensor(gs_alt, **opts)
    # gs_alt = np.array(t2n(gs_alt))

    # with gzip.open(LOSSES_FNAME, "wb") as fp:
    #    pickle.dump((gams, losses, gs, hs), fp)

    with gzip.open(LOSSES_FNAME + "2", "wb") as fp:
        pickle.dump((gams, losses, gs, gs_alt, hs), fp)

## scan through gamma values #######################################
if "compare" in actions:
    results = odict()
    diag_regs = torch.logspace(-15, -5, 10, **opts)
    for diag_reg in tqdm(diag_regs):
        Dzk_solve = lambda z, *params, rhs=None, T=False: OPT.Dzk_solve(
            z, Ztr, Ytr, *params, rhs=rhs, T=T, diag_reg=float(diag_reg.cpu())
        )
        optimizations = dict(Dzk_solve_fn=Dzk_solve, Hz_fn=Hz_fn)
        f_fn, g_fn, h_fn = generate_fns(
            loss_fn, opt_fn, k_fn, optimizations=optimizations
        )

        gams, losses, gs, hs = torch.linspace(-8, -5, 10, **opts), [], [], []
        for gam in tqdm(gams):
            V = OPT.solve(Ztr, Ytr, gam, method="cvx")
            losses.append(float(t2n(loss_fn(V, gam))))
            gs.append(g_fn(gam))
            # gs_alt.append(g_alt_fn(gam))
            hs.append(h_fn(gam))
            tqdm.write(
                "(%9.4e, %9.4e, %9.4e, %9.4e -> %9.4e)"
                % (float(gam), losses[-1], gs[-1], hs[-1], gs[-1] / hs[-1])
            )
        losses, gs, hs = [torch.as_tensor(z, **opts) for z in [losses, gs, hs]]
        gams, losses, gs, hs = [
            np.array(t2n(z)) for z in [gams, losses, gs, hs]
        ]
        results[float(diag_reg.cpu())] = (gams, losses, gs, hs)

    with gzip.open(DIAGREG_FNAME, "wb") as fp:
        pickle.dump(results, fp)

if "compvis" in actions:
    with gzip.open(DIAGREG_FNAME, "rb") as fp:
        results = pickle.load(fp)
    np.set_printoptions(precision=3, linewidth=100)
    vals = []
    for k in results.keys():
        gams, losses, gs, hs = results[k]
        if len(vals) == 0:
            vals.append(gams)
        vals.append(gs / hs)
    print(np.stack(vals))
    pdb.set_trace()
    pass


# visualize loss landscape ########################################
if "visualize" in actions:
    # with gzip.open(LOSSES_FNAME, "rb") as fp:
    #    gams, losses, gs, hs = pickle.load(fp)
    with gzip.open(LOSSES_FNAME + "2", "rb") as fp:
        gams, losses, gs, gs_alt, hs = pickle.load(fp)
    gams = np.array(gams)
    plt.plot(gams, losses, color="C0")
    grads = np.diff(losses) / np.diff(gams)
    #for (gam, loss, g, g_alt, h) in zip(gams, losses, gs, gs_alt, hs):
    for (gam, loss, g, h) in zip(gams, losses, gs, hs):
        dgam = 1.5e-1
        grad = np.interp(gam, gams[:-1], grads)
        plt.plot([gam, gam + dgam], [loss, loss + g * dgam], color="C1")
        #plt.plot([gam, gam + dgam], [loss, loss + g_alt * dgam], color="C2")
        # plt.plot([gam, gam + dgam], [loss, loss + grad * dgam], color="C3")
        print(grad / (g / h))
    pdb.set_trace()
    try:
        with gzip.open(OPTHIST_FNAME, "rb") as fp:
            gam, gam_hist, gam_hist_losses = pickle.load(fp)
            gam_hist = np.array(gam_hist)
            plt.plot(gam_hist, gam_hist_losses, color="C1")
            plt.scatter(gam_hist, gam_hist_losses, color="C1")
            plt.scatter(gam_hist[-1], gam_hist_losses[-1], color="black")
    except FileNotFoundError:
        pass

    plt.ylabel("$\\ell_\\operatorname{test}$")
    plt.xlabel("$\\gamma$")
    plt.title("SVM Tuning")
    plt.tight_layout()

    # ax = plt.gca()
    # a = plt.axes([0.3, 0.3, 0.6, 0.6])
    # p1 = list(zip(*[(g, l) for (g, l) in zip(gams, losses) if g < 3e-6]))
    # p2 = list(
    #    zip(*[(g, l) for (g, l) in zip(gam_hist, gam_hist_losses) if g < 3e-6])
    # )
    # plt.plot(*p1, color="C0")
    ## plt.scatter(*p1, color="C0")
    # plt.plot(*p2, color="C1")
    # plt.scatter(*p2, color="C1")
    # plt.ylabel("$\\ell_\\operatorname{test}$")
    # plt.xlabel("$\\gamma$")
    # plt.gca().get_xaxis().get_major_formatter().set_powerlimits((0, 0))

    # ax.add_patch(Rectangle([, 1.0], 5, 3, fc="y", lw=10))
    plt.savefig("figs/gam_optim.png", dpi=200)

    plt.show()
