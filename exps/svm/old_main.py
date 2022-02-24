import os, sys, pdb, time, gzip, pickle, math
from collections import OrderedDict as odict
from copy import copy

import torch, osqp
import cvxpy as cp, numpy as np, matplotlib.pyplot as plt, line_profiler as lp
import scipy.sparse as sp, scipy.sparse.linalg as spla
from matplotlib.patches import Rectangle
from tqdm import tqdm

import header

from implicit.interface import init

jaxm = init(dtype=np.float64, device="cpu")
# jaxm = init(dtype=np.float32, device="cuda")
from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd
from implicit.implicit import implicit_jacobian, implicit_hessian, generate_fns
from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit.pca import visualize_landscape
from implicit.utils import n2j, j2n

from objs import LS, CE

import mnist
#import fashion as mnist
from utils import scale_down

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
        W = jaxm.t(V[: self.e * self.k].reshape((self.k, self.e)))
        return W

    def _split_vars(self, V, n):
        assert V.size == self.k * self.e + 2 * n + n * self.k
        W = self._split_W(V)
        Zet = V[W.size : W.size + n]
        Lam = V[W.size + Zet.size : W.size + Zet.size + self.k * n + n]
        return W, Zet, Lam

    def pred(self, V, Z, *params):
        assert Z.ndim == 2
        W = jaxm.t(V[: self.e * self.k].reshape((self.k, self.e)))
        return Z @ W

    def _generate_problem_data(self, Z, Y, *params):
        assert Z.ndim == 2 and Y.ndim == 2
        (gam,) = params
        (n, e), k = Z.shape, Y.shape[-1]
        assert self.k == k and self.e == e

        A1 = jaxm.kron(jaxm.eye(k), Z)
        A2 = (
            (Y[..., None] * Z[..., None, :])
            .reshape((Y.shape[0], -1))
            .tile((k, 1))
        )

        # A1_ = spmat(j2n(A1))
        # A2_ = spmat(j2n(A2))
        # Aa_ = sphcat([A1_ - A2_, -spkron(np.ones((k, 1)), speye(n))])
        # Ab_ = sphcat([spzeros(n, e * k), -sp.eye(n)])
        # A_ = spvcat([Aa_, Ab_])

        Aa = jaxm.hstack([A1 - A2, -jaxm.kron(jaxm.ones((k, 1)), jaxm.eye(n))])
        Ab = jaxm.hstack([jaxm.zeros((n, e * k)), -jaxm.eye(n)])
        A_ = jaxm.vstack([Aa, Ab])

        # P_ = spdiags(np.concatenate([np.ones(k * e), 0.0 * np.ones(n)]))
        P_ = jaxm.diag(jaxm.cat([jaxm.ones(k * e), 0.0 * jaxm.ones(n)]))

        D = (Y[None, ...].argmax(-1) == jaxm.arange(k)[..., None]).reshape(-1)
        b = jaxm.cat([D - 1.0, jaxm.zeros(n)])
        q = jaxm.cat([jaxm.zeros(k * e), (10.0 ** gam) * jaxm.ones(n)])

        return P_, q, A_, b

    def solve(self, Z, Y, *params, method="cvx", verbose=False):
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        P, A, q, b = j2n(P), j2n(A), j2n(q), j2n(b)
        P, A = spmat(P), spmat(A)
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
        return n2j(np.concatenate([x, lam]))

    def grad(self, V, Z, Y, *params):
        assert Z.ndim == 2 and Y.ndim == 2
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        n = Z.shape[-2]

        x, lam = V[: self.k * self.e + n], V[self.k * self.e + n :]
        dL = P @ x + q + jaxm.t(A) @ lam
        cstr = lam * (A @ x) - lam * b

        return jaxm.cat([dL, cstr])

    def hess(self, V, Z, Y, *params):
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        # q, b = j2n(q), j2n(b)
        n = Z.shape[-2]
        # x, lam = j2n(V[: self.k * self.e + n]), j2n(V[self.k * self.e + n :])
        x, lam = V[: self.k * self.e + n], V[self.k * self.e + n :]
        # K11, K12 = P, A.T
        # K21 = spdiags(lam) @ A
        # K22 = spdiags(A @ x - b)
        # K = sphcat([spvcat([K11, K21]), spvcat([K12, K22])])
        K11, K12 = P, jaxm.t(A)
        K21 = lam.reshape((-1, 1)) * A
        K22 = jaxm.diag(A @ x - b)
        K = jaxm.hstack([jaxm.vstack([K11, K21]), jaxm.vstack([K12, K22])])
        return K

    def Dzk_solve(self, V, Z, Y, *params, rhs=None, T=False, diag_reg=math.nan):
        K = self.hess(V, Z, Y, *params)
        if diag_reg is not None:
            if math.isnan(diag_reg):
                diag_reg = 5 * jaxm.finfo(jaxm.zeros(()).dtype).eps
            # K = K + diag_reg * speye(K.shape[-1])
            K = K + diag_reg * jaxm.eye(K.shape[-1])
        # rhs = j2n(rhs)
        # ret = spla.spsolve(K, rhs) if T == False else spla.spsolve(K.T, rhs)
        # return n2j(ret)
        ret = (
            jaxm.linalg.solve(K, rhs)
            if T == False
            else jaxm.linalg.solve(K.T, rhs)
        )
        return ret


def loss_fn(V, *params):
    global OPT, Zts, Yts
    Yp = OPT.pred(V, Zts)
    #return -jaxm.mean(jaxm.log(jaxm.sum(jaxm.softmax(Yp, -1) * Yts, -1)))
    return jaxm.mean(-Yp[..., jaxm.argmax(Yts, -1)] + jaxm.nn.logsumexp(Yp, -1))


@jaxm.jit
def Hz_fn(V, *params):
    return jaxm.hessian(loss_fn)(V, *params)

    # global OPT, Zts, Yts
    # W = OPT._split_W(V)
    # ret = jaxm.hessian(loss_fn_)(W)
    # return ret


#    Yp_aug = OPT.pred(V, Zts)
#    s = jaxm.softmax(Yp_aug, -1)
#    Ds = -s[..., :, None] @ s[..., None, :] + jaxm.diag_embed(s[..., :], 0)
#    H = (
#        jaxm.einsum(
#            "bijkl,bijkl,bijkl->ijkl",
#            Ds[..., None, :, None, :],
#            Zts[..., :, None, None, None],
#            Zts[..., None, None, :, None],
#        )
#        / Zts.shape[-2]
#    )
#    H = jaxm.swapaxes(H, -1, -2)
#    H = jaxm.swapaxes(H, 1, 0)
#    H = H.reshape((W.size, W.size))
#    H = jaxm.scipy.linalg.block_diag(H, jaxm.zeros((V.size - W.size,) * 2))
#    return H

actions = [z for z in sys.argv[1:]]

# prepare data #################################################3
if __name__ == "__main__":
    global OPT, Zts, Yts
    dtype = jaxm.zeros(()).dtype

    Xtr, Ytr = mnist.train["images"], mnist.train["labels"]
    Xts, Yts = mnist.test["images"], mnist.test["labels"]

    # r = np.random.randint(Xtr.shape[0], size=(200,))
    r = np.random.randint(Xtr.shape[0], size=(2000,))
    Xtr = n2j(Xtr[r, :]).astype(dtype)
    Ytr = jaxm.nn.one_hot(n2j(Ytr[r]), 10).astype(dtype)

    r = np.random.randint(Xts.shape[0], size=(4000,))
    #r = np.arange(Xts.shape[0])
    Xts = n2j(Xts[r, :]).astype(dtype)
    Yts = jaxm.nn.one_hot(n2j(Yts[r]), 10).astype(dtype)

    Ztr = jaxm.cat([Xtr[:, :1] ** 0, scale_down(Xtr)], -1)  # [:, :500]
    Zts = jaxm.cat([Xts[:, :1] ** 0, scale_down(Xts)], -1)  # [:, :500]

    OPT = MSVM(k=10, e=Ztr.shape[-1])

LOSSES_FNAME = "data/losses.pkl.gz"
DIAGREG_FNAME = "data/diagreg.pkl.gz"
OPTHIST_FNAME = "data/opt_hist.pkl.gz"

# prepare the functions ###########################################
if __name__ == "__main__":
    gam = jaxm.array([-8.0])
    params = (gam,)
    k_fn = lambda z, *params: OPT.grad(z, Ztr, Ytr, *params)
    opt_fn = lambda *params: OPT.solve(Ztr, Ytr, *params, method="cvx")
    Dzk_solve = lambda z, *params, rhs=None, T=False: OPT.Dzk_solve(
        z, Ztr, Ytr, *params, rhs=rhs, T=T
    )
    optimizations = dict(Dzk_solve_fn=Dzk_solve, Hz_fn=Hz_fn)
    f_fn, g_fn, h_fn = generate_fns(
        loss_fn, opt_fn, k_fn, optimizations=optimizations, jit=True
    )

    # def main_():
    #    print(jaxm.sum(f_fn(*params)))
    #    print(jaxm.sum(g_fn(*params)))
    #    print(jaxm.sum(h_fn(*params)))

    # main_()

    # LP = lp.LineProfiler()
    # LP.add_function(f_fn.fn)
    # LP.add_function(g_fn.fn)
    # LP.add_function(h_fn.fn)
    # LP.add_function(main_)
    # main = LP.wrap_function(main_)
    # main()
    # LP.print_stats(output_unit=1e-3)


def g_alt_fn(*params):
    print("Called")
    z = OPT.solve(Ztr, Ytr, *params, method="cvx")
    k = OPT.grad(z, Ztr, Ytr, *params)
    # H_ = grad(lambda z: OPT.grad(z, Ztr, Ytr, *params), verbose=True)(z)
    H = jaxm.array(OPT.hess(z, Ztr, Ytr, *params).toarray())
    Dg = JACOBIAN(lambda z: loss_fn(z, *params), z)
    J = JACOBIAN(lambda *params: OPT.grad(z, Ztr, Ytr, *params), *params)
    v = -jaxm.linalg.solve(H.T, Dg)
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
    fns = dict(agd=[f_fn, g_fn], lbfgs=[f_fn, g_fn], sqp=[f_fn, g_fn, h_fn])

    hist, t_stamp = dict(), time.time()
    solver, cb_it = None, 0

    def cb_fn(*args, **kwargs):
        global solver, hist, t_stamp, cb_it
        cb_it += 1
        t_inc = time.time() - t_stamp
        z = opt_fn(*args)
        Yp = OPT.pred(z, Zts)
        acc = jaxm.mean((Yp.argmax(-1) == Yts.argmax(-1)))
        tqdm.write("Accuracy: %5.1f%%" % float(1e2 * acc))
        hist[solver]["loss"].append(float(loss_fn(z, gam)))
        hist[solver]["acc"].append(float(1e2 * acc))
        hist[solver]["it"].append(cb_it)
        hist[solver]["fns"].append(len(f_fn.cache.keys()))
        hist[solver]["t"].append(t_inc)
        t_stamp = time.time()

    agd_opts = dict(max_it=100, ai=1e-1, af=1e-2, callback_fn=cb_fn)
    lbfgs_opts = dict(max_it=10, lr=1e-1, callback_fn=cb_fn)
    sqp_opts = dict(
        max_it=10, reg0=1e-7, ls_pts_nb=1, force_step=True, callback_fn=cb_fn
    )

    minimize_fn = dict(agd=minimize_agd, lbfgs=minimize_lbfgs, sqp=minimize_sqp)
    opts_map = dict(agd=agd_opts, lbfgs=lbfgs_opts, sqp=sqp_opts)

    for solver in ["sqp", "lbfgs", "agd"]:
        keys = copy(list(f_fn.cache.keys()))
        assert (
            len(f_fn.cache.keys())
            == len(g_fn.cache.keys())
            == len(h_fn.cache.keys())
        )
        for k in keys:
            del f_fn.cache[k]
        assert len(f_fn.cache.keys()) == 0
        assert len(g_fn.cache.keys()) == 0
        assert len(h_fn.cache.keys()) == 0
        hist[solver] = dict(acc=[], fns=[], it=[], t=[], loss=[])
        opt_opts_ = dict(opt_opts, **opts_map[solver])
        _, gam_hist = minimize_fn[solver](*fns[solver], gam, **opt_opts_)
        t = time.time()
        # gam_hist_losses = [loss_fn(opt_fn(gam), gam) for gam in gam_hist]
        gam_hist_losses = [f_fn(gam) for gam in gam_hist]
        print("loss eval takes %9.4e" % (time.time() - t))

        gam_hist = [j2n(gam_) for gam_ in gam_hist]
        gam_hist_losses = [j2n(loss) for loss in gam_hist_losses]

        hist[solver]["gam"] = j2n(gam)
        hist[solver]["gam_hist"] = gam_hist
        hist[solver]["gam_hist_losses"] = gam_hist_losses

    with gzip.open("data/opt_hist.pkl.gz", "wb") as fp:
        pickle.dump(hist, fp)

# scan through gamma values #######################################
if "scan" in actions:
    compute_grads = True
    gams, losses, gs, hs, accs = jaxm.linspace(-9, 1, 100), [], [], [], []
    for gam in tqdm(gams):
        V = OPT.solve(Ztr, Ytr, gam, method="cvx")
        losses.append(j2n(loss_fn(V, gam)))
        Yp = OPT.pred(V, Zts)
        acc = jaxm.mean((jaxm.argmax(Yp, -1) == jaxm.argmax(Yts, -1)))
        accs.append(j2n(acc))
        if compute_grads:
            gs.append(j2n(g_fn(gam)))
            hs.append(j2n(h_fn(gam)))
            tqdm.write(
                "(%9.4e, %9.4e, %9.4e, %9.4e, %6.2f"
                % (gam, losses[-1], gs[-1], hs[-1], 1e2 * accs[-1])
            )
        else:
            tqdm.write(
                "(%9.4e, %9.4e, %6.2f)"
                % (float(gam), losses[-1], 1e2 * accs[-1])
            )
        with gzip.open(LOSSES_FNAME, "wb") as fp:
            pickle.dump(
                (
                    np.array(gams),
                    np.array(losses),
                    np.array(gs),
                    np.array(hs),
                    np.array(accs),
                ),
                fp,
            )

## scan through gamma values #######################################
if "compare" in actions:
    results = odict()
    diag_regs = jaxm.logspace(-15, -5, 10)
    for diag_reg in tqdm(diag_regs):
        Dzk_solve = lambda z, *params, rhs=None, T=False: OPT.Dzk_solve(
            z, Ztr, Ytr, *params, rhs=rhs, T=T, diag_reg=float(diag_reg.cpu())
        )
        # optimizations = dict(Dzk_solve_fn=Dzk_solve, Hz_fn=Hz_fn)
        optimizations = dict(Dzk_solve_fn=Dzk_solve)
        f_fn, g_fn, h_fn = generate_fns(
            loss_fn, opt_fn, k_fn, optimizations=optimizations, jit=False
        )

        gams, losses, gs, hs = jaxm.linspace(-8, -5, 10), [], [], []
        for gam in tqdm(gams):
            V = OPT.solve(Ztr, Ytr, gam, method="cvx")
            losses.append(float(j2n(loss_fn(V, gam))))
            gs.append(g_fn(gam))
            # gs_alt.append(g_alt_fn(gam))
            hs.append(h_fn(gam))
            tqdm.write(
                "(%9.4e, %9.4e, %9.4e, %9.4e -> %9.4e)"
                % (float(gam), losses[-1], gs[-1], hs[-1], gs[-1] / hs[-1])
            )
        losses, gs, hs = [jaxm.array(z) for z in [losses, gs, hs]]
        gams, losses, gs, hs = [
            np.array(j2n(z)) for z in [gams, losses, gs, hs]
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

if "acc" in actions:
    with gzip.open(LOSSES_FNAME, "rb") as fp:
        gams, ls, gs, hs = pickle.load(fp)
    order = np.argsort(ls)
    gams, ls, gs, hs = [z[order] for z in [gams, ls, gs, hs]]

    accs = []
    print("%6s --- %6s" % ("gam", "acc"))
    for (i, gam) in enumerate(tqdm(gams)):
        V = opt_fn(n2j(gam))
        Yp = OPT.pred(V, Zts)
        acc = jaxm.mean((jaxm.argmax(Yp, -1) == jaxm.argmax(Yts, -1)))
        accs.append(j2n(acc))
        tqdm.write("%6.2f --- %6.2f" % (gam, 1e2 * acc))
        with gzip.open("data/accs.pkl.gz", "wb") as fp:
            pickle.dump(
                (
                    np.array(gams),
                    np.array(ls),
                    np.array(gs),
                    np.array(hs),
                    np.array(accs),
                ),
                fp,
            )
    order = np.argsort(gams)
    accs = np.array(accs)[order]
    gams, ls, gs, hs = [z[order] for z in [gams, ls, gs, hs]]

    plt.plot(gams, 1e2 * accs)
    plt.xlabel("$\\gamma$")
    plt.ulabel("Test Accuracy")
    plt.tight_layout()
    plt.show()


# visualize loss landscape ########################################
if "visualize" in actions:
    with gzip.open(LOSSES_FNAME, "rb") as fp:
        gams, losses, gs, hs, accs = pickle.load(fp)
    plt.plot(gams, losses, color="C0")
    grads = np.diff(losses) / np.diff(gams)
    for (gam, loss, g, h) in zip(gams, losses, gs, hs):
        dgam = 1.5e-1
        grad = np.interp(gam, gams[:-1], grads)
        plt.plot([gam, gam + dgam], [loss, loss + g * dgam], color="C1")
    # try:
    #    with gzip.open(OPTHIST_FNAME, "rb") as fp:
    #        gam, gam_hist, gam_hist_losses = pickle.load(fp)
    #        gam_hist = np.array(gam_hist)
    #        plt.plot(gam_hist, gam_hist_losses, color="C1")
    #        plt.scatter(gam_hist, gam_hist_losses, color="C1")
    #        plt.scatter(gam_hist[-1], gam_hist_losses[-1], color="black")
    # except FileNotFoundError:
    #    pass

    plt.ylabel("$\\ell_\\operatorname{test}$")
    plt.xlabel("$\\gamma$")
    plt.title("SVM Tuning")
    plt.ylim([np.min(losses) - 0.1, np.max(losses) + 0.1])
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

    plt.figure()
    plt.plot(gams, accs)
    plt.xlabel("$\\gamma$")
    plt.ylabel("Test Accuracy")

    plt.show()
