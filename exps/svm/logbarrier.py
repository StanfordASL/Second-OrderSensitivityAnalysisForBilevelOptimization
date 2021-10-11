import os, sys, pdb, time, gzip, pickle, math
from collections import OrderedDict as odict
from copy import copy

import torch, osqp
import matplotlib.pyplot as plt, line_profiler as lp, cvxpy as cp, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from matplotlib.patches import Rectangle
from tqdm import tqdm

import header

from implicit.interface import init

jaxm = init(dtype=np.float32, device="cpu")
# jaxm = init(dtype=np.float32, device="cuda")
from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd
from implicit.implicit import implicit_jacobian, implicit_hessian, generate_fns
from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit.pca import visualize_landscape
from implicit.utils import n2j, j2n

from objs import LS, CE, OBJ

# import mnist
import fashion as mnist

from utils import scale_down

spmat = lambda x: sp.csc_matrix(x)

ALF = 1e2


class MSVM(OBJ):
    def __init__(self, k=10, e=784):
        self.k, self.e = k, e
        self.reg = 1e-15

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
        Aa = jaxm.hstack([A1 - A2, -jaxm.kron(jaxm.ones((k, 1)), jaxm.eye(n))])
        Ab = jaxm.hstack([jaxm.zeros((n, e * k)), -jaxm.eye(n)])
        A_ = jaxm.vstack([Aa, Ab])

        P_ = jaxm.diag(jaxm.cat([jaxm.ones(k * e), 0.0 * jaxm.ones(n)]))

        D = (Y[None, ...].argmax(-1) == jaxm.arange(k)[..., None]).reshape(-1)
        b = jaxm.cat([D - 1.0, jaxm.zeros(n)])
        q = jaxm.cat([jaxm.zeros(k * e), (10.0 ** gam) * jaxm.ones(n)])

        return P_, q, A_, b

    def fval(self, W, Z, Y, *params):
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        x = W.reshape(-1)
        obj = (
            jaxm.sum(x * (P @ x)) / 2
            + jaxm.sum(x * q)
            # + self.reg * jaxm.sum(x ** 2)
        )
        cstr = A @ x - b
        return obj - jaxm.mean(jaxm.log(-ALF * cstr)) / ALF

    def solve(self, Z, Y, *params, method="cvx", verbose=False):
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        P, A, q, b = j2n(P), j2n(A), j2n(q), j2n(b)
        P, A = spmat(P), spmat(A)
        if method == "cvx":
            x = cp.Variable(A.shape[-1])
            obj = (
                0.5 * cp.sum(cp.quad_form(x, P))
                + q @ x
                # + self.reg * cp.sum_squares(x)
            )
            cstr = A @ x - b
            obj = obj - cp.sum(cp.log(-ALF * cstr)) / ALF / cstr.size
            prob = cp.Problem(cp.Minimize(obj))
            prob.solve(cp.MOSEK, verbose=verbose)
            assert prob.status in ["optimal", "optimal_inaccurate"]
            x = x.value
        else:
            raise NotImplementedError
        return n2j(x)


def loss_fn(V, *params):
    global OPT, Zts, Yts
    Yp = OPT.pred(V, Zts)
    # return jaxm.mean(-Yp[..., jaxm.argmax(Yts, -1)] + jaxm.nn.logsumexp(Yp, -1))
    return jaxm.mean(jaxm.sum(-Yp * Yts, -1)) + jaxm.mean(
        jaxm.nn.logsumexp(Yp, -1)
    )


@jaxm.jit
def Hz_fn(V, *params):
    return jaxm.hessian(loss_fn)(V, *params)


actions = sys.argv[1:]


# prepare data #################################################3
if __name__ == "__main__":
    global OPT, Zts, Yts
    dtype = jaxm.zeros(()).dtype

    Xtr, Ytr = mnist.train["images"], mnist.train["labels"]
    Xts, Yts = mnist.test["images"], mnist.test["labels"]

    r = np.random.randint(Xtr.shape[0], size=(2000,))
    Xtr = n2j(Xtr[r, :]).astype(dtype)
    Ytr = jaxm.nn.one_hot(n2j(Ytr[r]), 10).astype(dtype)

    #r = np.random.randint(Xts.shape[0], size=(10 ** 4,))
    #Xts = n2j(Xts[r, :]).astype(dtype)
    #Yts = jaxm.nn.one_hot(n2j(Yts[r]), 10).astype(dtype)

    Xts = n2j(Xts).astype(dtype)
    Yts = jaxm.nn.one_hot(n2j(Yts), 10).astype(dtype)

    Ztr = jaxm.cat([Xtr[:, :1] ** 0, scale_down(Xtr, 2)], -1)
    Zts = jaxm.cat([Xts[:, :1] ** 0, scale_down(Xts, 2)], -1)

    OPT = MSVM(k=10, e=Ztr.shape[-1])

LOSSES_FNAME = "data/logbarrier_losses.pkl.gz"
DIAGREG_FNAME = "data/logbarrier_diagreg.pkl.gz"
# OPTHIST_FNAME = "data/logbarrier_opt_hist.pkl.gz"
OPTHIST_FNAME = "data/logbarrier_opt_hist"

# prepare the functions ###########################################
if __name__ == "__main__":
    gam = jaxm.array([-7.0])
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

# optimize starting from a bad guess ##############################
if "optimize" in actions:
    f, g, h = f_fn(*params), g_fn(*params), h_fn(*params)

    def main_():
        print(jaxm.sum(f_fn(*params)))
        print(jaxm.sum(g_fn(*params)))
        print(jaxm.sum(h_fn(*params)))

    main_()

    LP = lp.LineProfiler()
    LP.add_function(f_fn.fn)
    LP.add_function(g_fn.fn)
    LP.add_function(h_fn.fn)
    LP.add_function(main_)
    main = LP.wrap_function(main_)
    main()
    LP.print_stats(output_unit=1e-3)

    method = "sqp"
    opt_opts = dict(verbose=True, full_output=True)
    fns = dict(agd=[f_fn, g_fn], lbfgs=[f_fn, g_fn], sqp=[f_fn, g_fn, h_fn])

    hist, t_stamp = dict(), time.time()
    solver, cb_it = None, 0

    def cb_fn(*args, **kw):
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

    agd_opts = dict(max_it=50, ai=1e-1, af=1e-1, callback_fn=cb_fn)
    lbfgs_opts = dict(max_it=4, lr=1e-1, callback_fn=cb_fn)
    sqp_opts = dict(
        max_it=10, reg0=1e-9, ls_pts_nb=1, force_step=True, callback_fn=cb_fn
    )

    minimize_fn = dict(agd=minimize_agd, lbfgs=minimize_lbfgs, sqp=minimize_sqp)
    opts_map = dict(agd=agd_opts, lbfgs=lbfgs_opts, sqp=sqp_opts)

    for solver in ["agd", "sqp", "lbfgs"]:
        if solver not in actions:
            continue
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

        with gzip.open(
            OPTHIST_FNAME + "_".join(list(hist.keys())) + ".pkl.gz", "wb"
        ) as fp:
            pickle.dump(hist, fp)

# scan through gamma values #######################################
if "scan" in actions:
    compute_grads = True
    gams, losses, gs, hs, accs = jaxm.linspace(-9, 0, 100), [], [], [], []
    for gam in tqdm(gams):
        try:
            V = OPT.solve(Ztr, Ytr, gam, method="cvx")
            losses.append(j2n(loss_fn(V, gam)))
            Yp = OPT.pred(V, Zts)
            acc = jaxm.mean((jaxm.argmax(Yp, -1) == jaxm.argmax(Yts, -1)))
            accs.append(j2n(acc))
            if compute_grads:
                gs.append(j2n(g_fn(gam)))
                hs.append(j2n(h_fn(gam)))
        except:
            losses.append(math.nan)
            accs.append(math.nan)
            if compute_grads:
                gs.append(math.nan)
                hs.append(math.nan)
        if compute_grads:
            tqdm.write(
                "(%9.4e, %9.4e, %9.4e, %9.4e, %6.2f)"
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


# visualize loss landscape ########################################
if "visualize" in actions:
    with gzip.open(LOSSES_FNAME, "rb") as fp:
        gams, losses, gs, hs, accs = pickle.load(fp)
    gams = gams[: len(losses)]
    # mask = gams < -7.0
    # gams, losses, gs, hs = [z[mask] for z in [gams, losses, gs, hs]]

    plt.plot(gams, losses, color="C0")
    grads = np.diff(losses) / np.diff(gams)
    for (i, (gam, loss, g, h)) in enumerate(zip(gams, losses, gs, hs)):
        dgam = 1.5e-1
        grad = np.interp(gam, gams[:-1], grads)
        plt.plot([gam, gam + dgam], [loss, loss + g * dgam], color="C1")

        # if i == 2:
        if i == np.argmin(losses):
            # xp = np.linspace(-10 * dgam + gam, gam + 10 * dgam, 100)
            xp = np.linspace(gams[0], gams[-1], 100)
            fp = loss + g * (xp - gam) + 0.5 * h * (xp - gam) ** 2
            plt.plot(xp, fp, color="C2")
            plt.plot([gam, gam], [np.max(losses), np.min(losses)], color="C2")
        print(grad / (g / h))
    # try:
    #    with gzip.open(OPTHIST_FNAME, "rb") as fp:
    #        gam, gam_hist, gam_hist_losses = pickle.load(fp)
    #        gam_hist = np.array(gam_hist)
    #        plt.plot(gam_hist, gam_hist_losses, color="C1")
    #        plt.scatter(gam_hist, gam_hist_losses, color="C1")
    #        plt.scatter(gam_hist[-1], gam_hist_losses[-1], color="black")
    # except FileNotFoundError:
    #    pass

    plt.ylim([np.min(losses) - 0.1, np.max(losses) + 0.1])
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
    # plt.savefig("figs/gam_optim.png", dpi=200)

    plt.figure()
    plt.plot(gams, accs)
    plt.xlabel("$\\gamma$")
    plt.ylabel("Test Accuracy")

    plt.show()
