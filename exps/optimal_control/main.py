import math, sys, pdb, os

import matplotlib.pyplot as plt, numpy as np
import cvxpy as cp

import header
from implicit.interface import init

jaxm = init(dtype=np.float64, device="cpu")

from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit.implicit import implicit_jacobian, implicit_hessian, generate_fns
from implicit.utils import n2j, j2n
from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd
from implicit.pca import visualize_landscape

import dynamics as dyn

topts = lambda x: dict(dtype=x.dtype, device=x.device)

# barrier definition ###########################################################
class QuadPenalty:
    def __init__(self):
        pos = lambda x: jaxm.maximum(x, 0.0)
        self.barrier = lambda A, b, x, s: s * (pos(A @ x - b) ** 2) / 2
        self.dbarrier = lambda A, b, x, s: s * A.T @ pos(A @ x - b)
        self.cstr = (
            lambda A, b, x, s: s * cp.sum_squares(cp.pos((A @ x - b))) / 2
        )


class LogPenalty:
    def __init__(self):
        self.barrier = lambda A, b, x, s: -jaxm.log(-s * (A @ x - b)).sum() / s
        self.dbarrier = lambda A, b, x, s: -A.T @ (1 / (A @ x - b) / s)
        self.cstr = lambda A, b, x, s: -cp.sum(cp.log(-s * (A @ x - b))) / s


################################################################################

XDIM, UDIM, N = 4, 2, 20
SOLVER = cp.MOSEK
PENALTY = LogPenalty
################################################################################

# MPC (exactly constrained) definitions ########################################
def mpc_k_fn_(z, *params, **kw):
    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)
    Q_diag, R_diag, Ft, ft, A, b, gam = [
        kw[k] for k in ["Q_diag", "R_diag", "Ft", "ft", "A", "b", "gam"]
    ]

    Q, R = jaxm.diag(Q_diag.reshape(-1)), jaxm.diag(R_diag.reshape(-1))
    U, Lam = z[: (UDIM * N)], z[(UDIM * N) :]
    dLdx = (
        R @ (U - U_ref).reshape(-1)
        + Ft.T @ Q @ (Ft @ U.reshape(-1) + ft - X_ref.reshape(-1))
        + A.T @ Lam
    )
    ret = jaxm.cat([dLdx, Lam * (A @ U - b)])
    return ret


def mpc_opt_fn_(*params, **kw):
    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)
    Q_diag, R_diag, Ft, ft, A, b, gam = [
        kw[k] for k in ["Q_diag", "R_diag", "Ft", "ft", "A", "b", "gam"]
    ]

    Q, R = jaxm.diag(Q_diag.reshape(-1)), jaxm.diag(R_diag.reshape(-1))
    P = R + Ft.T @ Q @ Ft
    q = R @ U_ref.reshape(-1) + Ft.T @ Q @ (X_ref.reshape(-1) - ft.reshape(-1))

    P, q = j2n(P), j2n(q)
    x = cp.Variable(q.shape[-1])
    A, b = j2n(A), j2n(b)

    cstr = [A @ x <= b]
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) - x @ q),
        cstr,
    )
    prob.solve(cp.MOSEK)
    assert prob.status in ["optimal", "optimal_inaccurate"]
    U, Lam = n2j(x.value), n2j(cstr[0].dual_value)
    return jaxm.cat([U, Lam])


################################################################################


################################################################################
def barrier_k_fn_(z, *params, **kw):
    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)
    Q_diag, R_diag, Ft, ft, A, b, gam = [
        kw[k] for k in ["Q_diag", "R_diag", "Ft", "ft", "A", "b", "gam"]
    ]

    Q, R = jaxm.diag(Q_diag.reshape(-1)), jaxm.diag(R_diag.reshape(-1))
    U = z
    ret = (
        R @ (U - U_ref).reshape(-1)
        + Ft.T @ Q @ (Ft @ U.reshape(-1) + ft - X_ref.reshape(-1))
        + kw["penalty"].dbarrier(A, b, z, gam)
    )
    err = jaxm.norm(ret)
    if err > 1e-4 * (1 + gam):
        print("Warning: Optimality error: %9.4e" % err)
    return ret


def barrier_opt_fn_(*params, **kw):
    Q_diag, R_diag, Ft, ft, A, b, gam = [
        kw[k] for k in ["Q_diag", "R_diag", "Ft", "ft", "A", "b", "gam"]
    ]

    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)

    Q, R = jaxm.diag(Q_diag.reshape(-1)), jaxm.diag(R_diag.reshape(-1))
    P = R + Ft.T @ Q @ Ft
    q = R @ U_ref.reshape(-1) + Ft.T @ Q @ (X_ref.reshape(-1) - ft.reshape(-1))

    P, q = j2n(P), j2n(q)
    z = cp.Variable(q.shape[-1])
    A, b = j2n(A), j2n(b)

    prob = cp.Problem(
        cp.Minimize(
            0.5 * cp.quad_form(z, P) - z @ q + kw["penalty"].cstr(A, b, z, gam)
        ),
    )
    prob.solve(SOLVER)
    assert prob.status in ["optimal"]
    # assert prob.status in ["optimal", "optimal_inaccurate"]
    U = n2j(z.value)
    return U


################################################################################


def loss_fn_(z, *params, **kw):
    U = z[: (UDIM * N)].reshape((N, UDIM))
    X_expert, U_expert, Ft, ft = [
        kw[k] for k in ["X_expert", "U_expert", "Ft", "ft"]
    ]
    X = (Ft @ U.reshape(-1) + ft).reshape((N, XDIM))
    return jaxm.sum((U - U_expert) ** 2) + jaxm.sum((X - X_expert) ** 2)


if __name__ == "__main__":
    USE_CSTR = bool(int(sys.argv[1]))
    # seed_np, seed_jax = int(sys.argv[1]), int(sys.argv[2])
    seed_np, seed_jax = 18352, 26792
    np.random.seed(seed_np)
    jaxm.manual_seed(seed_jax)

    # invariant problem parameters
    Q_diag = jaxm.tile(jaxm.array([1.0, 1e-2, 1.0, 1e-2]), (N, 1))
    R_diag = 1e-2 * jaxm.ones((N, UDIM))
    X_prev, U_prev = jaxm.zeros((N, XDIM)), jaxm.zeros((N, UDIM))
    dt = 0.1
    P = jaxm.cat([dt * jaxm.ones((N, 1)), jaxm.ones((N, 2))], -1)
    x0 = jaxm.randn((XDIM,))
    A = jaxm.cat([jaxm.eye(UDIM * N), -jaxm.eye(UDIM * N)], -2)
    if USE_CSTR:
        b = 0.3 * jaxm.cat([jaxm.ones(UDIM * N), jaxm.ones(UDIM * N)])
    else:
        b = 1e7 * jaxm.cat([jaxm.ones(UDIM * N), jaxm.ones(UDIM * N)])

    # generate dynamics
    X, U = X_prev, U_prev
    f, fx, fu = dyn.f_fn(X, U, P), dyn.fx_fn(X, U, P), dyn.fu_fn(X, U, P)
    Ft, ft = dyn.dyn_mat(x0, f, fx, fu, X, U)

    gam = 1e2

    # generate expert trajectory
    X_ref_expert, U_ref_expert = jaxm.zeros((N, XDIM)), jaxm.zeros((N, UDIM))
    fn_kw = dict(Q_diag=Q_diag, R_diag=R_diag, Ft=Ft, ft=ft, A=A, b=b, gam=gam)
    fn_kw["penalty"] = PENALTY()
    U_expert = barrier_opt_fn_(X_ref_expert, U_ref_expert, **fn_kw)[
        : N * UDIM
    ].reshape((N, UDIM))
    X_expert = (Ft @ U_expert.reshape(-1) + ft).reshape((N, XDIM))
    fn_kw = dict(fn_kw, X_expert=X_expert, U_expert=U_expert)

    X_ref = X_ref_expert + 1e0 * jaxm.randn((N, XDIM))
    U_ref = U_ref_expert + 1e0 * jaxm.randn((N, UDIM))

    params2vec = lambda X_ref, U_ref: jaxm.cat(
        [X_ref.reshape(-1), U_ref.reshape(-1)]
    )
    vec2params = lambda v: (
        v[: X_ref.size].reshape(X_ref.shape),
        v[X_ref.size :].reshape(U_ref.shape),
    )

    # generate all optimization functions ######################################
    loss_fn = lambda *args: loss_fn_(*args, **fn_kw)
    barrier_opt_fn = lambda *args: barrier_opt_fn_(*args, **fn_kw)
    barrier_k_fn = lambda *args: barrier_k_fn_(*args, **fn_kw)
    mpc_opt_fn = lambda *args: mpc_opt_fn_(*args, **fn_kw)
    mpc_k_fn = lambda *args: mpc_k_fn_(*args, **fn_kw)

    barrier_fns = generate_fns(loss_fn, barrier_opt_fn, barrier_k_fn, jit=False)
    mpc_fns = generate_fns(loss_fn, mpc_opt_fn, mpc_k_fn, jit=False)

    sqp_loss_fn = lambda z, v: loss_fn(z, *vec2params(v))

    sqp_barrier_opt_fn = lambda v: barrier_opt_fn(*vec2params(v))
    sqp_barrier_k_fn = lambda z, v: barrier_k_fn(z, *vec2params(v))
    sqp_barrier_fns = generate_fns(
        sqp_loss_fn, sqp_barrier_opt_fn, sqp_barrier_k_fn, jit=False
    )

    sqp_mpc_opt_fn = lambda v: mpc_opt_fn(*vec2params(v))
    sqp_mpc_k_fn = lambda z, v: k_fn(z, *vec2params(v))
    sqp_mpc_fns = generate_fns(
        sqp_loss_fn, sqp_mpc_opt_fn, sqp_mpc_k_fn, jit=False
    )
    ############################################################################

    method = "sqp"
    opt_opts = dict(verbose=True, full_output=True)
    opt_vars = X_ref, U_ref
    fns = barrier_fns[:2]
    if method == "agd":
        minimize_fn = minimize_agd
        opt_opts = dict(opt_opts, ai=1e-2, af=1e-2, max_it=10 ** 3)
    elif method == "lbfgs":
        minimize_fn = minimize_lbfgs
        opt_opts = dict(opt_opts, lr=1e-1, max_it=10)
    elif method == "sqp":
        minimize_fn = minimize_sqp
        opt_opts = dict(
            opt_opts, ls_pts_nb=5, max_it=10, force_step=True, reg0=1e-12
        )
        fns = sqp_barrier_fns
        opt_vars = (params2vec(X_ref, U_ref),)
    x, x_hist = minimize_fn(*fns, *opt_vars, **opt_opts)
    X_ref, U_ref = vec2params(x) if method == "sqp" else x
    x_hist = x_hist if method == "sqp" else [params2vec(*z) for z in x_hist]

    n_pts = 60

    # gams = np.arange(1, 5)
    gams = np.linspace(1, 4, 3)
    figs, axs = [], []
    if USE_CSTR:
        for gam in gams:
            fn_kw["gam"] = 10.0 ** gam
            fn_kw["penalty"] = LogPenalty()
            X_barrier, Y_barrier = visualize_landscape(
                lambda x: sqp_loss_fn(sqp_barrier_opt_fn(x), x),
                x_hist,
                n_pts,
                log=False,
                verbose=True,
                zoom_scale=0.3,
            )
            axs.append(plt.gca())
            figs.append(plt.gcf())
    X_mpc, Y_mpc = visualize_landscape(
        lambda x: sqp_loss_fn(sqp_mpc_opt_fn(x), x),
        x_hist,
        n_pts,
        log=False,
        verbose=True,
        zoom_scale=0.3,
    )
    axs.append(plt.gca())
    figs.append(plt.gcf())

    zlims = [ax.get_zlim() for ax in axs]
    zlim = (min([x[0] for x in zlims]), max([x[1] for x in zlims]))
    if USE_CSTR:
        for (i, gam) in enumerate(gams):
            axs[i].set_zlim(zlim)
            axs[i].axes.xaxis.set_ticklabels([])
            axs[i].axes.yaxis.set_ticklabels([])
            axs[i].axes.zaxis.set_ticklabels([])
            figs[i].savefig(
                "figs/gam_%s%d_log_barrier_landscape_%d_%d.png"
                % ("p" if gam >= 0 else "n", abs(gam), seed_np, seed_jax),
                dpi=200,
                bbox_inches="tight",
                pad_inches=0,
            )
    axs[-1].set_zlim(zlim)
    axs[-1].axes.xaxis.set_ticklabels([])
    axs[-1].axes.yaxis.set_ticklabels([])
    axs[-1].axes.zaxis.set_ticklabels([])
    if USE_CSTR:
        figs[-1].savefig(
            "figs/mpc_landscape_%d_%d.png" % (seed_np, seed_jax),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
        )
    else:
        figs[-1].savefig(
            "figs/ilqr_landscape.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
        )

    # fn_kw["penalty"] = QuadPenalty()
    # X_barrier, Y_barrier = visualize_landscape(
    #    lambda x: sqp_loss_fn(sqp_barrier_opt_fn(x), x),
    #    x_hist,
    #    n_pts,
    #    log=False,
    #    verbose=True,
    #    zoom_scale=0.3,
    # )
    # plt.savefig("figs/quad_barrier_landscape.png", dpi=200)
    # plt.title("Quad")

    #plt.figure()
    #plt.imshow(Y_barrier - Y_mpc)
    #plt.colorbar()

    # l = loss_fn(
    #    barrier_opt_fn(X_ref_expert, U_ref_expert), X_ref_expert, U_ref_expert
    # )

    # params = X_ref, U_ref
    # f = f_fn(*params)
    # g = g_fn(*params)
    # h = h_fn(*params)

    # U = opt_fn(X_ref, U_ref).reshape((N, UDIM))
    # X = (Ft @ U.reshape(-1) + ft).reshape((N, XDIM))
    # X, U = X.cpu(), U.cpu()

    # plt.figure()
    # plt.title("U")
    # ls = ["-", "--", ":", "-."]
    # for r in range(UDIM):
    #    plt.plot(U[:, r], label="u%d" % (r + 1), color="C0", ls=ls[r])
    #    plt.plot(U_expert[:, r], label="u%d" % (r + 1), color="C1", ls=ls[r])
    # plt.legend()

    # plt.figure()
    # plt.title("X")
    # for r in range(XDIM):
    #    plt.plot(X[:, r], label="x%d" % (r + 1), color="C0", ls=ls[r])
    #    plt.plot(X_expert[:, r], label="x%d" % (r + 1), color="C1", ls=ls[r])
    # plt.legend()

    # plt.draw_all()
    # plt.pause(1e-1)

    # pdb.set_trace()
