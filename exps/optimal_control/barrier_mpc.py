import math, sys, pdb, os

import torch, matplotlib.pyplot as plt
import cvxpy as cp

import header
from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit import implicit_jacobian, implicit_hessian, generate_fns
from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd
from implicit.pca import visualize_landscape

import dynamics as dyn

topts = lambda x: dict(dtype=x.dtype, device=x.device)

# barrier definition ###########################################################
class QuadPenalty:
    def __init__(self):
        pos = lambda x: torch.maximum(x, torch.zeros((), **topts(x)))
        self.barrier = lambda A, b, x, s: s * (pos(A @ x - b) ** 2) / 2
        self.dbarrier = lambda A, b, x, s: s * A.T @ pos(A @ x - b)
        self.cstr = (
            lambda A, b, x, s: s * cp.sum_squares(cp.pos((A @ x - b))) / 2
        )


class LogPenalty:
    def __init__(self):
        self.barrier = lambda A, b, x, s: -torch.log(-s * (A @ x - b)).sum() / s
        self.dbarrier = lambda A, b, x, s: -A.T @ (1 / (A @ x - b) / s)
        self.cstr = lambda A, b, x, s: -cp.sum(cp.log(-s * (A @ x - b))) / s

################################################################################

XDIM, UDIM, N = 4, 2, 20
device = "cpu"
torch.set_default_dtype(torch.float64)
SOLVER = cp.MOSEK
PENALTY = LogPenalty
################################################################################



def k_fn_(z, *params, **kw):
    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)
    Q_diag, R_diag, Ft, ft, A, b, gam = [
        kw[k] for k in ["Q_diag", "R_diag", "Ft", "ft", "A", "b", "gam"]
    ]

    Q, R = torch.diag(Q_diag.reshape(-1)), torch.diag(R_diag.reshape(-1))
    U = z
    ret = (
        R @ (U - U_ref).reshape(-1)
        + Ft.T @ Q @ (Ft @ U.reshape(-1) + ft - X_ref.reshape(-1))
        + kw["penalty"].dbarrier(A, b, z, gam)
    )
    err = torch.norm(ret)
    if err > 1e-4 * (1 + gam):
        print("Warning: Optimality error: %9.4e" % err)
    return ret


def opt_fn_(*params, **kw):
    Q_diag, R_diag, Ft, ft, A, b, gam = [
        kw[k] for k in ["Q_diag", "R_diag", "Ft", "ft", "A", "b", "gam"]
    ]

    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)

    Q, R = torch.diag(Q_diag.reshape(-1)), torch.diag(R_diag.reshape(-1))
    P = R + Ft.T @ Q @ Ft
    q = R @ U_ref.reshape(-1) + Ft.T @ Q @ (X_ref.reshape(-1) - ft.reshape(-1))

    P, q = P.cpu().detach().numpy(), q.cpu().detach().numpy()
    z = cp.Variable(q.shape[-1])
    A = A.cpu().detach().numpy()
    b = b.cpu().detach().numpy()

    prob = cp.Problem(
        cp.Minimize(
            0.5 * cp.quad_form(z, P) - z @ q + kw["penalty"].cstr(A, b, z, gam)
        ),
    )
    prob.solve(SOLVER)
    assert prob.status in ["optimal"]
    # assert prob.status in ["optimal", "optimal_inaccurate"]
    U = torch.tensor(z.value, device=X_ref.device, dtype=X_ref.dtype)
    return U


def loss_fn_(z, *params, **kw):
    U = z[: (UDIM * N)].reshape((N, UDIM))
    X_expert, U_expert, Ft, ft = [
        kw[k] for k in ["X_expert", "U_expert", "Ft", "ft"]
    ]
    X = (Ft @ U.reshape(-1) + ft).reshape((N, XDIM))
    return torch.sum((U - U_expert) ** 2) + torch.sum((X - X_expert) ** 2)


if __name__ == "__main__":
    # utility functions
    zeros = lambda *args, **kw: torch.zeros(*args, **kw, device=device)
    ones = lambda *args, **kw: torch.ones(*args, **kw, device=device)
    randn = lambda *args, **kw: torch.randn(*args, **kw, device=device)
    tensor = lambda *args, **kw: torch.tensor(*args, **kw, device=device)

    # invariant problem parameters
    Q_diag = torch.tile(tensor([1.0, 1e-2, 1.0, 1e-2]), (N, 1))
    R_diag = 1e-2 * ones((N, UDIM))
    X_prev, U_prev = zeros((N, XDIM)), zeros((N, UDIM))
    dt = 0.1
    P = torch.cat([dt * ones((N, 1)), ones((N, 2))], -1)
    x0 = randn(XDIM)
    A = torch.cat([torch.eye(UDIM * N), -torch.eye(UDIM * N)], -2).to(device)
    b = 0.3 * torch.cat([torch.ones(UDIM * N), torch.ones(UDIM * N)]).to(device)

    # generate dynamics
    X, U = X_prev, U_prev
    f, fx, fu = dyn.f_fn(X, U, P), dyn.fx_fn(X, U, P), dyn.fu_fn(X, U, P)
    Ft, ft = dyn.dyn_mat(x0, f, fx, fu, X, U)

    gam = 1e4

    # generate expert trajectory
    X_ref_expert, U_ref_expert = zeros((N, XDIM)), zeros((N, UDIM))
    fn_kw = dict(Q_diag=Q_diag, R_diag=R_diag, Ft=Ft, ft=ft, A=A, b=b, gam=gam)
    fn_kw["penalty"] = PENALTY()
    U_expert = opt_fn_(X_ref_expert, U_ref_expert, **fn_kw).reshape((N, UDIM))
    X_expert = (Ft @ U_expert.reshape(-1) + ft).reshape((N, XDIM))
    fn_kw = dict(fn_kw, X_expert=X_expert, U_expert=U_expert)

    loss_fn = lambda *args: loss_fn_(*args, **fn_kw)
    opt_fn = lambda *args: opt_fn_(*args, **fn_kw)
    k_fn = lambda *args: k_fn_(*args, **fn_kw)

    X_ref = X_ref_expert + 1e0 * randn((N, XDIM))
    U_ref = U_ref_expert + 1e0 * randn((N, UDIM))

    f_fn, g_fn, h_fn = generate_fns(loss_fn, opt_fn, k_fn)

    params2vec = lambda X_ref, U_ref: torch.cat(
        [X_ref.reshape(-1), U_ref.reshape(-1)]
    )
    vec2params = lambda v: (
        v[: X_ref.numel()].reshape(X_ref.shape),
        v[X_ref.numel() :].reshape(U_ref.shape),
    )

    params = X_ref, U_ref
    gs = g_fn(*params)
    gs2 = JACOBIAN(lambda *params: loss_fn(opt_fn(*params), *params), params)
    hs = h_fn(*params)
    hs2 = HESSIAN_DIAG(
        lambda *params: loss_fn(opt_fn(*params), *params), params
    )

    opt_fns = lambda v: opt_fn(*vec2params(v))
    k_fns = lambda z, v: k_fn(z, *vec2params(v))
    loss_fns = lambda z, v: loss_fn(z, *vec2params(v))
    f_fn_, g_fn_, h_fn_ = generate_fns(loss_fns, opt_fns, k_fns)
    h_fn__ = h_fn_

    def h_fn_(*params):
        H = h_fn__(*params)
        e = torch.sort(torch.linalg.eigvals(H).real)[0][[0, -1]]
        print("(%+9.4e, %+9.4e)" % (e[0].cpu(), e[1].cpu()))
        # if e[0] < -1e-7:
        #    print("Hessian is not PSD")
        print(e[0])
        return H

    method = "sqp"
    opt_opts = dict(verbose=True, full_output=True)
    opt_vars = X_ref, U_ref
    fns = [f_fn, g_fn]
    if method == "agd":
        minimize_fn = minimize_agd
        opt_opts = dict(opt_opts, ai=1e-2, af=1e-2, max_it=10 ** 3)
    elif method == "lbfgs":
        minimize_fn = minimize_lbfgs
        opt_opts = dict(opt_opts, lr=1e-1, max_it=10)
    elif method == "sqp":
        minimize_fn = minimize_sqp
        opt_opts = dict(
            opt_opts, ls_pts_nb=5, max_it=40, force_step=True, reg0=1e0
        )
        fns = [f_fn_, g_fn_, h_fn_]
        opt_vars = (params2vec(X_ref, U_ref),)
    x, x_hist = minimize_fn(*fns, *opt_vars, **opt_opts)
    X_ref, U_ref = vec2params(x) if method == "sqp" else x
    x_hist = x_hist if method == "sqp" else [params2vec(*z) for z in x_hist]

    X, Y = visualize_landscape(
        lambda x: loss_fns(opt_fns(x), x), x_hist, 100, log=False, verbose=True
    )

    l = loss_fn(opt_fn(X_ref_expert, U_ref_expert), X_ref_expert, U_ref_expert)
    plt.draw_all()
    plt.pause(1e-1)

    params = X_ref, U_ref
    f = f_fn(*params)
    g = g_fn(*params)
    h = h_fn(*params)

    U = opt_fn(X_ref, U_ref).reshape((N, UDIM))
    X = (Ft @ U.reshape(-1) + ft).reshape((N, XDIM))
    X, U = X.cpu(), U.cpu()

    plt.figure()
    plt.title("U")
    ls = ["-", "--", ":", "-."]
    for r in range(UDIM):
        plt.plot(U[:, r], label="u%d" % (r + 1), color="C0", ls=ls[r])
        plt.plot(U_expert[:, r], label="u%d" % (r + 1), color="C1", ls=ls[r])
    plt.legend()

    plt.figure()
    plt.title("X")
    for r in range(XDIM):
        plt.plot(X[:, r], label="x%d" % (r + 1), color="C0", ls=ls[r])
        plt.plot(X_expert[:, r], label="x%d" % (r + 1), color="C1", ls=ls[r])
    plt.legend()

    plt.draw_all()
    plt.pause(1e-1)

    pdb.set_trace()
