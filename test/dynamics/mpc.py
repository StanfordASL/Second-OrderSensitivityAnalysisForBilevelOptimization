import math, sys, pdb, os

import torch, matplotlib.pyplot as plt
import cvxpy as cp

import header
from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit import implicit_grads_1st, implicit_grads_2nd, generate_fns
from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd
from implicit.pca import visualize_landscape

import dynamics as dyn

XDIM, UDIM, N = 4, 2, 20
device = "cpu"
torch.set_default_dtype(torch.float64)


def k_fn(z, *params):
    self = k_fn
    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)
    Q_diag, R_diag, Ft, ft = self.Q_diag, self.R_diag, self.Ft, self.ft
    A, b = self.A, self.b

    Q, R = torch.diag(Q_diag.reshape(-1)), torch.diag(R_diag.reshape(-1))
    U, Lam = z[: (UDIM * N)], z[(UDIM * N) :]
    dLdx = (
        R @ (U - U_ref).reshape(-1)
        + Ft.T @ Q @ (Ft @ U.reshape(-1) + ft - X_ref.reshape(-1))
        + A.T @ Lam
    )
    ret = torch.cat([dLdx, Lam * (A @ U - b)])
    return ret


def opt_fn(*params):
    self = opt_fn
    Q_diag, R_diag, Ft, ft = self.Q_diag, self.R_diag, self.Ft, self.ft
    A, b = self.A, self.b

    X_ref, U_ref = params
    X_ref = X_ref.reshape(-1)
    U_ref = U_ref.reshape(-1)

    Q, R = torch.diag(Q_diag.reshape(-1)), torch.diag(R_diag.reshape(-1))
    P = R + Ft.T @ Q @ Ft
    q = R @ U_ref.reshape(-1) + Ft.T @ Q @ (X_ref.reshape(-1) - ft.reshape(-1))

    P, q = P.cpu().detach().numpy(), q.cpu().detach().numpy()
    x = cp.Variable(q.shape[-1])
    A = A.cpu().detach().numpy()
    b = b.cpu().detach().numpy()

    cstr = [A @ x <= b]
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) - x @ q),
        cstr,
    )
    prob.solve(cp.GUROBI)
    assert prob.status == "optimal"
    U = torch.as_tensor(x.value, device=X_ref.device, dtype=X_ref.dtype)
    Lam = torch.as_tensor(
        cstr[0].dual_value, device=X_ref.device, dtype=X_ref.dtype
    )
    return torch.cat([U, Lam])


def loss_fn(z, *params):
    self = loss_fn
    U = z[: (UDIM * N)]
    Q_diag, R_diag, Ft, ft = self.Q_diag, self.R_diag, self.Ft, self.ft
    X_expert, U_expert = self.X_expert, self.U_expert
    X = (Ft @ U.reshape(-1) + ft).reshape((N, XDIM))
    return torch.sum((U - U_expert) ** 2) + torch.sum((X - X_expert) ** 2)


if __name__ == "__main__":
    # utility functions
    zeros = lambda *args, **kwargs: torch.zeros(*args, **kwargs, device=device)
    ones = lambda *args, **kwargs: torch.ones(*args, **kwargs, device=device)
    randn = lambda *args, **kwargs: torch.randn(*args, **kwargs, device=device)
    tensor = lambda *args, **kwargs: torch.tensor(
        *args, **kwargs, device=device
    )

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
    opt_fn.Q_diag, opt_fn.R_diag, opt_fn.Ft, opt_fn.ft = Q_diag, R_diag, Ft, ft
    opt_fn.A, opt_fn.b = A, b

    k_fn.Q_diag, k_fn.R_diag, k_fn.Ft, k_fn.ft = Q_diag, R_diag, Ft, ft
    k_fn.A, k_fn.b = A, b
    loss_fn.Q_diag, loss_fn.R_diag, loss_fn.Ft, loss_fn.ft = (
        Q_diag,
        R_diag,
        Ft,
        ft,
    )

    # generate expert trajectory
    X_ref_expert, U_ref_expert = zeros((N, XDIM)), zeros((N, UDIM))
    U_expert = opt_fn(X_ref_expert, U_ref_expert)[: (UDIM * N)]
    X_expert = (Ft @ U_expert.reshape(-1) + ft).reshape((N, XDIM))
    loss_fn.X_expert, loss_fn.U_expert = X_expert, U_expert

    X_ref = X_ref_expert + 1e-1 * randn((N, XDIM))
    U_ref = U_ref_expert + 1e-1 * randn((N, UDIM))

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

    opt_fn_ = lambda v: opt_fn(*vec2params(v))
    k_fn_ = lambda z, v: k_fn(z, *vec2params(v))
    loss_fn_ = lambda z, v: loss_fn(z, *vec2params(v))
    f_fn_, g_fn_, h_fn_ = generate_fns(loss_fn_, opt_fn_, k_fn_)

    method = "agd"
    if method == "agd":
        (X_ref, U_ref), x_hist = minimize_agd(
            f_fn,
            g_fn,
            X_ref,
            U_ref,
            verbose=True,
            ai=1e-1,
            af=1e-3,
            max_it=10 ** 3,
            full_output=True,
        )
        x_hist = [params2vec(*z) for z in x_hist]
    elif method == "lbfgs":
        (X_ref, U_ref), x_hist = minimize_lbfgs(
            f_fn,
            g_fn,
            X_ref,
            U_ref,
            verbose=True,
            lr=1e-1,
            max_it=10 ** 2,
            full_output=True,
        )
        x_hist = [params2vec(*z) for z in x_hist]
    elif method == "sqp":
        x, x_hist = minimize_sqp(
            f_fn_,
            g_fn_,
            h_fn_,
            params2vec(X_ref, U_ref),
            verbose=True,
            full_output=True,
            ls_pts_nb=1,
        )
    else:
        raise ValueError("No such method [%s]" % method)

    visualize_landscape(
        lambda x: loss_fn_(opt_fn_(x), x), x_hist, 100, log=True
    )

    l = loss_fn(opt_fn(X_ref_expert, U_ref_expert), X_ref_expert, U_ref_expert)
    plt.draw_all()
    plt.pause(1e-1)

    params = X_ref, U_ref
    f = f_fn(*params)
    g = g_fn(*params)
    h = h_fn(*params)
    pdb.set_trace()

    X, U = X.cpu(), U.cpu()
    plt.figure()
    plt.title("U")
    for r in range(UDIM):
        plt.plot(U[:, r], label="u%d" % (r + 1))
    plt.legend()

    plt.figure()
    plt.title("X")
    for r in range(XDIM):
        plt.plot(X[:, r], label="x%d" % (r + 1))
    plt.legend()

    plt.draw_all()
    plt.pause(1e-1)

    pdb.set_trace()
