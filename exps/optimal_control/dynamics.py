import sys, math, pdb, os

import header
from implicit.interface import init

jaxm = init()
from implicit.diff import JACOBIAN

bmv = lambda A, x: (A @ x[..., None])[..., 0]


def rollout(f_fn, x0, U, P):
    assert U.ndim == 2 and x0.ndim == 1 and P.ndim == 2
    xs = []
    x = x0
    for i in range(U.shape[-2]):
        x = f_fn(x, U[i, :], P[i, :])
        xs.append(x)
    return jaxm.stack(xs)


def f_fn(X, U, P):
    assert X.shape[-1] == 4 and U.shape[-1] == 2 and P.shape[-1] >= 3
    dt, u_scale1, u_scale2 = P[..., 0], P[..., 1], P[..., 2]
    x1, x2, x3, x4 = (X[..., i] for i in range(4))
    u1, u2 = U[..., 0], U[..., 1]
    return jaxm.stack(
        [x1 + dt * x2, x2 + u_scale1 * u1, x3 + dt * x4, x4 + u_scale2 * u2], -1
    )


def fx_fn(X, U, P):
    ret = JACOBIAN(
        lambda X: jaxm.sum(f_fn(X, U, P).reshape((-1, X.shape[-1])), -2)
    )(X)
    return jaxm.swapaxes(ret, 0, 1).reshape(
        X.shape[:-1] + (X.shape[-1], X.shape[-1])
    )


def fu_fn(X, U, P):
    ret = JACOBIAN(
        lambda U: jaxm.sum(f_fn(X, U, P).reshape((-1, X.shape[-1])), -2)
    )(U)
    return jaxm.swapaxes(ret, 0, 1).reshape(
        X.shape[:-1] + (X.shape[-1], U.shape[-1])
    )


def dyn_mat(x0, f, fx, fu, X_prev, U_prev):
    """
    construct the matrix and bias vector that gives from a local linearization
    vec(X) = Ft @ vec(U) + ft
    """
    bshape, (N, xdim), udim = fx.shape[:-3], fx.shape[-3:-1], fu.shape[-1]

    Fts = [[None for _ in range(N)] for _ in range(N)]
    Z_ = jaxm.zeros(bshape + (xdim, udim))
    Fts = [[Z_ for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(i):
            Fts[i][j] = fx[..., i, :, :] @ Fts[i - 1][j]
        Fts[i][i] = fu[..., i, :, :]
    Ft = jaxm.cat(
        [jaxm.cat([Fts[i][j] for i in range(N)], -2) for j in range(N)], -1
    )

    fts = [None for i in range(N)]
    f_ = f - bmv(fx, X_prev) - bmv(fu, U_prev)
    fts[0] = bmv(fx[..., 0, :, :], x0) + f_[..., 0, :]
    for i in range(1, N):
        fts[i] = bmv(fx[..., i, :, :], fts[i - 1]) + f_[..., i, :]
    ft = jaxm.cat(fts, -1)
    return Ft, ft


if __name__ == "__main__":
    xdim, udim, N = 4, 2, 10
    dt = 0.1
    x0 = jaxm.randn((xdim,))
    X, U = jaxm.randn((N, xdim)), jaxm.randn((N, udim))
    P = jaxm.stack([dt * jaxm.ones(N), jaxm.ones(N), jaxm.ones(N)], -1)
    X_prev, U_prev = jaxm.randn((N, xdim)), jaxm.randn((N, udim))

    f, fx, fu = (
        f_fn(X_prev, U_prev, P),
        fx_fn(X_prev, U_prev, P),
        fu_fn(X_prev, U_prev, P),
    )
    Ft, ft = dyn_mat(x0, f, fx, fu, X_prev, U_prev)
    Xt = (Ft @ U.reshape(-1) + ft).reshape((N, xdim))
    Xt2 = rollout(f_fn, x0, U, P)

    pdb.set_trace()
