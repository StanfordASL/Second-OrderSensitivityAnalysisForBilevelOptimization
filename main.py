import math, pdb, os, sys, time

import torch

from utils import t, topts
from diff import torch_grad as grad, torch_hessian as hessian
from implicit import implicit_grads, prod

EPS = 1e-4


def k_fn(x, A, b):
    return t(A) @ (A @ x - b) + EPS * x


def F_fn(A, b):
    H = t(A) @ A + EPS * torch.eye(A.shape[-1], **topts(A))
    return torch.cholesky_solve((t(A) @ b)[..., None], torch.cholesky(H))[
        ..., 0
    ]


if __name__ == "__main__":
    # generate the problem data
    m, n = 5, 3
    A, b = torch.randn((m, n)), torch.randn((m,))

    # focus only on the matrix parameter A
    k = lambda x, A: k_fn(x, A, b)
    F = lambda A: F_fn(A, b)

    # compute the optimal value of x and assert 1st order optimality
    x = F(A)
    assert torch.norm(k(x, A)) < 1e-5

    # compute the derivatives directly, to compare against
    Dpz = grad(F)(A)
    Dppz = hessian(F)(A)

    # compute the derivative components of 2nd order implicit gradients
    Dpk = grad(k, argnums=1)(x, A)
    Dzk = grad(k, argnums=0)(x, A)
    Dppk = hessian(k, argnums=1)(x, A)
    Dzpk = grad(grad(k, argnums=1, create_graph=True), argnums=0)(x, A)
    Dpzk = grad(grad(k, argnums=0, create_graph=True), argnums=1)(x, A)
    Dzzk = hessian(k, argnums=0)(x, A)

    # reform the matrices into the 3 dimension convention (for vector Hessians)
    shz, shp = prod(x.shape), prod(A.shape)
    Dzk_ = Dzk.reshape((shz, shz))
    Dzzk_ = Dzzk.reshape((shz, shz, shz))
    Dpk_ = Dpk.reshape((shz, shp))
    Dppk_ = Dppk.reshape((shz, shp, shp))
    Dzpk_ = Dzpk.reshape((shz, shp, shz))
    Dpzk_ = Dpzk.reshape((shz, shz, shp))

    # solve for the first order implicit gradients
    Dpz_ = -torch.solve(Dpk_, Dzk_)[0].reshape((shz, shp))
    assert torch.norm(Dpz_.reshape(-1) - Dpz.reshape(-1)) < 1e-5

    # form the 2nd order implicit gradient implicit eqution terms first & second
    first = (
        Dppk_
        + Dzpk_ @ Dpz_[None, ...] # first dim is batched
        + t(Dpz_)[None, ...] @ t(Dzpk_) # first dim is batched
        + (t(Dpz_)[None, ...] @ Dzzk_) @ Dpz_[None, ...] # first dim is batched
    )
    second = (Dzk_ @ Dppz.reshape((shz, -1))).reshape((shz, shp, shp))
    assert torch.norm(first + second) < 1e-5

    # solve for Dppz; this time the 3rd dimension is batched
    lhs = first
    Dppz_ = -torch.solve(lhs.reshape((shz, shp * shp)), Dzk_)[0].reshape(
        x.shape + A.shape + A.shape
    )
    assert torch.norm(Dppz_ - Dppz) < 1e-5

    # compare against the integrated routine
    Dpz3, Dppz3 = implicit_grads(lambda x, A, b: k_fn(x, A, b), x, A, b)
    assert torch.norm(Dpz3[0] - Dpz) < 1e-5
    assert torch.norm(Dppz3[0] - Dppz) < 1e-5

    print("All assertions satisfied")
