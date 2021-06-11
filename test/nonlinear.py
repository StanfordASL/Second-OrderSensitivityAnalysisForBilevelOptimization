import pdb, os, sys

import matplotlib.pyplot as plt, numpy as np, torch
from tqdm import tqdm

import include_implicit

from implicit.utils import t, diag, topts, fn_with_sol_cache
import implicit.plot_utils as pu
from implicit.opt import minimize_sqp, minimize_agd, minimize_lbfgs

from implicit import implicit_grads_1st, implicit_grads_2nd, generate_fns
from implicit import JACOBIAN, HESSIAN

torch.set_default_dtype(torch.float64)


def partition_var(lam_th, n):
    var1 = lam_th.reshape(-1)[:n].reshape((-1, 1))
    var2 = lam_th.reshape(-1)[n : n + 2].reshape((-1, 1))
    return var1, var2


def poly_feat(X, n=1):
    return torch.cat(
        [X ** i if i > 0 else X[..., 0:1] ** 0 for i in range(n + 1)], -1
    )


def F_fn(Z, Y, lam):
    n = Z.shape[-1]
    lam_, lam_z = partition_var(lam, n)
    A = t(Z) @ Z / n + torch.diag(lam_.reshape(-1) ** 2)
    th_ = torch.solve(t(Z) @ Y / n, A)[0]
    th_z = lam_z
    return torch.cat([th_, th_z], -2)


def k_fn(th, Z, Y, lam):
    n = Z.shape[-1]
    lam_, lam_z = partition_var(lam, n)
    th_, th_z = partition_var(th, n)
    k_ = t(Z) @ (Z @ th_ - Y) / n + lam_ ** 2 * th_
    k_z = th_z * (lam_z - th_z)
    return torch.cat([k_, k_z], -2)


def loss(th, Z, Y, lam):
    n = Z.shape[-1]
    th_, th_z = partition_var(th, n)
    return torch.mean(torch.norm(Z @ th_ - Y, dim=-1) ** 2) + torch.norm(
        th_z - 1.0
    )


if __name__ == "__main__":
    n = 10 ** 3
    Xtr = torch.linspace(-10, 0, n)[..., None]
    Xts = torch.linspace(0, 10, n)[..., None]
    fn = torch.sin
    Ytr, Yts = fn(Xtr), fn(Xts)
    p = 3
    Ztr, Zts = poly_feat(Xtr, n=p), poly_feat(Xts, n=p)
    Ztr = torch.cat([Ztr, torch.sin(Xtr + 0.1)], -1)
    Zts = torch.cat([Zts, torch.sin(Xts + 0.1)], -1)

    lam = torch.rand(Ztr.shape[-1] + 2 + 1) + 0.1
    th = F_fn(Ztr, Ytr, lam)
    g = k_fn(th, Ztr, Ytr, lam)

    Dpz = JACOBIAN(lambda lam: F_fn(Ztr, Ytr, lam), lam)
    Dppz = HESSIAN(lambda lam: F_fn(Ztr, Ytr, lam), lam)
    ret = implicit_grads_2nd(lambda th, lam: k_fn(th, Ztr, Ytr, lam), th, lam)

    assert torch.norm(Dpz - ret[0]) / torch.norm(Dpz) < 1e-9
    assert torch.norm(Dppz - ret[1]) / torch.norm(Dppz) < 1e-9
    print("Norm verification passed")

    pu.plot_3d(Dppz.reshape((th.numel(),) + (lam.numel(),) * 2))
    plt.draw_all()
    plt.pause(1e-2)

    # define functions ##############################################
    loss_fn_ = lambda th, lam: loss(th, Zts, Yts, lam)
    opt_fn_ = lambda lam: F_fn(Ztr, Ytr, lam)
    k_fn_ = lambda th, lam: k_fn(th, Ztr, Ytr, lam)

    Dpz = implicit_grads_1st(k_fn_, opt_fn_(lam), lam)
    f_fn, g_fn, h_fn = generate_fns(loss_fn_, opt_fn_, k_fn_)

    lam0 = 1e-3 * torch.randn(lam.shape)
    VERBOSE = True
    lams = minimize_sqp(
        f_fn, g_fn, h_fn, lam0, verbose=VERBOSE, reg0=1e-9, max_it=30
    )
    #lams = minimize_lbfgs(f_fn, g_fn, lam0, verbose=VERBOSE, lr=1e-2, max_it=30)
    print()
    print(loss(F_fn(Ztr, Ytr, lams), Zts, Yts, lams))
    print()
    pdb.set_trace()
