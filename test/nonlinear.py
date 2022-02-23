import pdb, os, sys

import matplotlib.pyplot as plt, numpy as np, torch
from tqdm import tqdm

import include_implicit

from implicit.interface import init

jaxm = init(dtype=np.float64)
import jax

from implicit.utils import fn_with_sol_cache
import implicit.plot_utils as pu
from implicit.opt import minimize_sqp, minimize_agd, minimize_lbfgs

from implicit.implicit import implicit_jacobian, implicit_hessian, generate_fns
from implicit.implicit import JACOBIAN, HESSIAN


def partition_var(lam_th, n):
   var1 = lam_th.reshape(-1)[:n].reshape((-1, 1))
   var2 = lam_th.reshape(-1)[n : n + 2].reshape((-1, 1))
   return var1, var2


def poly_feat(X, n=1):
    return jaxm.cat(
        [X ** i if i > 0 else X[..., 0:1] ** 0 for i in range(n + 1)], -1
    )


def F_fn(Z, Y, lam):
    n = Z.shape[-1]
    lam_, lam_z = partition_var(lam, n)
    A = jaxm.t(Z) @ Z / n + jaxm.diag(lam_.reshape(-1) ** 2)
    th_ = jaxm.linalg.solve(A, jaxm.t(Z) @ Y / n)
    th_z = lam_z
    return jaxm.cat([th_, th_z], -2)


def k_fn(th, Z, Y, lam):
    n = Z.shape[-1]
    lam_, lam_z = partition_var(lam, n)
    th_, th_z = partition_var(th, n)
    k_ = jaxm.t(Z) @ (Z @ th_ - Y) / n + lam_ ** 2 * th_
    k_z = th_z * (lam_z - th_z)
    return jaxm.cat([k_, k_z], -2)


def loss(th, Z, Y, lam):
    n = Z.shape[-1]
    th_, th_z = partition_var(th, n)
    return jaxm.mean(jaxm.norm(Z @ th_ - Y, axis=-1) ** 2) + jaxm.norm(
        th_z - 1.0
    )


if __name__ == "__main__":
    n = 10 ** 3
    Xtr = jaxm.linspace(-10, 0, n)[..., None]
    Xts = jaxm.linspace(0, 10, n)[..., None]
    fn = jaxm.sin
    Ytr, Yts = fn(Xtr), fn(Xts)
    p = 3
    Ztr, Zts = poly_feat(Xtr, n=p), poly_feat(Xts, n=p)
    Ztr = jaxm.cat([Ztr, jaxm.sin(Xtr + 0.1)], -1)
    Zts = jaxm.cat([Zts, jaxm.sin(Xts + 0.1)], -1)

    lam = jaxm.rand((Ztr.shape[-1] + 2 + 1,)) + 0.1
    th = F_fn(Ztr, Ytr, lam)
    g = k_fn(th, Ztr, Ytr, lam)

    Dpz = JACOBIAN(lambda lam: F_fn(Ztr, Ytr, lam))(lam)
    Dppz = HESSIAN(lambda lam: F_fn(Ztr, Ytr, lam))(lam)
    ret = implicit_hessian(lambda th, lam: k_fn(th, Ztr, Ytr, lam), th, lam)

    pdb.set_trace()
    assert jaxm.norm(Dpz - ret[0]) / jaxm.norm(Dpz) < 1e-9
    assert jaxm.norm(Dppz - ret[1]) / jaxm.norm(Dppz) < 1e-9
    print("Norm verification passed")

    pu.plot_3d(Dppz.reshape((th.size,) + (lam.size,) * 2))
    plt.draw_all()
    plt.pause(1e-2)

    # define functions ##############################################
    loss_fn_ = lambda th, lam: loss(th, Zts, Yts, lam)
    opt_fn_ = lambda lam: F_fn(Ztr, Ytr, lam)
    k_fn_ = lambda th, lam: k_fn(th, Ztr, Ytr, lam)

    Dpz = implicit_jacobian(k_fn_, opt_fn_(lam), lam)
    f_fn, g_fn, h_fn = generate_fns(loss_fn_, opt_fn_, k_fn_)

    lam0 = 1e-3 * jaxm.randn(lam.shape)
    f = f_fn(lam0)
    g = g_fn(lam0)
    h = h_fn(lam0)

    VERBOSE = True
    lams = minimize_sqp(
        f_fn, g_fn, h_fn, lam0, verbose=VERBOSE, reg0=1e-9, max_it=30
    )
    #lams = minimize_lbfgs(f_fn, g_fn, lam0, verbose=VERBOSE, lr=1e-1, max_it=30)
    print()
    print(loss(F_fn(Ztr, Ytr, lams), Zts, Yts, lams))
    print()
    pdb.set_trace()
