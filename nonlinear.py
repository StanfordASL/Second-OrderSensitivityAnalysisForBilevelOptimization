import pdb, os, sys

import matplotlib.pyplot as plt, numpy as np, torch
from tqdm import tqdm

sys.path.append(os.path.expanduser("~/Dropbox/lib/python"))

from torch_tools.diff import torch_grad as grad, torch_hessian as hessian
from torch_tools.utils import t, diag, topts, fn_with_sol_cache
from torch_tools.opt import minimize_sqp, minimize_agd, minimize_lbfgs

# from ipopt_tools.opt import minimize_ipopt

from implicit import implicit_grads_1st, implicit_grads_2nd, generate_fns

torch.set_default_dtype(torch.float64)

CHECK_GRADS = True


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

    Dpz = grad(lambda lam: F_fn(Ztr, Ytr, lam))(lam)
    Dppz = hessian(lambda lam: F_fn(Ztr, Ytr, lam))(lam)
    ret = implicit_grads_2nd(lambda th, lam: k_fn(th, Ztr, Ytr, lam), th, lam)

    assert torch.norm(Dpz - ret[0]) < 1e-5
    assert torch.norm(Dppz - ret[1]) < 1e-5

    # define functions ##############################################
    ##^
    #fn, sol_cache = (lambda lam: F_fn(Ztr, Ytr, lam)), dict()
    #@fn_with_sol_cache(fn, sol_cache)
    #def f_fn(th, lam):
    #    th, lam = th.detach(), lam.detach()
    #    return loss(th, Zts, Yts, lam)

    #@fn_with_sol_cache(fn, sol_cache)
    #def g_fn(th, lam):
    #    th, lam = th.detach(), lam.detach()
    #    Dg = grad(lambda th: loss(th, Zts, Yts, lam))(th).reshape(
    #        (1, th.numel())
    #    )
    #    Dp = implicit_grads_1st(
    #        lambda th, lam: k_fn(th, Ztr, Ytr, lam), th, lam, Dg=Dg
    #    )
    #    ret = Dp + grad(lambda lam: loss(th, Zts, Yts, lam))(lam)

    #    if CHECK_GRADS:
    #        Dpz = implicit_grads_1st(
    #            lambda th, lam: k_fn(th, Ztr, Ytr, lam), th, lam
    #        )
    #        Df = Dpz.reshape((th.numel(), lam.numel()))
    #        g = torch.sum(Dg @ Df, -2).reshape(lam.shape)
    #        ret_ = g + grad(lambda lam: loss(th, Zts, Yts, lam))(lam)
    #        g_ = grad(lambda lam: loss(F_fn(Ztr, Ytr, lam), Zts, Yts, lam))(lam)
    #        err = torch.norm(ret - ret_) / torch.norm(ret)
    #        err_ = torch.norm(ret - g_) / torch.norm(ret)
    #        assert err < 1e-5 and err_ < 1e-5
    #    return ret

    #@fn_with_sol_cache(fn, sol_cache)
    #def h_fn(th, lam):
    #    th, lam = th.detach(), lam.detach()

    #    Dg = grad(lambda th: loss(th, Zts, Yts, lam))(th)
    #    Dg = Dg.reshape((th.numel(), 1, 1))
    #    Hg = hessian(lambda th: loss(th, Zts, Yts, lam))(th)
    #    Hg = Hg.reshape((th.numel(),) * 2)
    #    Dp, Dpp = implicit_grads_2nd(
    #        lambda th, lam: k_fn(th, Ztr, Ytr, lam), th, lam, Dg=Dg, Hg=Hg
    #    )
    #    ret = Dpp + hessian(lambda lam: loss(th, Zts, Yts, lam))(lam)
    #    if CHECK_GRADS:
    #        Dpz, Dppz = implicit_grads_2nd(
    #            lambda th, lam: k_fn(th, Ztr, Ytr, lam), th, lam
    #        )

    #        Df = Dpz.reshape((th.numel(), lam.numel()))
    #        H1 = t(Df) @ Hg @ Df

    #        Hf = Dppz.reshape((th.numel(), lam.numel(), lam.numel()))
    #        H2 = torch.sum(Dg * Hf, -3)

    #        H = (H1 + H2).reshape(lam.shape + lam.shape)
    #        ret_ = H + hessian(lambda lam: loss(th, Zts, Yts, lam))(lam)

    #        H_ = hessian(lambda lam: loss(F_fn(Ztr, Ytr, lam), Zts, Yts, lam))(
    #            lam
    #        )
    #        err = torch.norm(ret - ret_) / torch.norm(ret_)
    #        err_ = torch.norm(ret - H_) / torch.norm(H_)

    #        # print("H_err = %9.4e" % err)
    #        assert err < 1e-5 and err_ < 1e-5
    #    return ret
    ##$
    loss_fn_ = lambda th, lam: loss(th, Ztr, Ytr, lam)
    opt_fn_ = lambda lam: F_fn(Ztr, Ytr, lam)
    k_fn_ = lambda th, lam: k_fn(th, Zts, Yts, lam)

    Dpz = implicit_grads_1st(k_fn_, opt_fn_(lam), lam)
    f_fn, g_fn, h_fn = generate_fns(loss_fn_, opt_fn_, k_fn_)
    pdb.set_trace()

    lam0 = 1e-3 * torch.randn(lam.shape)
    VERBOSE = True
    lams = minimize_sqp(
        f_fn, g_fn, h_fn, lam0, verbose=VERBOSE, reg0=1e-9, max_it=30
    )
    print()
    print(loss(F_fn(Ztr, Ytr, lams), Zts, Yts, lams))
    print()

    # lams = minimize_agd(
    #    f_fn, g_fn, lam0, ai=1e-2, af=1e-2, verbose=VERBOSE, max_it=10 ** 3
    # )
    # print()
    # print(loss(F_fn(Ztr, Ytr, lams), Zts, Yts, lams))
    # print()

    #lams = minimize_lbfgs(f_fn, g_fn, lam0, lr=1e0, verbose=VERBOSE, max_it=30)
    #print()
    #print(loss(F_fn(Ztr, Ytr, lams), Zts, Yts, lams))
    #print()

    # lams = minimize_ipopt(f_fn, g_fn, h_fn, lam0, verbose=VERBOSE, max_it=100)
    # print()
    # print(loss(F_fn(Ztr, Ytr, lams), Zts, Yts, lams))
    # print()

    #plt.figure()
    #plt.plot(Xtr, Ytr, label="ctx", color="#15F4EE")
    #plt.plot(Xts, Yts, label="test", color="red")
    #plt.plot(
    #    Xts,
    #    Zts @ F_fn(Ztr, Ytr, lam0),
    #    linestyle="--",
    #    label="guess",
    #    color="green",
    #)
    #plt.plot(
    #    Xts,
    #    Zts @ F_fn(Ztr, Ytr, lams),
    #    linestyle="--",
    #    label="optimal",
    #    color="black",
    #)
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig("lstsq.png", dpi=200)
    ## plt.title("$\\lambda = %6.3e$" % lam_org)
    ## plt.show()
    #plt.draw_all()
    #plt.pause(1e-2)

    pdb.set_trace()
