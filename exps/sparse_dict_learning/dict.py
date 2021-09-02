import os, sys, time, pdb, math

import torch, numpy as np, matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from sklearn.linear_model import ElasticNet

import header

from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd
from implicit import implicit_jacobian, implicit_hessian, generate_fns
from implicit import JACOBIAN, HESSIAN_DIAG, grad
from implicit.utils import t2n, n2t

import brca_dataset as brca
from lasso import optimize_admm, optimize_sklearn
from lasso import optimize_cvxpy

fill_fn = lambda W: (W.abs() > 1e-5).to(torch.float32).mean()


def feat_map(X):
    Z = torch.cat([X[..., :1] ** 0, X], -1)
    if not hasattr(feat_map, "A_sketch"):
        in_dim, out_dim = Z.shape[-1], 1000
        norm = (in_dim + out_dim) / 2
        opts = dict(device=X.device, dtype=X.dtype)
        feat_map.A_sketch = torch.randn((in_dim, out_dim), **opts) / norm
    return Z @ feat_map.A_sketch


def find_bet2(fn, l=-9, u=1, max_it=20, target_val=0.5):
    it = 0
    fl = fn(l)
    while (u - l) > 1e-3:
        m = (l + u) / 2.0
        fm = fn(m)
        if (fl < target_val) == (fm < target_val):
            l = m
        else:
            u = m
        it += 1
        if it >= max_it:
            break
    return (l + u) / 2.0


METHOD = "sklearn"


class SC2:
    def __init__(self, k=None, e=None):
        # assert k is not None and e is not None
        self.k, self.e = k, e
        self.SC = SC(k=self.k, e=self.e)

    def solve(self, Z, *params, method=METHOD):
        th, bet1 = params
        bet2 = find_bet2(
            lambda bet2: fill_fn(
                self.SC.solve(Z, th, bet1, bet2, method=METHOD)
            )
        )
        return self.SC.solve(Z, th, bet1, bet2, method=method)

    def fval(self, X, Z, *params):
        th, bet1 = params
        bet2 = find_bet2(
            lambda bet2: fill_fn(
                self.SC.solve(Z, th, bet1, bet2, method=METHOD)
            )
        )
        return self.SC.fval(X, Z, th, bet1, bet2, method=method)

    def grad(self, X, Z, *params):
        th, bet1 = params
        bet2 = find_bet2(
            lambda bet2: fill_fn(
                self.SC.solve(Z, th, bet1, bet2, method=METHOD)
            )
        )
        return self.SC.grad(X, Z, th, bet1, bet2)

    def hess(self, X, Z, *params, small=False):
        th, bet1 = params
        bet2 = find_bet2(
            lambda bet2: fill_fn(
                self.SC.solve(Z, th, bet1, bet2, method=METHOD)
            )
        )
        return self.SC.hess(X, Z, th, bet1, bet2, small=small)

    def Dzk_solve(self, X, Z, *params, rhs=None, T=False):
        th, bet1 = params
        bet2 = find_bet2(
            lambda bet2: fill_fn(
                self.SC.solve(Z, th, bet1, bet2, method=METHOD)
            )
        )
        ret = self.SC.Dzk_solve(X, Z, th, bet1, bet2, rhs=rhs, T=T)
        return ret


class SC:
    def __init__(self, k=None, e=None):
        # assert k is not None and e is not None
        self.k, self.e = k, e

    def solve(self, Z, *params, method=METHOD):
        th, bet1, bet2 = params
        bet1, bet2 = 10.0 ** bet1, 10.0 ** bet2
        method = method.lower()
        assert method in ["cvx", "admm", "sklearn", "cvxpy"]

        # t_ = lambda: time.perf_counter()
        # t = t_()
        # ret1 = optimize_sklearn(th.T, Z.T, bet1, bet2).T
        # print("Elapsed %9.4e s" % (t_() - t))
        # t = t_()
        # ret2 = optimize_admm(th.T, Z.T, bet1, bet2, rho=1e-1, verbose=False).T
        # print("Elapsed %9.4e s" % (t_() - t))
        # t = t_()
        # ret3 = optimize_cvxpy(th.T, Z.T, bet1, bet2).T
        # print("Elapsed %9.4e s" % (t_() - t))

        # g1 = self.grad(ret1, Z, *params)
        # g2 = self.grad(ret2, Z, *params)
        # g3 = self.grad(ret3, Z, *params)
        # print("ret1.fill = %5.2f%%" % (fill_fn(ret1).cpu() * 1e2))
        # print("ret2.fill = %5.2f%%" % (fill_fn(ret2).cpu() * 1e2))
        # print("ret3.fill = %5.2f%%" % (fill_fn(ret3).cpu() * 1e2))
        # print("g1.norm =   %9.4e" % (g1.norm().cpu()))
        # print("g2.norm =   %9.4e" % (g2.norm().cpu()))
        # print("g3.norm =   %9.4e" % (g3.norm().cpu()))

        if method == "sklearn":
            return optimize_sklearn(th.T, Z.T, bet1, bet2).T
        elif method == "cvx" or method == "cvxpy":
            return optimize_cvxpy(th.T, Z.T, bet1, bet2).T
        elif method == "admm":
            return optimize_admm(th.T, Z.T, bet1, bet2).T

    def fval(self, X, Z, *params):
        th, bet1, bet2 = params
        bet1, bet2 = 10.0 ** bet1, 10.0 ** bet2
        obj = (
            0.5 * torch.sum((X @ th - Z) ** 2) / Z.shape[-2]
            + 0.5 * bet1 * torch.sum(X ** 2)
            + bet2 * torch.sum(torch.abs(X))
        )
        return obj

    def grad(self, X, Z, *params):
        th, bet1, bet2 = params
        bet1, bet2 = 10.0 ** bet1, 10.0 ** bet2
        # J = JACOBIAN(lambda X: self.fval(X, Z, th, bet1, bet2), X)
        J2 = (X @ th - Z) @ th.T / Z.shape[-2] + bet1 * X + bet2 * torch.sign(X)
        return J2

    def hess(self, X, Z, *params, small=False):
        th, bet1, bet2 = params
        bet1, bet2 = 10.0 ** bet1, 10.0 ** bet2
        H_block = th @ th.T / Z.shape[-2]
        if small == True:
            return H_block
        H2 = torch.block_diag(*[H_block for _ in range(X.shape[-2])])
        H2.diagonal()[:] += bet1
        # H = JACOBIAN(lambda X: self.grad(X, Z, th, bet1, bet2), X)
        # err = torch.max((H.reshape((X.numel(),) * 2) - H2).abs())
        return H2

    def Dzk_solve(self, X, Z, *params, rhs=None, T=False):
        th, bet1, bet2 = params
        bet1, bet2 = 10.0 ** bet1, 10.0 ** bet2
        H_block = self.hess(X, Z, *params, small=True)
        # sol2 = torch.linalg.solve(self.hess(X, Z, *params), rhs)
        rhs_shape = rhs.shape
        rhs = rhs.reshape(X.shape + (-1,))
        sol = torch.cholesky_solve(
            rhs, torch.linalg.cholesky(H_block[None, :, :])
        )
        sol = sol.reshape(rhs_shape)
        return sol


if __name__ == "__main__":
    device, dtype = "cuda", torch.float32
    opts = dict(device=device, dtype=dtype)

    X, Y = torch.tensor(brca.X, **opts), torch.tensor(brca.Y, **opts)
    r = torch.randperm(X.shape[0], device=device)
    X, Y = X[r, :], Y[r]
    n = X.shape[0]
    splits = [round(0.6 * n), round(0.2 * n)]
    splits = splits + [n - sum(splits)]
    Xtr, Xvl, Xts = X.split(splits, dim=0)
    Ytr, Yvl, Yts = Y.split(splits, dim=0)
    Ztr, Zvl, Zts = [feat_map(X) for X in (Xtr, Xvl, Xts)]

    OPT = SC2(k=10, e=Ztr.shape[-1])
    th = torch.randn((OPT.k, OPT.e), **opts) / (OPT.k + OPT.e) * 2
    bet1, bet2 = torch.as_tensor([-2.0, -6.0], **opts)
    # params = (th, bet1, bet2)
    params = (th, bet1)
    Xtr = OPT.solve(Ztr, *params, method=METHOD)

if True and __name__ == "__main__":
    # prepare encoding and functions for this encoding ###################
    Xtr = OPT.solve(Ztr, *params)
    g = OPT.grad(Xtr, Ztr, *params)

    k_fn = lambda z, th, bet1: OPT.grad(z, Ztr, th, bet1, *params[2:])
    opt_fn = lambda th, bet1: OPT.solve(
        Ztr, th, bet1, *params[2:], method=METHOD
    )
    W = torch.randn((OPT.k, 1), **opts)
    loss_fn = lambda z, th, bet1: torch.nn.BCELoss()(
        torch.sigmoid(z @ loss_fn.W).reshape(-1), loss_fn.Ytr
    )
    loss_fn.W, loss_fn.Ytr = W, Ytr

    # optimize W for this theta ##########################################
    def fW_fn(W):
        loss_fn.W = W
        return loss_fn(Xtr, *params)

    gW_fn = lambda W: JACOBIAN(fW_fn, W)
    hW_fn = lambda W: HESSIAN_DIAG(fW_fn, W)[0]

    W.requires_grad = True
    W = minimize_sqp(fW_fn, gW_fn, hW_fn, W, verbose=True, max_it=100)

    acc = (torch.sigmoid(Xtr @ W).reshape(-1).round() == Ytr).to(dtype).mean()
    print("Accuracy on train: %5.2f%%" % (1e2 * acc))

    Xts = OPT.solve(Zts, *params)
    acc = (torch.sigmoid(Xts @ W).reshape(-1).round() == Yts).to(dtype).mean()
    print("Accuracy on test:  %5.2f%%" % (1e2 * acc))

    # optimize parameters jointly ########################################
    Dzk_solve_fn = lambda z, th, bet1, rhs=None, T=False: OPT.Dzk_solve(
        z, Ztr, th, bet1, *params[2:], rhs=rhs, T=T
    )
    optimizations = dict(Dzk_solve_fn=Dzk_solve_fn)

    f_fn, g_fn, h_fn = generate_fns(
        loss_fn, opt_fn, k_fn, optimizations=optimizations
    )
    import line_profiler as lp

    prof = lp.LineProfiler(g_fn.fn, implicit_jacobian, SC.Dzk_solve)
    minimize_sqp = prof.wrap_function(minimize_sqp)
    minimize_lbfgs = prof.wrap_function(minimize_lbfgs)
    minimize_agd = prof.wrap_function(minimize_agd)

    t = time.perf_counter()
    f = f_fn(*params)
    print("Time elapsed %9.4e s" % (time.perf_counter() - t))

    t = time.perf_counter()
    g = g_fn(*params)
    print("Time elapsed %9.4e s" % (time.perf_counter() - t))

    t = time.perf_counter()
    h = h_fn(*params)
    print("Time elapsed %9.4e s" % (time.perf_counter() - t))

    def f_fn_(th, bet1, W):
        loss_fn.W = W
        return f_fn(th, bet1)

    def g_fn_(th, bet1, W):
        loss_fn.W = W
        gs = (*g_fn(th, bet1), gW_fn(W))
        return gs

    ths, bet1s, Ws = minimize_lbfgs(
        f_fn_,
        g_fn_,
        th,
        bet1,
        W,
        verbose=True,
        max_it=5,
        lr=1e-1,
    )
    Xtr = OPT.solve(Ztr, ths, bet1s)
    acc = (torch.sigmoid(Xtr @ W).reshape(-1).round() == Ytr).to(dtype).mean()
    print("Accuracy on train: %5.2f%%" % (1e2 * acc))

    Xts = OPT.solve(Zts, ths, bet1s)
    acc = (torch.sigmoid(Xts @ W).reshape(-1).round() == Yts).to(dtype).mean()
    print("Accuracy on test:  %5.2f%%" % (1e2 * acc))

    prof.print_stats(output_unit=1e-3)

    pdb.set_trace()

if False and __name__ == "__main__":
    bet2s, ys = torch.linspace(-9, 1, 100), []
    for bet2 in tqdm(bet2s):
        try:
            val = t2n(fill_fn(OPT.solve(Ztr, params[0], params[1], bet2)))
        except:
            val = math.nan
        ys.append(val)
    plt.plot(t2n(bet2s), ys)
    plt.scatter(t2n(bet2s), ys)
    plt.show()

    pdb.set_trace()
