import sys, os, math, pdb, time, pickle, gzip
from shutil import rmtree

os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch, matplotlib.pyplot as plt, numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "auto_tuning"))

import header

from mnist import train, test
import implicit
from implicit import generate_fns, implicit_jacobian
from implicit.opt import minimize_lbfgs, minimize_agd, minimize_sqp
from implicit import utils
from objs import LS, CE
from tv import tv, tv_l1
import vis

from feat_map import feat_map

OPT = CE()

def k_fn(z, *params):
    self = k_fn
    lam, Y = self.lam, self.Y
    (X,) = params
    Z, W = feat_map(X), z
    return OPT.grad(W, Z, Y, lam)


def opt_fn(*params):
    self = opt_fn
    lam, Y = self.lam, self.Y
    (X,) = params
    Z = feat_map(X)
    return OPT.solve(Z, Y, lam, max_it=50)


def loss_fn(z, *params):
    self = loss_fn
    Zts, Yts, lam = self.Zts, self.Yts, self.lam
    params0 = loss_fn.params0
    (X,) = params
    Z, W = feat_map(X), z
    # l = torch.nn.MSELoss()(OPT.pred(W, Zts, lam), Yts)
    l = torch.nn.CrossEntropyLoss()(
        OPT.pred(W, Zts, lam), torch.argmax(Yts, -1)
    )
    # ltv = 3e-2 * tv_l1(X.reshape((-1, 28, 28))).mean()
    # l = l + ltv
    opts = dict(device=z.device, dtype=z.dtype)
    maximum = lambda a, b: torch.log(torch.exp(a) + torch.exp(b))
    # l_dev = 1e-2 * sum(
    #    torch.sum(
    #        maximum((param - param0).abs() - 1.0, torch.zeros((), **opts))
    #        ** 2
    #    )
    #    for (param, param0) in zip(params, params0)
    # )
    # l_dev = 1e-1 * sum(
    #    torch.mean((param - param0) ** 2)
    #    for (param, param0) in zip(params, params0)
    # )
    l_reg = 1e-3 * sum(torch.norm(param) ** 2 for param in params)
    l = l + l_reg
    return l


actions = sys.argv[1:]

if __name__ == "__main__":
    dirname = os.path.join(os.path.dirname(__file__), "runs")
    if os.path.isdir(dirname):
        rmtree(dirname)

    device, dtype = "cuda", torch.float32
    opts = dict(device=device, dtype=dtype)
    tensor = lambda x: torch.as_tensor(x, **opts)

    Xtr = tensor(train["images"])
    Ytr = torch.nn.functional.one_hot(
        tensor(train["labels"]).to(torch.long)
    ).to(dtype)
    Xts = tensor(test["images"])
    Yts = torch.nn.functional.one_hot(tensor(test["labels"]).to(torch.long)).to(
        dtype
    )
    lam = tensor(-3.0)

    Xtrp = (torch.mean(Xtr, dim=-2), torch.std(Xtr, dim=-2) + 1e-1)
    Xtr = (Xtr - Xtrp[0]) / Xtrp[1]
    Xtsp = (torch.mean(Xts, dim=-2), torch.std(Xts, dim=-2) + 1e-1)
    Xts = (Xts - Xtsp[0]) / Xtsp[1]

    # r = torch.randint(Xts.shape[-2], size=(10 ** 1,), device=device)
    # Xts, Yts = Xts[r, :], Yts[r, :]

    loss_fn.Zts, loss_fn.Yts, loss_fn.lam = feat_map(Xts), Yts, lam
    Y = torch.nn.functional.one_hot(torch.arange(10)).to(dtype).to(device)
    opt_fn.lam, opt_fn.Y = lam, Y
    k_fn.lam, k_fn.Y = lam, Y

if "eval" in actions:
    with gzip.open("data/data_ls_rand.pkl.gz", "rb") as fp:
        X_ls_rand = tensor(pickle.load(fp))
    with gzip.open("data/data_ls_mean.pkl.gz", "rb") as fp:
        X_ls_mean = tensor(pickle.load(fp))
    with gzip.open("data/data_ce_rand.pkl.gz", "rb") as fp:
        X_ce_rand = tensor(pickle.load(fp))
    with gzip.open("data/data_ce_mean.pkl.gz", "rb") as fp:
        X_ce_mean = tensor(pickle.load(fp))

    names = ["ls_rand", "ls_mean", "ce_rand", "ce_mean"]
    for (i, X) in enumerate([X_ls_rand, X_ls_mean, X_ce_rand, X_ce_mean]):
        W = opt_fn(X)
        Yp_ts = OPT.pred(W, loss_fn.Zts, lam)
        l = torch.nn.CrossEntropyLoss()(Yp_ts, torch.argmax(Yts, -1))
        acc = (Yp_ts.argmax(-1) == loss_fn.Yts.argmax(-1)).to(W.dtype).mean()
        print(names[i])
        print("loss = %9.4e" % l)
        print("acc =  %9.4e" % acc)
        print()

if "train" in actions:
    with gzip.open("data/data_ls_rand.pkl.gz", "rb") as fp:
        X = tensor(pickle.load(fp))
    #X = torch.randn((10, Xtr.shape[-1]), **opts) / math.sqrt(Xtr.shape[-1])
    #X = torch.stack(
    #    [Xtr[torch.argmax(Ytr, -1) == i, :].mean(-2) for i in range(10)]
    #)
    #X = torch.stack(
    #    [Xtr[torch.argmax(Ytr, -1) == i, :][0, :] for i in range(10)]
    #)

    params = (X,)
    loss_fn.params0 = tuple(param.detach().clone() for param in params)

    W = opt_fn(*params)
    k = k_fn(W, *params)
    W = W + torch.randn(W.shape, **opts)
    k2 = k_fn(W, *params)

    Dzk_solve = lambda z, *params, rhs=None, T=False: OPT.Dzk_solve(
        z, *params, k_fn.Y, k_fn.lam, rhs=rhs, T=T
    )
    optimizations = dict(Dzk_solve_fn=Dzk_solve)
    f_fn, g_fn, h_fn = generate_fns(
        loss_fn, opt_fn, k_fn, optimizations=optimizations
    )

    writer = SummaryWriter()

    def callback_fn(*params):
        self = callback_fn
        self.it = self.it if hasattr(self, "it") else 0
        if (self.it + 1) % 10 == 0:
            W = opt_fn(*params)
            Yp = OPT.pred(W, loss_fn.Zts, loss_fn.lam)
            acc = (
                (Yp.argmax(dim=-1) == loss_fn.Yts.argmax(dim=-1))
                .to(torch.float32)
                .mean()
            )
            #vis.main(feat_map(params[0]))
            writer.add_scalar("acc", float(acc), self.it)
            writer.flush()
            print("Accuracy of trained %3.1f%%" % (acc * 1e2))
        self.it += 1

    method = "agd"
    opt_opts = dict(callback_fn=callback_fn, verbose=True, use_writer=True)
    fns = [f_fn, g_fn] if method != "sqp" else [f_fn, g_fn, h_fn]
    if method == "agd":
        opt_opts = dict(opt_opts, ai=1e-2, af=1e-2, max_it=10 ** 3)
        minimize_fn = minimize_agd
    elif method == "lbfgs":
        opt_opts = dict(opt_opts, lr=1e-1, max_it=10)
        minimize_fn = minimize_lbfgs
    elif method == "sqp":
        opt_opts = dict(opt_opts, max_it=10, force_step=True)
        minimize_fn = minimize_sqp

    Xs = minimize_fn(*fns, *params, **opt_opts)
    with gzip.open("data/data_ce_rand.pkl.gz", "wb") as fp:
        pickle.dump(Xs.cpu().numpy(), fp)
