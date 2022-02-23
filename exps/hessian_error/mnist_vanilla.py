import os, pdb, sys, math, time, gzip, pickle

os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch, numpy as np
from tqdm import tqdm
from collections import OrderedDict as odict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import header

from implicit.interface import init

jaxm = init(dtype=np.float32, device="cuda")
assert jaxm.zeros(()).device().platform == "gpu"

from implicit.implicit import implicit_jacobian
from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit.opt import minimize_agd, minimize_lbfgs
from implicit.nn_tools import nn_all_params, nn_forward_gen

from mnist import train, test

# from fashion import train, test

cat = lambda *args: np.concatenate(list(args))

CONFIG = dict(
    n_train=3 * 10 ** 3,
    lam=-2.5,
    train_it=2 * 10 ** 4,
    refine_it=4000,
    save_its=50,
)


def make_nn():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, (4, 4), 2),
        torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, (2, 2), 2),
        torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, (2, 2), 2),
        torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, (2, 2), 2),
        torch.nn.Tanh(),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 10),
        torch.nn.Softmax(dim=-1),
    )
    # return torch.nn.Sequential(
    #    torch.nn.Conv2d(1, 16, (6, 6), 3),
    #    torch.nn.Tanh(),
    #    torch.nn.Conv2d(16, 16, (2, 2), 2),
    #    torch.nn.Tanh(),
    #    torch.nn.Conv2d(16, 16, (2, 2), 1),
    #    torch.nn.Tanh(),
    #    torch.nn.Conv2d(16, 16, (2, 2), 2),
    #    torch.nn.Tanh(),
    #    torch.nn.Flatten(),
    #    torch.nn.Linear(16 * 1, 16 * 1),
    #    torch.nn.Tanh(),
    #    torch.nn.Linear(16 * 1, 10),
    #    torch.nn.Softmax(dim=-1),
    # )


def loss_fn_gen(fwd_fn):
    def loss_fn(z, lam, X, Y):
        l = jaxm.mean(jaxm.sum((fwd_fn(X, z) - Y) ** 2, -1))
        l = l + (10.0 ** lam) * jaxm.sum(z ** 2)
        return l

    return loss_fn


def main(mode, config):
    STATE, dtype = dict(), jaxm.zeros(()).dtype

    # read in the data na make the NN ######################################
    nn = make_nn()
    Xts = jaxm.array(test["images"]).reshape((-1, 1, 28, 28)).astype(dtype)
    Yts = jaxm.nn.one_hot(jaxm.array(test["labels"]), 10).astype(dtype)

    # generate the train functions #########################################
    fwd_fn = nn_forward_gen(nn)
    loss_fn = loss_fn_gen(fwd_fn)

    @jaxm.jit
    def acc_fn(params, X, Y):
        return jaxm.mean(
            jaxm.argmax(fwd_fn(X, params), -1) == jaxm.argmax(Y, -1)
        )

    f_fn_ = jaxm.jit(loss_fn)
    g_fn_ = jaxm.jit(jaxm.grad(f_fn_))

    def f_fn(z):
        nonlocal STATE
        X, Y = STATE["X"], STATE["Y"]
        return f_fn_(z, config["lam"], X, Y)

    def g_fn(z):
        nonlocal STATE
        X, Y = STATE["X"], STATE["Y"]
        ret = g_fn_(z, config["lam"], X, Y)
        return ret

    # define callback logic ################################################
    STATE["param_hist"], STATE["it"] = odict(), 0
    train_it = jaxm.round(
        jaxm.linspace(0, config["train_it"] + 1, config["save_its"] // 2)
    )[1:]
    refine_it = jaxm.round(
        jaxm.linspace(0, config["refine_it"] + 1, config["save_its"] // 2)
    )[1:]
    STATE["save_its"] = jaxm.cat([train_it, train_it[-1] + refine_it])
    STATE["train_phase"] = True

    def callback_fn(z, opt=None):
        nonlocal STATE
        it = STATE["it"]
        STATE["train_phase"] = it < config["train_it"]
        if (it + 1) % 100 == 0:
            tqdm.write("Acc %6.2f%%" % (1e2 * acc_fn(z, Xts, Yts)))

        if it in STATE["save_its"]:
            STATE["param_hist"][it] = np.array(z)

        Xtr, Ytr = STATE["Xtr"], STATE["Ytr"]
        if STATE["train_phase"]:
            ridx = jaxm.randint(0, Xtr.shape[0], (config["n_train"],))
            STATE["X"], STATE["Y"] = Xtr[ridx, ...], Ytr[ridx, ...]
        else:
            STATE["X"], STATE["Y"] = Xtr, Ytr
            opt.param_groups[0]["lr"] = 1e-4

        STATE["it"] += 1

    fname = "data/results.pkl.gz"
    opts = dict(
        verbose=True,
        callback_fn=callback_fn,
        ai=1e-4,
        af=1e-4,
        max_it=config["train_it"] + config["refine_it"],
        use_writer=True,
    )
    if mode == "train":
        Xtr = jaxm.array(train["images"]).reshape((-1, 1, 28, 28)).astype(dtype)
        Ytr = jaxm.nn.one_hot(jaxm.array(train["labels"]), 10).astype(dtype)
        STATE["Xtr"], STATE["Ytr"] = Xtr, Ytr
        params0 = nn_all_params(nn)
        print("params.shape =", params0.shape)
        paramss = minimize_agd(f_fn, g_fn, params0, **opts)

        with gzip.open(fname, "wb") as fp:
            pickle.dump(
                dict(
                    param_hist=STATE["param_hist"],
                    config=config,
                    X=np.array(Xtr),
                    Y=np.array(Ytr),
                ),
                fp,
            )
    elif mode == "eval":
        assert len(sys.argv) >= 3
        CONFIG = dict(idx=int(sys.argv[2]))

        with gzip.open(fname, "rb") as fp:
            results = pickle.load(fp)
        X, Y = jaxm.array(results["X"]), jaxm.array(results["Y"])
        # ridx = jaxm.randint(0, X.shape[0], size=(100,))
        # X, Y = X[ridx, ...], Y[ridx, ...]

        lam = jaxm.array(config["lam"])

        fname = "data/Dpz_%03d_2.pkl.gz"
        idx, results_dpz = 0, odict()
        its = np.array(list(results["param_hist"].keys()))
        print(len(its))
        for (idx, it) in enumerate(its):
            if idx == CONFIG["idx"]:
                results_dpz[it] = dict()
                z = jaxm.array(results["param_hist"][it])
                print("Acc: %5.2f%%" % (1e2 * acc_fn(z, Xts, Yts)))
                print("||g|| = %9.4e" % jaxm.norm(g_fn_(z, lam, X, Y)))

                hi_fn = jaxm.grad(
                    lambda z, lam, i: g_fn_(z, lam, X, Y).reshape(-1)[i]
                )
                hi_fn = jaxm.jit(hi_fn)
                H = []
                for i in tqdm(range(z.size)):
                    H.append(hi_fn(z, lam, i))
                Dzk = jaxm.stack(H, -1)

                results_dpz[it]["Dzk"] = np.array(Dzk)
                optimizations = dict(Dzk=Dzk)
                Dpz = implicit_jacobian(
                    lambda z, lam: g_fn_(z, lam, X, Y),
                    z,
                    lam,
                    optimizations=optimizations,
                )
                results_dpz[it]["Dpz"] = np.array(Dpz)
                results_dpz[it]["Dzl"] = np.array(g_fn_(z, lam, X, Y))
                with gzip.open(fname % idx, "wb") as fp:
                    pickle.dump(results_dpz, fp)
    else:
        print("Nothing to be done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) >= 2 else "train", CONFIG)
