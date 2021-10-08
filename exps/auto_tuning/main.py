import pdb, os, sys, time, gzip, pickle, math
from pprint import pprint
from collections import OrderedDict as odict

import matplotlib.pyplot as plt, numpy as np, torch
from tqdm import tqdm
from sklearn.cluster import k_means
import line_profiler

import header

from implicit.interface import init

# jaxm = init(dtype=np.float32, device="cuda")
jaxm = init(dtype=np.float64, device="cpu")

from implicit.opt import minimize_sqp, minimize_agd, minimize_lbfgs
import implicit.utils as utl
from implicit.implicit import implicit_jacobian, implicit_hessian
from implicit.implicit import generate_fns
from implicit.pca import visualize_landscape

import mnist

# import fashion as mnist

from objs import LS, OPT_with_centers, CE, OPT_with_diag, OPT_conv
from objs import LS_with_diag, poly_feat

PRINT_FN = lambda *args: tqdm.write(" ".join([str(x) for x in args]))


def acc_fn(Yp, Y):
    return jaxm.mean(1e2 * (jaxm.argmax(Yp, -1) == jaxm.argmax(Y, -1)))


def loss_fn(Yp, Y, param):
    return -jaxm.mean(jaxm.log(jaxm.sum(jaxm.softmax(Yp, -1) * Y, -1)))


def get_mnist_data(dataset, n=-1, Xp=None, Yp=None):
    X, Y = dataset["images"], dataset["labels"]
    if n <= 0:
        r = np.arange(X.shape[-2])
    else:
        r = np.random.randint(X.shape[-2], size=(n,))
    X = jaxm.array(X[r, :])
    Y = jaxm.nn.one_hot(jaxm.array(Y[r]), 10)
    X, Xp = utl.normalize(X, params=Xp)
    Y, Yp = utl.normalize(Y, params=Yp)
    return (X, Xp), (Y, Yp)


def get_centers(X, Y, n=1):
    ys = jaxm.unique(jaxm.argmax(Y, -1))
    centers = [None for y in ys]
    for (i, y) in enumerate(ys):
        mask = jaxm.argmax(Y, -1) == y
        centers[i] = k_means(X[mask, :], n_clusters=n)[0]
    centers = np.concatenate(centers, -2)
    return jaxm.array(centers)


def main(config):
    results = dict()
    np.random.seed(config["seed"])
    jaxm.manual_seed(config["seed"])
    n_tr, n_ts = config["train_n"], config["test_n"]
    # read in the parameters ########################################
    Xp, Yp = None, (0.0, 1.0)
    fname = "data/cache2.pkl.gz"
    try:
        with gzip.open(fname, "rb") as fp:
            Xp, Yp, centers = pickle.load(fp)
    except FileNotFoundError:
        (X_all, Xp), (Y_all, Yp) = get_mnist_data(mnist.train, Xp=Xp, Yp=Yp)
        centers = get_centers(X_all, Y_all)
        with gzip.open(fname, "wb") as fp:
            pickle.dump((Xp, Yp, centers), fp)

    (Xtr, Xtrp), (Ytr, Ytrp) = get_mnist_data(mnist.train, n_tr, Xp=Xp, Yp=Yp)
    (Xts, Xtsp), (Yts, Ytsp) = get_mnist_data(mnist.test, n_ts, Xp=Xp, Yp=Yp)

    assert config["opt_low"] in ["ls", "ce"]
    OPT = CE(max_it=50) if config["opt_low"] == "ce" else LS()
    lam0 = -4.0
    if config["fmap"] == "vanilla":
        Ztr = poly_feat(Xtr, n=1)
        Zts = poly_feat(Xts, n=1)
        OPT = OPT
        param = lam0 * jaxm.ones(1)
    elif config["fmap"] == "centers":
        Ztr = poly_feat(Xtr, n=1, centers=centers)
        Zts = poly_feat(Xts, n=1, centers=centers)
        OPT = OPT_with_centers(OPT, centers.shape[-2])
        param = jaxm.array([-3.0, lam0])
    elif config["fmap"] == "diag":
        Ztr = poly_feat(Xtr, n=1)
        Zts = poly_feat(Xts, n=1)
        #OPT = OPT_with_diag(OPT)
        OPT = LS_with_diag()
        if config["opt_low"] == "ce":
            raise NotImplementedError
            param = lam0 * jaxm.ones(Ztr.shape[-1] * (Ytr.shape[-1] - 1))
        elif config["opt_low"] == "ls":
            param = lam0 * jaxm.ones(Ztr.shape[-1] * Ytr.shape[-1] + 1)
            # param = lam0 * jaxm.ones(Ztr.shape[-1])
    elif config["fmap"] == "conv":
        Ztr, Zts = Xtr, Xts
        in_channels, out_channels = 1, 2
        kernel_size, stride = 3, config["conv_size"]
        OPT = OPT_conv(OPT, in_channels, out_channels, kernel_size, stride)
        param = OPT.generate_parameter()
    else:
        raise ValueError

    print("#" * 80)
    print(jaxm.norm(param))
    print("#" * 80)

    print("Starting solve")
    W = OPT.solve(Ztr, Ytr, param)
    print("Solved")
    Yp_tr = OPT.pred(W, Ztr, param)
    Yp_ts = OPT.pred(W, Zts, param)
    loss_tr, acc_tr = loss_fn(Yp_tr, Ytr, param), acc_fn(Yp_tr, Ytr)
    loss_ts, acc_ts = loss_fn(Yp_ts, Yts, param), acc_fn(Yp_ts, Yts)
    print("Loss: %9.4e" % loss_tr)
    print("Accuracy: %3.2f%%" % acc_tr)
    print("Loss: %9.4e" % loss_ts)
    print("Accuracy: %3.2f%%" % acc_ts)
    results["before"] = dict(
        loss_tr=float(loss_tr),
        acc_tr=float(acc_tr),
        loss_ts=float(loss_ts),
        acc_ts=float(acc_ts),
    )

    # define functions ##############################################
    loss_fn_ = lambda W, param: loss_fn(OPT.pred(W, Zts, param), Yts, param)
    opt_fn_ = lambda param: OPT.solve(Ztr, Ytr, param)
    k_fn_ = lambda W, param: OPT.grad(W, Ztr, Ytr, param)
    Dzk_solve_ = lambda W, param, rhs=None, T=False: OPT.Dzk_solve(
        W, Ztr, Ytr, param, rhs=rhs, T=T
    )

    W = opt_fn_(param)
    k = k_fn_(W, param)

    # Dpz = implicit_jacobian(k_fn_, W, param, Dzk_solve_fn=Dzk_solve_)
    optimizations = dict(Dzk_solve_fn=Dzk_solve_)
    f_fn, g_fn, h_fn = generate_fns(
        loss_fn_, opt_fn_, k_fn_, optimizations=optimizations
    )
    # f_fn(param), g_fn(param), h_fn(param)

    hist = dict(loss_ts=odict(), acc_ts=odict())

    def callback_fn(param, **kw):
        nonlocal hist, f_fn
        W = OPT.solve(Ztr, Ytr, param)
        Yp_ts = OPT.pred(W, Zts, param)
        loss_ts = float(loss_fn(Yp_ts, Yts, param))
        acc_ts = float(acc_fn(Yp_ts, Yts))
        PRINT_FN("Accuracy: %5.2f%%" % acc_ts)
        idx = len(f_fn.cache.keys())  # function evaluations so far
        hist["loss_ts"][idx] = loss_ts
        hist["acc_ts"][idx] = acc_ts

    oopt = dict(verbose=True, callback_fn=callback_fn, full_output=True)
    oopt["max_it"] = (
        min(10 ** config["max_it"], 500)
        if config["solver"] == "agd"
        else config["max_it"]
    )
    opt_fns = [f_fn, g_fn, h_fn] if config["solver"] == "sqp" else [f_fn, g_fn]
    if config["solver"] == "sqp":
        param, param_hist = minimize_sqp(
            *opt_fns, param, reg0=1e-9, ls_pts_nb=5, force_step=True, **oopt
        )
    elif config["solver"] == "ipopt":
        param = minimize_ipopt(*opt_fns, param, **oopt)
    elif config["solver"] == "lbfgs":
        param, param_hist = minimize_lbfgs(*opt_fns, param, lr=1e-1, **oopt)
    elif config["solver"] == "agd":
        param, param_hist = minimize_agd(
            *opt_fns, param, ai=1e-2, af=1e-2, **oopt
        )
    else:
        raise ValueError

    W = OPT.solve(Ztr, Ytr, param)
    Yp_tr = OPT.pred(W, Ztr, param)
    Yp_ts = OPT.pred(W, Zts, param)
    loss_tr, acc_tr = loss_fn(Yp_tr, Ytr, param), acc_fn(Yp_tr, Ytr)
    loss_ts, acc_ts = loss_fn(Yp_ts, Yts, param), acc_fn(Yp_ts, Yts)
    print("Loss: %9.4e" % loss_tr)
    print("Accuracy: %3.2f%%" % acc_tr)
    print("Loss: %9.4e" % loss_ts)
    print("Accuracy: %3.2f%%" % acc_ts)

    results["after"] = dict(
        loss_tr=float(loss_tr),
        acc_tr=float(acc_tr),
        loss_ts=float(loss_ts),
        acc_ts=float(acc_ts),
    )

    return param, results, hist


if __name__ == "__main__":
    assert len(sys.argv) >= 2
    idx_job = int(sys.argv[1])
    seeds_fname = "data/seeds.pkl.gz"
    # if idx_job == 0 and os.path.isfile(seeds_fname):
    #    os.remove(seeds_fname)

    # fmaps = ["centers", "diag", "conv", "vanilla"]
    # opt_lows = ["ls", "ce"]
    # solvers = ["agd", "sqp", "lbfgs"]
    fmaps = ["diag"]
    opt_lows = ["ls"]
    solvers = ["sqp"]

    trials = range(10)

    params = [
        (fmap, solver, opt_low, trial)
        for fmap in fmaps
        for opt_low in opt_lows
        for trial in trials
        for solver in solvers
    ]
    print(len(params))

    all_results = dict()
    try:
        with gzip.open(seeds_fname, "rb") as fp:
            seeds = pickle.load(fp)
    except FileNotFoundError:
        seeds = {
            (fmap, trial): np.random.randint(2 ** 31)
            for fmap in fmaps
            for trial in trials
        }
        with gzip.open(seeds_fname, "wb") as fp:
            pickle.dump(seeds, fp)

    for (idx, setting) in enumerate(params):
        if idx != idx_job:
            continue
        fmap, solver, opt_low, trial = setting
        config = dict(
            seed=seeds[(fmap, trial)],
            train_n=10 ** 3,
            test_n=10 ** 4,
            opt_low=opt_low,
            conv_size=2,
            solver=solver,
            max_it=15,
            fmap=fmap,
        )
        print("#" * 80)
        pprint(config)
        param, results, hist = main(config)
        all_results[setting] = dict(results=results, hist=hist)
    # with gzip.open("data/all_results_%03d.pkl.gz" % idx_job, "wb") as fp:
    #    pickle.dump(all_results, fp)
