import pdb, os, sys, time, gzip, pickle, math
from pprint import pprint

import matplotlib.pyplot as plt, numpy as np, torch
from tqdm import tqdm
from sklearn.cluster import k_means

sys.path.append(os.path.abspath(os.path.join(os.dirname(__file__), "..")))

from utils import t, diag, topts, fn_with_sol_cache
from opt import minimize_sqp, minimize_agd, minimize_lbfgs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from implicit import implicit_grads_1st, implicit_grads_2nd
from implicit import generate_fns
import mnist

from objs import LS, OPT_with_centers, CE, OPT_with_diag, OPT_conv

torch.set_default_dtype(torch.float64)


def acc_fn(Yp, Y):
    return torch.mean(1e2 * (torch.argmax(Yp, -1) == torch.argmax(Y, -1)))


def poly_feat(X, n=1, centers=None):
    Z = torch.cat([X[..., 0:1] ** 0] + [X ** i for i in range(1, n + 1)], -1)
    if centers is not None:
        t_ = time.time()
        dist = torch.norm(X[..., None, :] - centers, dim=-1) / X.shape[-1]
        Z = torch.cat([Z, dist], -1)
    return Z


def loss_fn(Yp, Y, param):
    return torch.nn.functional.cross_entropy(Yp, torch.argmax(Y, -1))


def get_mnist_data(dataset, n=-1, dtype=None, Xp=None, Yp=None):
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    X, Y = dataset["images"], dataset["labels"]
    if n <= 0:
        r = np.arange(X.shape[-2])
    else:
        r = np.random.randint(X.shape[-2], size=(n,))
    X = torch.tensor(X[r, :], dtype=dtype)
    Y = utl.onehot(torch.tensor(Y[r], dtype=dtype), num_classes=10)
    X, Xp = utl.normalize(X, params=Xp)
    Y, Yp = utl.normalize(Y, params=Yp)
    return (X, Xp), (Y, Yp)


def get_centers(X, Y, n=1):
    ys = torch.unique(torch.argmax(Y, dim=-1))
    centers = [None for y in ys]
    for (i, y) in enumerate(ys):
        mask = torch.argmax(Y, dim=-1) == y
        centers[i] = k_means(X[mask, :], n_clusters=n)[0]
    centers = np.concatenate(centers, -2)
    #with gzip.open(fname, "wb") as fp:
    #    pickle.dump(centers, fp)
    return torch.as_tensor(centers, **topts(X))


def main(config):
    results = dict()
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
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
    OPT = CE() if config["opt_low"] == "ce" else LS()
    lam0 = -3.0
    if config["fmap"] == "vanilla":
        Ztr = poly_feat(Xtr, n=1)
        Zts = poly_feat(Xts, n=1)
        OPT = OPT
        param = -lam0 * torch.ones(1)
    elif config["fmap"] == "centers":
        Ztr = poly_feat(Xtr, n=1, centers=centers)
        Zts = poly_feat(Xts, n=1, centers=centers)
        OPT = OPT_with_centers(OPT, centers.shape[-2])
        param = torch.tensor([-1.0, lam0])
    elif config["fmap"] == "diag":
        Ztr = poly_feat(Xtr, n=1)
        Zts = poly_feat(Xts, n=1)
        OPT = OPT_with_diag(OPT)
        if config["opt_low"] == "ce":
            param = -lam0 * torch.ones(Ztr.shape[-1] * (Ytr.shape[-1] - 1) + 1)
        elif config["opt_low"] == "ls":
            param = -lam0 * torch.ones(Ztr.shape[-1] * Ytr.shape[-1] + 1)
    elif config["fmap"] == "conv":
        Ztr, Zts = Xtr, Xts
        lam = -lam0 * torch.ones(1)
        conv_layer = torch.nn.Conv2d(1, 1, (config["conv_size"],) * 2)
        param = [z.reshape(-1) for z in conv_layer.parameters()][::-1]
        assert param[0].numel() == 1
        param = torch.cat([lam] + param)
        OPT = OPT_conv(OPT, stride=config["conv_size"])
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
    results["before"] = dict(
        loss_tr=loss_tr, acc_tr=acc_tr, loss_ts=loss_ts, acc_ts=acc_ts
    )

    # define functions ##############################################
    loss_fn_ = lambda W, param: loss_fn(OPT.pred(W, Zts, param), Yts, param)
    opt_fn_ = lambda param: OPT.solve(Ztr, Ytr, param)
    k_fn_ = lambda W, param: OPT.grad(W, Ztr, Ytr, param)
    Dzk_solve_ = lambda W, param, rhs, T=False: OPT.Dzk_solve(
        W, Ztr, Ytr, param, rhs, T=T
    )
    # Dpz = implicit_grads_1st(k_fn_, W, param, Dzk_solve_fn=Dzk_solve_)
    f_fn, g_fn, h_fn = generate_fns(
        loss_fn_,
        opt_fn_,
        k_fn_,
        Dzk_solve_fn=Dzk_solve_,
        normalize_grad=False,
    )
    h_fn_ = h_fn
    H_hist = []

    def h_fn(*args, **kwargs):
        ret = h_fn_(*args, **kwargs)
        H_hist.append(ret)
        return ret

    # fn = lambda: h_fn.fn(W, param)
    # import line_profiler

    # lp = line_profiler.LineProfiler(
    #    *[
    #        # g_fn.fn,
    #        h_fn.fn,
    #        # minimize_agd,
    #        minimize_lbfgs,
    #        # minimize_sqp,
    #        # implicit_grads_1st,
    #        implicit_grads_2nd,
    #    ]
    # )

    VERBOSE = True
    if config["solver"] == "sqp":
        param = minimize_sqp(
            f_fn,
            g_fn,
            h_fn,
            param,
            reg0=1e-5,
            verbose=VERBOSE,
            max_it=config["max_it"],
        )
    elif config["solver"] == "ipopt":
        param = minimize_ipopt(
            f_fn, g_fn, h_fn, param, verbose=VERBOSE, max_it=config["max_it"]
        )
    elif config["solver"] == "lbfgs":
        param = minimize_lbfgs(
            f_fn, g_fn, param, verbose=VERBOSE, max_it=config["max_it"], lr=1e-1
        )
    elif config["solver"] == "agd":
        param = minimize_agd(
            f_fn, g_fn, param, verbose=VERBOSE, max_it=10 ** 2, ai=1e-2, af=1e-2
        )
    else:
        raise ValueError
    # main_ = lp.wrap_function(main)
    # lp.print_stats()

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
        loss_tr=loss_tr, acc_tr=acc_tr, loss_ts=loss_ts, acc_ts=acc_ts
    )

    return param, results, H_hist


if __name__ == "__main__":
    fmaps = ["vanilla", "conv", "diag", "centers"]
    #fmaps = ["centers"]
    opt_lows = ["ls", "ce"]
    #opt_lows = ["ce"]

    params = [(fmap, opt_low) for fmap in fmaps for opt_low in opt_lows]

    for (fmap, opt_low) in params:
        config = dict(
            seed=0,
            train_n=10 ** 3,
            test_n=10 ** 3,
            opt_low=opt_low,
            conv_size=2,
            solver="sqp",
            max_it=10,
            fmap=fmap,
        )
        print("#" * 80)
        pprint(config)
        param, results, H_hist = main(config)
    pdb.set_trace()


if False and __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    n_tr, n_ts = 10 ** 3, 10 ** 2
    # read in the parameters ########################################
    Xp, Yp = None, (0.0, 1.0)
    fname = "data/cache.pkl.gz"
    try:
        with gzip.open(fname, "rb") as fp:
            centers, Xp, Yp = pickle.load(fp)
    except FileNotFoundError:
        (X_all, Xp), (Y_all, Yp) = get_mnist_data(mnist.train, Xp=Xp, Yp=Yp)
        centers = get_centers(X_all, Y_all)
        with gzip.open(fname, "wb") as fp:
            pickle.dump((centers, Xp, Yp), fp)

    (Xtr, Xtrp), (Ytr, Ytrp) = get_mnist_data(mnist.train, n_tr, Xp=Xp, Yp=Yp)
    (Xts, Xtsp), (Yts, Ytsp) = get_mnist_data(mnist.test, n_ts, Xp=Xp, Yp=Yp)
    Ztr = poly_feat(Xtr, n=1, centers=centers)
    Zts = poly_feat(Xts, n=1, centers=centers)

    OPT = OPT_with_centers(CE(), centers.shape[-2])
    param = torch.tensor([0.0, -2])

    W = OPT.solve(Ztr, Ytr, param)
    Yp_tr = OPT.pred(W, Ztr, param)
    Yp_ts = OPT.pred(W, Zts, param)
    print("Loss: %9.4e" % loss_fn(Yp_tr, Ytr, param))
    print("Accuracy: %3.2f%%" % acc_fn(Yp_tr, Ytr))
    print("Loss: %9.4e" % loss_fn(Yp_ts, Yts, param))
    print("Accuracy: %3.2f%%" % acc_fn(Yp_ts, Yts))

    # define functions ##############################################
    loss_fn_ = lambda W, param: loss_fn(OPT.pred(W, Zts, param), Yts, param)
    opt_fn_ = lambda param: OPT.solve(Ztr, Ytr, param)
    k_fn_ = lambda W, param: OPT.grad(W, Ztr, Ytr, param)
    Dzk_solve_ = lambda W, param, rhs, T=False: OPT.Dzk_solve(
        W, Ztr, Ytr, param, rhs, T=T
    )
    # Dpz = implicit_grads_1st(k_fn_, W, param, Dzk_solve_fn=Dzk_solve_)
    f_fn, g_fn, h_fn = generate_fns(
        loss_fn_,
        opt_fn_,
        k_fn_,
        Dzk_solve_fn=Dzk_solve_,
        normalize_grad=False,
    )

    # fn = lambda: h_fn.fn(W, param)
    # import line_profiler
    # lp = line_profiler.LineProfiler()
    # lp.add_function(h_fn.fn)
    # lp.add_function(implicit_grads_1st)
    # lp.add_function(implicit_grads_2nd)
    # wrapper = lp(fn)
    # wrapper()
    # lp.print_stats()
    # sys.exit()

    VERBOSE = True
    # param = minimize_sqp(f_fn, g_fn, h_fn, param, verbose=VERBOSE, max_it=10)
    # param = minimize_ipopt(f_fn, g_fn, h_fn, param, verbose=VERBOSE, max_it=20)
    param = minimize_lbfgs(
        f_fn, g_fn, param, verbose=VERBOSE, max_it=10, lr=1e-1
    )
    # param = minimize_agd(
    #   f_fn, g_fn, param, verbose=VERBOSE, max_it=10 ** 3, ai=1e-1, af=1e-1
    # )
    print(param)

    W = OPT.solve(Ztr, Ytr, param)
    Yp_tr = OPT.pred(W, Ztr, param)
    Yp_ts = OPT.pred(W, Zts, param)
    print("Loss: %9.4e" % loss_fn(Yp_tr, Ytr, param))
    print("Accuracy: %3.2f%%" % acc_fn(Yp_tr, Ytr))
    print("Loss: %9.4e" % loss_fn(Yp_ts, Yts, param))
    print("Accuracy: %3.2f%%" % acc_fn(Yp_ts, Yts))


if False and __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    n_tr, n_ts = 10 ** 4, 10 ** 3
    # read in the parameters ########################################
    Xp, Yp = (0.0, 1.0), (0.0, 1.0)
    fname = "data/cache.pkl.gz"
    try:
        with gzip.open(fname, "rb") as fp:
            centers, Xp, Yp = pickle.load(fp)
    except FileNotFoundError:
        (X_all, Xp), (Y_all, Yp) = get_mnist_data(mnist.train, Xp=Xp, Yp=Yp)
        centers = get_centers(X_all, Y_all)
        with gzip.open(fname, "wb") as fp:
            pickle.dump((centers, Xp, Yp), fp)

    (Xtr, Xtrp), (Ytr, Ytrp) = get_mnist_data(mnist.train, n_tr, Xp=Xp, Yp=Yp)
    (Xts, Xtsp), (Yts, Ytsp) = get_mnist_data(mnist.test, n_ts, Xp=Xp, Yp=Yp)
    Ztr = poly_feat(Xtr, n=1, centers=centers)
    Zts = poly_feat(Xts, n=1, centers=centers)

    W0 = 1e-5 * torch.randn((Ztr.shape[-1], Ytr.shape[-1] - 1))
    lam = 2
    OPT = OPT_with_centers(CE(), centers.shape[-2])
    param0 = torch.tensor([-1.0, lam])
    f_fn = lambda W: OPT.fval(W.reshape((Ztr.shape[-1], -1)), Ztr, Ytr, param0)
    g_fn = lambda W: OPT.grad(W.reshape((Ztr.shape[-1], -1)), Ztr, Ytr, param0)
    h_fn = lambda W: OPT.hess(W.reshape((Ztr.shape[-1], -1)), Ztr, Ytr, param0)

    # th = minimize_ipopt(
    #    f_fn, g_fn, h_fn, W0.reshape(-1), verbose=True, max_it=20
    # ).reshape(W0.shape)
    W0 = LS().solve(
        Ztr, torch.where(Ytr == 0.0, torch.tensor(-5.0), torch.tensor(0.0)), lam
    )[..., :-1]
    th = OPT_with_centers(CE(), centers.shape[-2]).solve(Ztr, Ytr, param0)
    # th = minimize_lbfgs(f_fn, g_fn, W0, lr=1e-1, verbose=True, max_it=10)

    acc_fn = lambda Yp, Y: torch.mean(
        1e2 * (torch.argmax(Yp, -1) == torch.argmax(Y, -1))
    )
    print("Accuracy: %3.2f%%" % acc_fn(CE.pred(th, Ztr), Ytr))
    print("Accuracy: %3.2f%%" % acc_fn(CE.pred(th, Zts), Yts))
    print()

    OPT = OPT_with_centers(LS(), centers.shape[-2])
    th = OPT.solve(Ztr, Ytr, param0)

    Yp_tr = OPT.pred(th, Ztr, param0)
    Yp_ts = OPT.pred(th, Zts, param0)
    print("Loss: %9.4e" % loss_fn(Yp_tr, Ytr, param0))
    print("Accuracy: %3.2f%%" % acc_fn(Yp_tr, Ytr))
    print("Loss: %9.4e" % loss_fn(Yp_ts, Yts, param0))
    print("Accuracy: %3.2f%%" % acc_fn(Yp_ts, Yts))
    sys.exit()

    # define functions ##############################################
    OPT = OPT_with_centers(LS(), centers.shape[-2])
    # loss_fn_ = lambda th, param: loss_fn(th, Zts, Yts, param)
    loss_fn_ = lambda th, param: loss_fn(OPT.pred(th, W, param), Ytr, param)
    opt_fn_ = lambda param: OPT.solve(Ztr, Ytr, param)
    k_fn_ = lambda th, param: OPT.grad(th, Ztr, Ytr, param)
    Dzk_solve_fn_ = lambda th, param, rhs, T=False: OPT.Dzk_solve_fn(
        th, Ztr, Ytr, param, rhs, T=T
    )
    Dpz = implicit_grads_1st(
        k_fn_, opt_fn_(param0), param0, Dzk_solve_fn=Dzk_solve_fn_
    )
    f_fn, g_fn, h_fn = generate_fns(
        loss_fn_,
        opt_fn_,
        k_fn_,
        Dzk_solve_fn=Dzk_solve_fn_,
        normalize_grad=False,
    )

    VERBOSE = True
    # param = minimize_sqp(f_fn, g_fn, h_fn, param0, verbose=VERBOSE, max_it=13)
    # param = minimize_ipopt(f_fn, g_fn, h_fn, param0, verbose=VERBOSE, max_it=20)
    # param = minimize_lbfgs(
    #   f_fn, g_fn, param0, verbose=VERBOSE, max_it=10, lr=1e-1
    # )
    param = minimize_agd(
        f_fn, g_fn, param0, verbose=VERBOSE, max_it=10 ** 3, ai=1e-3, af=1e-3
    )

    th = opt_fn(Ztr, Ytr, param)
    print("Loss: %9.4e" % loss_fn(th, Ztr, Ytr, param))
    print("Accuracy: %3.2f%%" % acc(Ytr, Z2Za(Ztr, param[0]) @ th, Ytrp))
    print("Loss: %9.4e" % loss_fn(th, Zts, Yts, param))
    print("Accuracy: %3.2f%%" % acc(Yts, Z2Za(Zts, param[0]) @ th, Ytsp))

    pdb.set_trace()
