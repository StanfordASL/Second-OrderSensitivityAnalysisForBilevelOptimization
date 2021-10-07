import sys, os, pdb, time, math, re, gzip, pickle
from collections import OrderedDict as odict
from pprint import pprint

import torch, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

import header

from implicit.interface import init

jaxm = init(dtype=np.float32, device="gpu")

from implicit.nn_tools import nn_forward_gen
from implicit.diff import JACOBIAN

import mnist_vanilla
################################################################################


def cosine_similarity(a, b):
    return jaxm.sum(a.reshape(-1) * b.reshape(-1)) / jaxm.norm(a) / jaxm.norm(b)


def pct_diff(a, b):
    return jaxm.norm(a.reshape(-1) - b.reshape(-1)) / jaxm.norm(b)


if __name__ == "__main__":
    device = "cuda"

    # read in the neural networks ###############################
    with gzip.open("data/results.pkl.gz", "rb") as fp:
        results = pickle.load(fp)
    config = results["config"]
    lam = jaxm.array([config["lam"]])
    nn = mnist_vanilla.make_nn()
    X, Y = jaxm.array(results["X"]), jaxm.array(results["Y"])

    fwd_fn = nn_forward_gen(nn)

    @jaxm.jit
    def loss_fn(z, lam, X, Y):
        l = jaxm.mean(jaxm.sum((fwd_fn(X, z) - Y) ** 2, -1))
        l = l + jaxm.sum((10.0 ** lam) * z ** 2)
        return l

    k_fn = jaxm.jit(jaxm.grad(lambda z, lam: loss_fn(z, lam, X, Y)))

    gs = odict()
    for (it, param) in tqdm(results["param_hist"].items()):
        gs[it] = k_fn(param, lam)
    gs = odict([(k, v) for (k, v) in gs.items()])

    # read in the results #######################################
    dirname = "data"
    fname_list = [
        os.path.join(dirname, fname)
        for fname in os.listdir(dirname)
        if re.match(r"Dpz_[0-9]{3}_2.pkl.gz", fname)
    ]
    Dzk, Dpz, Dzl, Dll = odict(), odict(), odict(), odict()
    for fname in tqdm(fname_list):
        with gzip.open(fname, "rb") as fp:
            data = pickle.load(fp)
        it = list(data.keys())[0]
        Dzk[it] = data[it]["Dzk"]
        Dpz[it] = data[it]["Dpz"]
        Dzl[it] = data[it]["Dlz"] if "Dlz" in data[it] else data[it]["Dzl"]
        Dll[it] = jaxm.sum(Dzl[it] * Dpz[it])

    # collate the results #######################################
    its = jaxm.sort(jaxm.intersect1d(list(gs.keys()), list(Dll.keys())))
    its = [int(it) for it in its]
    gs, Dzk, Dpz, Dzl, Dll = [
        odict([(it, z[it]) for it in its]) for z in [gs, Dzk, Dpz, Dzl, Dll]
    ]

    it_ref = its[
        int(jaxm.argmin(jaxm.array([jaxm.norm(gs[it]) for it in its])))
    ]
    Dll_ref = Dll[it_ref]
    Dzk_ref = Dzk[it_ref]
    Dpz_ref = Dpz[it_ref]

    pdb.set_trace()

    # plotting ##################################################
    plt.figure()
    plt.title("dldlam Error")
    plt.loglog(
        [jaxm.norm(gs[it]) for it in its],
        [jaxm.abs(Dll[it] - Dll_ref) for it in its],
        "o",
    )

    plt.figure()
    plt.title("dldlam cosine")
    plt.semilogx(
        [jaxm.norm(gs[it]) for it in its],
        [cosine_similarity(Dll[it], Dll_ref) for it in its],
        "o",
    )

    plt.figure()
    plt.title("Dzk Error")
    plt.loglog(
        [jaxm.norm(gs[it]) for it in its],
        [pct_diff(Dzk[it], Dzk_ref) for it in its],
        "o",
    )

    plt.figure()
    plt.title("Dpz Error")
    plt.loglog(
        [jaxm.norm(gs[it]) for it in its],
        [pct_diff(Dpz[it], Dpz_ref) for it in its],
        "o",
    )

    plt.draw_all()
    plt.pause(1e-1)

    # quantify the error in gradients ########################################
    # key = "Dpz"
    # ref = results[it_ref][key]
    # vals = odict((k, z[key]) for (k, z) in results.items())
    # loss_fn, scale = cosine_similarity, "log"
    # errs = odict((k, float(loss_fn(v, ref).cpu())) for (k, v) in vals.items())

    # plt.figure()
    # xs = [it_ref - z + 1 for z in to_np(errs.keys())]
    # if scale == "log":
    #    plt.loglog(xs, to_np(errs.values()))
    # else:
    #    plt.semilogy(xs, to_np(errs.values()))
    # plt.scatter(xs, to_np(errs.values()))
    # plt.title("MNIST NN with $\\approx$ 4k parameters")
    # plt.ylabel("Hessian Error - $||H - H*||_F / ||H*||_F$")
    # plt.xlabel("it from last")
    # plt.grid(b=True, which="major", axis="both")
    # plt.grid(b=True, which="minor", axis="both")
    # plt.tight_layout()
    ## plt.savefig("figs/it_vs_herr.png", dpi=200)

    # plt.figure()
    # plt.plot(
    #    to_np([z.norm() for z in grads.values()]),
    #    to_np(errs.values()),
    #    color="none",
    # )
    # plt.scatter(to_np([z.norm() for z in grads.values()]), to_np(errs.values()))
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("MNIST NN with $\\approx$ 4k parameters")
    # plt.ylabel("Hessian Error - $||H - H*||_F / ||H*||_F$")
    # plt.xlabel("$||g||_2$ - optimality condition")
    # plt.grid(b=True, which="major", axis="both")
    # plt.grid(b=True, which="minor", axis="both")
    # plt.tight_layout()
    # plt.savefig("figs/g_vs_herr.png", dpi=200)

    # plt.draw_all()
    # plt.pause(1e-2)

    pdb.set_trace()
