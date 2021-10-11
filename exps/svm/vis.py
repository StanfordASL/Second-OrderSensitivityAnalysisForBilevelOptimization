import sys, os, pickle, gzip, time, pdb

import numpy as np, matplotlib.pyplot as plt

ACTIONS = sys.argv[1:]


def step(x, y, **kwargs):
    x_, y_ = np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]
    plt.plot(x_, y_, **kwargs)


def step_between(x, y1, y2, **kwargs):
    x_ = np.repeat(x, 2)[1:]
    y1_, y2_ = np.repeat(y1, 2)[:-1], np.repeat(y2, 2)[:-1]
    plt.fill_between(x_, y1_, y2_, **kwargs)


if "loss" in ACTIONS and __name__ == "__main__":
    with gzip.open("data/losses.pkl.gz", "rb") as fp:
        x, y, _, _ = pickle.load(fp)
    # x, y = x[1:-1], np.diff(y, n=2)

    # plt.scatter(x, y)
    # plt.xlim([1e-8, 1e-6])

    plt.ylabel("$\\ell_\\operatorname{test}$")
    plt.xlabel("$\\operatorname{log}_{10}\\left(\\lambda\\right)$")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig("figs/svm_loss.png", dpi=200)
    plt.show()

if "opt" in ACTIONS and __name__ == "__main__":
    #with gzip.open("data/opt_hist.pkl.gz", "rb") as fp:
    with gzip.open("data/logbarrier_opt_hist.pkl.gz", "rb") as fp:
        hist = pickle.load(fp)

    plt.rc("font", size=16)

    color_map = dict(sqp="C0", agd="C2", lbfgs="C1")
    label_map = dict(sqp="SQP (Ours)", agd="Adam", lbfgs="L-BFGS")
    plt.figure()
    min_loss = min([np.min(data["loss"]) for data in hist.values()])
    for (k, data) in hist.items():
        x = data["fns"]
        t = np.cumsum(data["t"])
        # step(x, data["acc"], color=color_map[k], label=k)
        # step(t, data["acc"], color=color_map[k], label=k)
        step(
            t,
            #data["loss"] - min_loss,
            data["loss"],
            color=color_map[k],
            label=label_map[k],
            lw=2,
        )
        plt.yscale("log")
    plt.grid(b=True, which="major")
    plt.xlabel("Time (s)")
    plt.ylabel("$\\left|f(x) - f(x^\\star)\\right|$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/opt_vs_time.png", dpi=200)
    plt.draw_all()
    plt.pause(1e-1)
    pdb.set_trace()
