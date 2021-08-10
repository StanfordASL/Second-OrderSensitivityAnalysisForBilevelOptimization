import pdb, math, os, sys, pickle, gzip, re
from pprint import pprint

import torch, matplotlib.pyplot as plt, numpy as np


def compute_xy(data_list, key):
    xs, ys = [], []
    for data in data_list:
        val0 = data["results"]["before"][key]
        value = [(k, v) for (k, v) in data["hist"][key].items()]
        x, y = zip(*value)
        x, y = np.array([1] + list(x)), np.array([val0] + list(y))
        xs.append(x)
        ys.append(y)
    z = np.sort(np.unique(np.concatenate(xs)))
    vals = np.stack(
        [np.interp(z, x, y, left=y[0], right=y[-1]) for (x, y) in zip(xs, ys)]
    )
    mu, err = np.mean(vals, -2), np.std(vals, -2) / np.sqrt(vals.shape[-2])
    return z, mu, err


def step(x, y, **kwargs):
    x_, y_ = np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]
    plt.plot(x_, y_, **kwargs)

def step_between(x, y1, y2, **kwargs):
    x_ = np.repeat(x, 2)[1:]
    y1_, y2_ = np.repeat(y1, 2)[:-1], np.repeat(y2, 2)[:-1]
    plt.fill_between(x_, y1_, y2_, **kwargs)

if __name__ == "__main__":
    matches = lambda x: re.match(r"all_results_[0-9]{3}.pkl.gz", x) is not None
    fnames = sum(
        [
            [os.path.join(root, fname) for fname in fnames if matches(fname)]
            for (root, dirs, fnames) in os.walk("data")
        ],
        [],
    )
    all_results = dict()
    for fname in fnames:
        with gzip.open(fname, "rb") as fp:
            data = pickle.load(fp)
            for (k, v) in data.items():
                all_results[k] = v
    results = dict()
    x, y = dict(), dict()
    for (k, v) in all_results.items():
        model, solver, lopt, trial = k
        key = model + "_" + lopt
        results.setdefault(key, dict())
        results[key].setdefault(solver, [])
        results[key][solver].append(v)
        x.setdefault(key, dict(sqp=[])
        y.setdefault(key, [])
        x[key].append(trial)
        y[key].append(results[key][solver][-1]["results"]["before"]["acc_ts"])
    pdb.set_trace()

    color_map = dict(sqp="C0", agd="C2", lbfgs="C1")

    for (k, data) in results.items():
        if k[-2:] != "ls":
            continue
        plt.figure()
        for solver in data.keys():
            key = "acc_ts"
            z, mu, err = compute_xy(data[solver], key)
            print(mu[0])

            c = color_map[solver]
            step(z, mu, label=solver, color=c)
            step_between(
                z, mu - 1.96 * err, mu + 1.96 * err, alpha=0.3, color=c
            )
            #plt.xscale("log")
            plt.grid(b=True, which="major")
            plt.grid(b=True, which="minor")
        plt.legend()
        plt.title(k)
        plt.tight_layout()
    plt.show()
