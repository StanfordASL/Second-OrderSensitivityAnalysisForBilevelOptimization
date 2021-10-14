import pdb, math, os, sys, pickle, gzip, re
from pprint import pprint

import torch, matplotlib.pyplot as plt, numpy as np

KEY = "loss_ts" if len(sys.argv) <= 1 else sys.argv[1].lower()


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
    assert np.all(np.sort(z) == z)
    return z, mu, err


def step(x, y, **kwargs):
    x_, y_ = np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]
    plt.plot(x_, y_, **kwargs)


def step_between(x, y1, y2, **kwargs):
    x_ = np.repeat(x, 2)[1:]
    y1_, y2_ = np.repeat(y1, 2)[:-1], np.repeat(y2, 2)[:-1]
    plt.fill_between(x_, y1_, y2_, **kwargs)


if __name__ == "__main__":
    plt.rc("font", size=16)

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
    for (k, v) in all_results.items():
        model, solver, lopt, trial = k
        key = model + "_" + lopt
        results.setdefault(key, dict())
        results[key].setdefault(solver, [])
        results[key][solver].append(v)

    #color_map = dict(sqp="C0", agd="C2", lbfgs="C1")
    color_map = dict(sqp="black", agd="C0", lbfgs="C4")
    label_map = dict(sqp="SQP (Ours)", lbfgs="L-BFGS", agd="Adam")
    cat = np.concatenate

    ymin, ymax = math.inf, -math.inf
    for (k, data) in results.items():
        for solver in data.keys():
            z, mu, err = compute_xy(data[solver], KEY)
            if KEY == "acc_ts":
                mu, err = mu / 100, err / 100
            ymin = min(ymin, np.min(mu - 1.96 * err))
            ymax = max(ymax, np.max(mu + 1.96 * err))
    yrng = ymax - ymin
    if KEY == "loss_ts":
        ymax = 2.1

    for (k, data) in results.items():
        plt.figure(figsize=(5, 4))
        max_it = -math.inf
        for solver in data.keys():
            z, mu, err = compute_xy(data[solver], KEY)
            max_it = max(max_it, z[-1])
        #for solver in data.keys():
        for solver in ["sqp", "agd", "lbfgs"]:
            if solver not in data.keys():
                continue
            z, mu, err = compute_xy(data[solver], KEY)
            #if KEY == "acc_ts":
            #    mu, err = mu / 100, err / 100
            if z[-1] != max_it:
                z = cat([z, [max_it]])
                mu, err = cat([mu, [mu[-1]]]), cat([err, [err[-1]]])

            c = color_map[solver]
            step(z, mu, label=label_map[solver], color=c)
            step_between(
                z, mu - 1.96 * err, mu + 1.96 * err, alpha=0.3, color=c
            )
            # plt.xscale("log")
            plt.grid(b=True, which="major")
            plt.grid(b=True, which="minor")
        plt.xlim([0, 200])

        #plt.ylim([ymin - 0.1 * yrng, ymax + 0.1 * yrng])
        if KEY == "loss_ts":
            plt.ylim([1.7, 2.15])
        elif KEY == "acc_ts":
            plt.ylim([60.0, 90.0])
        plt.xlabel("$z^\\star$ evaluations")
        if key == "loss_ts":
           plt.ylabel("$f_U\\left(z^\\star, p\\right)$")
        elif key == "acc_ts":
           plt.ylabel("Test Accuracy")
        if key == "acc_ts":
            plt.legend(loc="lower right")
        elif key == "loss_ts":
            plt.legend(loc="upper right")
        #plt.title(k)

        plt.margins(0)

        if k.split("_")[0] == "vanilla":
            plt.legend()
            if KEY == "loss_ts":
                plt.ylabel("$\\ell_\\operatorname{test}$")
            elif KEY == "acc_ts":
                plt.ylabel("Test Accuracy")
        plt.tight_layout()

        plt.savefig(
            "figs/%s_%s.png" % (k, KEY),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()
