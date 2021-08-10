import pdb, math, os, sys, pickle, gzip, re
from pprint import pprint

import torch, matplotlib.pyplot as plt, numpy as np


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
    kname = "acc_ts"
    results = dict()
    for (k, v) in all_results.items():
        model, solver, lopt, trial = k
        key = model + "_%d" % trial
        results.setdefault(key, dict())
        val = v["results"]["before"][kname]
        val = round(1e6 * val) / 1e6
        results[key][solver] = val

    for (k, v) in results.items():
        print(k)
        pprint(v)
        print()
