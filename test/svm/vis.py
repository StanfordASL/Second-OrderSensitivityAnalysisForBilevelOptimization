import sys, os, pickle, gzip, time

import numpy as np, matplotlib.pyplot as plt

if __name__ == "__main__":
    with gzip.open("data/losses.pkl.gz", "rb") as fp:
        x, y = pickle.load(fp)
    #x, y = x[1:-1], np.diff(y, n=2)
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.xlim([1e-8, 1e-6])
    plt.ylabel("$\\ell$")
    plt.xlabel("$\\lambda$")
    #plt.yscale("log")
    #plt.xscale("log")
    plt.tight_layout()

    plt.show()
