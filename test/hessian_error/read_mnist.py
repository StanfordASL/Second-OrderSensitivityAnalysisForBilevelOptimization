import sys, os, pdb, time, math, re, gzip, pickle
from collections import OrderedDict as odict

import torch, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

import mnist_vanilla
import header
from implicit.nn_tools import nn_forward
from implicit.diff import JACOBIAN


def cosine_similarity(a, b):
    return torch.sum(a.reshape(-1) * b.reshape(-1)) / a.norm() / b.norm()

def pct_diff(a, b):
    return torch.norm(a.reshape(-1) - b.reshape(-1)) / b.norm()


def to_np(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return type(x)(to_np(z) for z in x)
    elif isinstance(x, dict):
        return {k: v for (k, v) in zip(to_np(x.keys()), to_np(x.values()))}
    elif isinstance(x, odict):
        return odict(zip(to_np(x.keys()), to_np(x.values())))
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif hasattr(x, "__iter__"):
        return to_np(list(x))
    else:
        return x


def to_torch(x, device=None, dtype=None):
    if isinstance(x, list) or isinstance(x, tuple):
        return type(x)(to_torch(z, device=device, dtype=dtype) for z in x)
    elif isinstance(x, dict):
        return {
            k: v
            for (k, v) in zip(
                to_torch(x.keys(), device=device, dtype=dtype),
                to_torch(x.values(), device=device, dtype=dtype),
            )
        }
    elif isinstance(x, odict):
        return odict(
            zip(
                to_torch(x.keys(), device=device, dtype=dtype),
                to_torch(x.values(), device=device, dtype=dtype),
            )
        )
    elif any(isinstance(x, z) for z in [float, int, np.ndarray]):
        return torch.as_tensor(x, device=device, dtype=dtype)
    elif hasattr(x, "__iter__"):
        return to_torch(list(x), device=device, dtype=dtype)
    else:
        return x


if __name__ == "__main__":
    device = "cuda"

    # read in the neural networks ###############################
    fname = "data/grads.pkl.gz"
    try:
        with gzip.open(fname, "rb") as fp:
            grads = pickle.load(fp)
    except FileNotFoundError:
        nn = mnist_vanilla.make_nn(device)
        lam = torch.as_tensor(mnist_vanilla.LAM, device=device)

        fwd_fn = lambda X, params: nn_forward(nn, X, params)
        loss_fn_ = torch.nn.MSELoss()
        with gzip.open("data/results.pkl.gz", "rb") as fp:
            results = pickle.load(fp)

        X = to_torch(results["X"], device=device)
        Y = to_torch(results["Y"], device=device)

        loss_fn = (
            lambda z, lam: loss_fn_(fwd_fn(X, z), Y)
            + (10.0 ** lam.reshape(())) * (z ** 2).sum()
        )

        def k_fn(z, lam):
            return JACOBIAN(lambda z: loss_fn(z, lam), z)

        k_fn_ = lambda z: k_fn(torch.as_tensor(z, device=device), lam)

        x = []
        for (it, z) in tqdm(list(results["iters"].items())):
            x.append((it, k_fn_(z).cpu().numpy()))
        grads = odict(x)

        with gzip.open(fname, "wb") as fp:
            pickle.dump(grads, fp)

    grads = odict((k, to_torch(v, device=device)) for (k, v) in grads.items())
    # pdb.set_trace()

    # read in the results #######################################
    fname_list = sum(
        [
            [
                os.path.join(root, file)
                for file in files
                if re.match(r"Dpz_[0-9]{3}.pkl.gz", file) is not None
            ]
            for (root, _, files) in os.walk("data")
        ],
        [],
    )
    results = dict()
    for fname in tqdm(fname_list):
        with gzip.open(fname, "rb") as fp:
            results_ = pickle.load(fp)
        for (k, v) in results_.items():
            results.setdefault(k, dict())
            for (k2, v2) in v.items():
                results[k][k2] = torch.as_tensor(v2, device=device)
    results = odict((k, results[k]) for k in sorted(list(results.keys())))

    grad_fn = lambda Dpz, g: float(
        torch.sum(Dpz.reshape(-1) * g.reshape(-1)).cpu()
    )
    grads = odict((k, v) for (k, v) in grads.items() if k in results.keys())

    it_ref = np.max(list(results.keys()))

    # quantify the error in gradients ########################################
    key = "Dpz"
    ref = results[it_ref][key]
    vals = odict((k, z[key]) for (k, z) in results.items())
    loss_fn, scale = cosine_similarity, "log"
    errs = odict((k, float(loss_fn(v, ref).cpu())) for (k, v) in vals.items())

    plt.figure()
    xs = [it_ref - z + 1 for z in to_np(errs.keys())]
    if scale == "log":
        plt.loglog(xs, to_np(errs.values()))
    else:
        plt.semilogy(xs, to_np(errs.values()))
    plt.scatter(xs, to_np(errs.values()))
    plt.title("MNIST NN with $\\approx$ 4k parameters")
    plt.ylabel("Hessian Error - $||H - H*||_F / ||H*||_F$")
    plt.xlabel("it from last")
    plt.grid(b=True, which="major", axis="both")
    plt.grid(b=True, which="minor", axis="both")
    plt.tight_layout()
    #plt.savefig("figs/it_vs_herr.png", dpi=200)

    plt.figure()
    plt.plot(to_np([z.norm() for z in grads.values()]), to_np(errs.values()),
            color="none")
    plt.scatter(to_np([z.norm() for z in grads.values()]), to_np(errs.values()))
    plt.xscale("log")
    plt.yscale("log")
    plt.title("MNIST NN with $\\approx$ 4k parameters")
    plt.ylabel("Hessian Error - $||H - H*||_F / ||H*||_F$")
    plt.xlabel("$||g||_2$ - optimality condition")
    plt.grid(b=True, which="major", axis="both")
    plt.grid(b=True, which="minor", axis="both")
    plt.tight_layout()
    plt.savefig("figs/g_vs_herr.png", dpi=200)

    plt.draw_all()
    plt.pause(1e-2)

    pdb.set_trace()
