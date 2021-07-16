import sys, os, pdb, time, math, re, gzip, pickle
from collections import OrderedDict as odict

import torch, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

import sinusoid_nn
import include_implicit
from implicit.nn_tools import nn_forward, nn_all_params
from implicit.diff import JACOBIAN


def cosine_similarity(a, b):
    a, b = to_torch(a), to_torch(b)
    return torch.sum(a.reshape(-1) * b.reshape(-1)) / a.norm() / b.norm()


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
    fname = "data/sin_grads.pkl.gz"
    try:
        with gzip.open(fname, "rb") as fp:
            data = pickle.load(fp)
        grads, grad_norms = data["grads"], data["norms"]
    except FileNotFoundError:
        nn = sinusoid_nn.make_nn(device)
        lam = sinusoid_nn.LAM * torch.ones(
            nn_all_params(nn).numel(), device=device
        )

        fwd_fn = lambda X, params: nn_forward(nn, X, params)
        loss_fn = sinusoid_nn.loss_fn_gen(fwd_fn)
        with gzip.open("data/sin_results.pkl.gz", "rb") as fp:
            results = pickle.load(fp)

        X = torch.as_tensor(results["X"], device=device)
        Y = torch.as_tensor(results["Y"], device=device)

        def k_fn(z, lam):
            return JACOBIAN(lambda z: loss_fn(z, lam, X, Y), z)

        k_fn_ = lambda z: k_fn(torch.as_tensor(z, device=device), lam)

        x = []
        for (it, z) in tqdm(list(results["iters"].items())):
            x.append((it, k_fn_(z).cpu().numpy()))
        grads = odict(x)
        grad_norms = odict((k, np.linalg.norm(v)) for (k, v) in grads.items())

        with gzip.open(fname, "wb") as fp:
            pickle.dump(dict(norms=grad_norms, grads=grads), fp)

    grads = odict(
        (k, torch.as_tensor(v, device=device)) for (k, v) in grads.items()
    )
    grad_norms = odict(
        (k, torch.as_tensor(v, device=device)) for (k, v) in grad_norms.items()
    )

    # read in the results #######################################
    fname_list = sum(
        [
            [
                os.path.join(root, file)
                for file in files
                if re.match(r"sin_Dpz_[0-9]{3}.pkl.gz", file) is not None
            ]
            for (root, _, files) in os.walk("data")
        ],
        [],
    )
    results = dict()
    for fname in tqdm(fname_list):
        with gzip.open(fname, "rb") as fp:
            results_ = pickle.load(fp)
        assert len(list(results_.keys())) == 1
        for (k, v) in results_.items():
            print("%d, %s" % (k, fname))
            results.setdefault(k, dict())
            for (k2, v2) in v.items():
                results[k][k2] = torch.as_tensor(v2, device=device)
    results = odict((k, results[k]) for k in sorted(list(results.keys())))

    grad_fn = lambda Dpz, g: (g.reshape((1, -1)) @ Dpz).reshape(-1).cpu()
    grads = odict((k, v) for (k, v) in grads.items() if k in results.keys())
    pdb.set_trace()
    grad_norms = odict(
        (k, v) for (k, v) in grad_norms.items() if k in results.keys()
    )

    it_ref = list(results.keys())[-1]
    key = "Dpz"
    ref = results[it_ref][key]

    ref = grad_fn(ref, grads[it_ref])
    vals = odict((k, z[key]) for (k, z) in results.items())
    vs = odict(
        (k1, grad_fn(Dpz, g))
        for ((k1, Dpz), (k2, g)) in zip(vals.items(), grads.items())
    )
    pdb.set_trace()
    errs = odict((k, (v - ref).norm() / ref.norm()) for (k, v) in vs.items())
    #errs = odict((k, abs(v - ref) / abs(ref)) for (k, v) in vs.items())
    #errs = odict((k, float(cosine_similarity(v, ref))) for (k, v) in vs.items())

    plt.figure()
    plt.semilogy(to_np(errs.keys()), to_np(errs.values()))
    plt.scatter(to_np(errs.keys()), to_np(errs.values()))
    plt.ylabel("err")
    plt.xlabel("it")
    plt.tight_layout()

    plt.figure()
    plt.semilogy(to_np(grad_norms.values()), to_np(errs.values()))
    plt.scatter(to_np(grad_norms.values()), to_np(errs.values()))
    plt.ylabel("err")
    plt.xlabel("gs")
    plt.tight_layout()

    plt.draw_all()
    plt.pause(1e-2)

    pdb.set_trace()
