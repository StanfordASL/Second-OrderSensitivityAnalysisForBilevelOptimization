import time, pdb, math
from collections import OrderedDict as odict

import torch, numpy as np, matplotlib.pyplot as plt

import include_implicit

from implicit.inverse import solve_cg_torch


def main(mode):
    torch.set_default_dtype(torch.double)

    A = torch.randn((1000, 1000))
    A = A @ A.T / A.shape[-1] + 1e-5 * torch.eye(A.shape[-1])
    A.diagonal()[:] += torch.cat(
        [1e-1 * torch.ones(A.shape[-1] // 2), torch.zeros(A.shape[-1] // 2)]
    )
    b = torch.randn((A.shape[-1], 1))

    xs = torch.linalg.lstsq(A, b)[0]

    err_fn = lambda x: torch.norm(A @ x - b)
    A_fn = lambda x: A @ x

    if True:
        lr = 1e-2
        x = torch.nn.Parameter(1e-5 * torch.randn((A.shape[0], 1)))
        opt = torch.optim.Adam([x], lr=lr)
        for it in range(10 ** 4):
            opt.zero_grad()
            l = torch.norm((A_fn(x) - b))
            l.backward()
            opt.step()
            print("%05d | %9.4e" % (it, float(l)))
        s = opt.state_dict()["state"][0]["exp_avg_sq"]
        s = s.sqrt() + 1e-7

        # worse than no-cond
        M1_fn = lambda x: x * (torch.diag(A).reshape(x.shape) + 1e-9)
        M3_fn = lambda x: x * (torch.diag(A).reshape(x.shape) ** 2 + 1e-9)
        M5_fn = lambda x: x * (
            torch.sqrt(torch.diag(A).reshape(x.shape)) + 1e-9
        )

        # not that good
        M6_fn = lambda x: x / (
            torch.sqrt(torch.diag(A).reshape(x.shape)) + 1e-9
        )

        # good
        M2_fn = lambda x: x / (torch.diag(A).reshape(x.shape) + 1e-9)
        M4_fn = lambda x: x / (torch.diag(A).reshape(x.shape) ** 2 + 1e-9)

        # adam based
        MA1_fn = lambda x: x / (s.reshape(x.shape))
        MA2_fn = lambda x: x / (s.reshape(x.shape) / lr)
        MA3_fn = lambda x: x / (s.reshape(x.shape) / math.sqrt(lr))

        labels = [
            "base",
            # "dmul",
            "ddiv",
            # "dmul2",
            # "ddiv2",
            # "dmulsqrt",
            # "ddivsqrt",
            "sdiv",
            "slrdiv",
            "slrsqrtdiv",
        ]
        fns = [
            lambda x: x,
            # M1_fn,
            M2_fn,
            # M3_fn,
            # M4_fn,
            # M5_fn,
            # M6_fn,
            MA1_fn,
            MA2_fn,
            MA3_fn,
        ]
        assert len(labels) == len(fns)

        # evaluate ####################################################
        results_err = {label: odict() for label in labels}
        results_dist = {label: odict() for label in labels}
        for it in range(1, 100, 5):
            for (label, fn) in zip(labels, fns):
                x = solve_cg_torch(
                    A_fn, b, M_fn=fn, x0=b.clone()[None, ...], max_it=it
                )
                results_err[label][it] = float(err_fn(x))
                results_dist[label][it] = float(
                    torch.norm(x - xs) / torch.norm(xs)
                )

        # plot ########################################################
        plt.figure()
        plt.title("Error")
        for k in results_err.keys():
            plt.semilogy(
                list(results_err[k].keys()),
                list(results_err[k].values()),
                label=k,
            )
        plt.legend()

        plt.figure()
        plt.title("Distance")
        for k in results_dist.keys():
            plt.semilogy(
                list(results_dist[k].keys()),
                list(results_dist[k].values()),
                label=k,
            )
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main("adam")
