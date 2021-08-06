import os, pdb, sys, math, time, gzip, pickle

os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch, numpy as np
from tqdm import tqdm
from collections import OrderedDict as odict
import matplotlib.pyplot as plt

import header

from implicit import implicit_grads_1st
from implicit.diff import grad, JACOBIAN, HESSIAN_DIAG
from implicit.opt import minimize_agd
from implicit.nn_tools import nn_all_params, nn_forward

from mnist import train, test


# torch.set_num_threads(4)


def make_nn(device):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, (6, 6), 3),
        torch.nn.Tanh(),
        torch.nn.Conv2d(8, 8, (2, 2), 2),
        torch.nn.Tanh(),
        torch.nn.Conv2d(8, 2, (2, 2), 1),
        torch.nn.Tanh(),
        torch.nn.Flatten(),
        torch.nn.Linear(18, 18),
        torch.nn.Tanh(),
        torch.nn.Linear(18, 10),
        torch.nn.Softmax(dim=-1),
    ).to(device)


def main(mode):
    device = torch.device("cuda")
    nn = make_nn(device)
    Xtr, Xts = [
        torch.tensor(z, dtype=torch.float32, device=device).reshape(
            (-1, 1, 28, 28)
        )
        for z in [train["images"], test["images"]]
    ]
    Ytr, Yts = [
        torch.nn.functional.one_hot(
            torch.tensor(z, device=device, dtype=torch.long)
        )
        for z in [train["labels"], test["labels"]]
    ]

    loss_fn = torch.nn.MSELoss()
    # loss_fn = lambda a, b: torch.sum((a - b) ** 2)

    def eval_loss(n_train, params, X=None, Y=None):
        if X is None or Y is None:
            r = torch.randint(Xtr.shape[0], size=(n_train,))
            X, Y = Xtr[r, ...], Ytr[r, ...]
        loss = loss_fn(nn_forward(nn, X, params), Y.to(X.dtype))
        return loss

    def acc_fn(params, X, Y):
        return torch.mean(
            (
                torch.argmax(nn_forward(nn, X, params), -1)
                == torch.argmax(Y, -1)
            ).to(torch.float32)
        )

    fname = "data/results_fisher.pkl.gz"
    if mode == "train":
        params0 = nn_all_params(nn).clone()
        n_train = 10 ** 3

        params = torch.nn.Parameter(params0.clone())
        opt = torch.optim.Adam([params], lr=1e-4)
        results = odict()
        for it in tqdm(range(10 ** 4)):
            if it % 1000 == 0:
                results[it] = params.detach().clone().cpu().numpy()
            l = eval_loss(n_train, params, torch.tensor(0.0))
            opt.zero_grad()
            l.backward()
            old_params = params.detach().clone()
            g = params.grad.detach().norm()
            opt.step()
            imprv = (params.detach() - old_params).norm()
            tqdm.write(
                "%05d %9.4e %9.4e %9.4e"
                % (it, float(l), float(g), float(imprv))
            )
            if (it + 1) % (10 ** 2) == 0:
                print(
                    "-------------------------- %3.1f"
                    % (acc_fn(params, Xts, Yts) * 1e2)
                )
        with gzip.open(fname, "wb") as fp:
            pickle.dump(results, fp)
    elif mode == "eval":
        with gzip.open(fname, "rb") as fp:
            results = pickle.load(fp)
        params = torch.as_tensor(list(results.values())[-1], device=device)
        print("Accuracy %3.1f" % (1e2 * acc_fn(params, Xts, Yts)))

        r = torch.randint(Xtr.shape[0], size=(10 ** 5,))
        X, Y = Xtr[r, :], Ytr[r, :]

        z = params.clone().requires_grad_()

        def k_fn(z):
            ret = JACOBIAN(
                lambda z: eval_loss(-1, z, X=Xtr, Y=Ytr),
                z,
                create_graph=True,
            )
            return ret

        t = time.perf_counter()
        H = JACOBIAN(k_fn, z)
        t = time.perf_counter() - t
        print("H: %9.4e" % t)
        with gzip.open("data/hessian_fisher.pkl.gz", "wb") as fp:
            pickle.dump(H.detach().cpu().numpy(), fp)
    elif mode == "read":
        with gzip.open("data/hessian_fisher.pkl.gz", "rb") as fp:
            H = pickle.load(fp)

        plt.figure()
        plt.title("Value")
        plt.contourf(H, 50)
        plt.colorbar()
        plt.tight_layout()

        plt.figure()
        plt.title("Abs")
        plt.contourf(np.abs(H), 50)
        plt.colorbar()
        plt.tight_layout()

        plt.draw_all()
        plt.pause(1e-1)

        H = torch.as_tensor(H, device="cpu")

        pdb.set_trace()
    elif mode == "fisher":
        with gzip.open(fname, "rb") as fp:
            results = pickle.load(fp)
        params = torch.as_tensor(list(results.values())[-1], device=device)
        print("Accuracy %3.1f" % (1e2 * acc_fn(params, Xts, Yts)))

        def fisher_hess(n_train, params, X=None, Y=None):
            if X is None or Y is None:
                r = torch.randint(Xtr.shape[0], size=(n_train,))
                X, Y = Xtr[r, ...], Ytr[r, ...]
            Z = nn_forward(nn, X, params).detach()
            Y = Y.to(Z.dtype)
            dlogp = JACOBIAN(lambda Z: loss_fn(Z, Y), Z)
            dlogp2 = JACOBIAN(lambda Y: loss_fn(Z, Y), Y)

            t = time.perf_counter()
            H = JACOBIAN(
                lambda Z: JACOBIAN(
                    lambda Z: loss_fn(Z, Y), Z, create_graph=True
                ).sum(-2),
                Z,
            ).sum(-2)
            t = time.perf_counter() - t
            print("%9.4e" % t)

            t = time.perf_counter()
            J = JACOBIAN(
                lambda params: nn_forward(nn, X, params).sum(-2), params
            )
            t = time.perf_counter() - t

            t = time.perf_counter()
            Js = torch.stack(
               [
                   JACOBIAN(
                       lambda params: nn_forward(
                           nn, X[i : i + 1, ...], params
                       )[0, :],
                       params,
                   )
                   for i in range(X.shape[0])
               ]
            )
            Js = torch.diag(torch.diag(H).sqrt())[None, ...] @ Js
            H = torch.einsum("ijl,ilk->jk", Js.transpose(-1, -2), Js) / n_train
            with gzip.open("data/hessian_fisher.pkl.gz", "rb") as fp:
                H2 = torch.as_tensor(pickle.load(fp), device=device)
            #Js = JACOBIAN(
            #    lambda params: nn_forward(nn, X, params),
            #    params,
            #)
            t = time.perf_counter() - t
            print("%9.4e" % t)

            plt.figure()
            plt.title("H fisher")
            plt.contourf(H.cpu(), 50)
            plt.tight_layout()

            plt.figure()
            plt.title("H actual")
            plt.contourf(H2.cpu(), 50)
            plt.tight_layout()

            plt.show()

            pdb.set_trace()

        fisher_hess(10 ** 4, params)

    else:
        print("Nothing to be done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) >= 2 else "eval")
