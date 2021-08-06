import os, pdb, sys, math, time, gzip, pickle

os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch, numpy as np
from tqdm import tqdm
from collections import OrderedDict as odict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import header

from implicit import implicit_grads_1st
from implicit.diff import grad, JACOBIAN, HESSIAN_DIAG, torch_grad
from implicit.opt import minimize_agd
from implicit.nn_tools import nn_all_params, nn_structure, nn_forward
from implicit.nn_tools import nn_forward2

from mnist import train, test

LAM = -4.0


def make_nn(device):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, (6, 6), 3),
        torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, (2, 2), 2),
        torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, (2, 2), 1),
        torch.nn.Tanh(),
        torch.nn.Conv2d(16, 16, (2, 2), 2),
        torch.nn.Tanh(),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 1, 16 * 1),
        torch.nn.Tanh(),
        torch.nn.Linear(16 * 1, 10),
        torch.nn.Softmax(dim=-1),
    ).to(device)


def loss_fn_gen(fwd_fn):
    l = torch.nn.MSELoss()

    def loss_fn(z, lam, X, Y):
        return l(fwd_fn(X, z), Y) + (10.0 ** lam.reshape(())) * (z ** 2).sum()

    return loss_fn


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
        ).to(Xtr.dtype)
        for z in [train["labels"], test["labels"]]
    ]

    params = nn_all_params(nn)
    # nn_struct = nn_structure(nn)
    # Yp = nn_forward2(nn_struct, Xtr, params)
    fwd_fn = lambda X, params: nn_forward(nn, X, params)

    loss_fn = loss_fn_gen(fwd_fn)

    def acc_fn(params, X, Y):
        return torch.mean(
            (torch.argmax(fwd_fn(X, params), -1) == torch.argmax(Y, -1)).to(
                torch.float32
            )
        )

    fname = "data/results.pkl.gz"
    first_phase_it = 2 * 10 ** 5
    if mode == "train":
        params0 = nn_all_params(nn).clone()
        lam = LAM
        params = torch.nn.Parameter(params0.clone())
        opt = torch.optim.Adam([params], lr=1e-4)
        results = dict(iters=odict(), X=None, Y=None)
        first_phase = True

        def step_fn(X, Y):
            l = loss_fn(params, torch.tensor(lam, device=device), X, Y)
            opt.zero_grad()
            l.backward()
            return l

        writer = SummaryWriter()

        n_train = 10 ** 3
        for it in tqdm(range(first_phase_it + 10 ** 4)):
            if it == first_phase_it:
                first_phase, n_train = False, 10 ** 5
                r = torch.randint(Xtr.shape[0], size=(n_train,))
                X, Y = Xtr[r, ...], Ytr[r, ...]
                results["X"] = X.cpu().clone().numpy()
                results["Y"] = Y.cpu().clone().numpy()
                opt.param_groups[0]["lr"] = 1e-5
            if not first_phase or it % (10 ** 2) == 0:
                acc = 1e2 * acc_fn(params, Xts, Yts)
            if (first_phase and it % 1000 == 0) or (
                not first_phase and it % 50
            ):
                results["iters"][it] = params.detach().clone().cpu().numpy()
            if first_phase:
                r = torch.randint(Xtr.shape[0], size=(n_train,))
                X, Y = Xtr[r, ...], Ytr[r, ...]
            old_params = params.detach().clone()
            l = opt.step(lambda: step_fn(X, Y))
            g = params.grad.detach().norm()
            imprv = (params.detach() - old_params).norm()
            tqdm.write(
                "%05d %9.4e %9.4e %9.4e %3.1f%%"
                % (it, float(l), float(g), float(imprv), float(acc))
            )
            writer.add_scalar("loss", float(l), it)
            writer.add_scalar("grad", float(g), it)
            writer.add_scalar("imprv", float(imprv), it)
            writer.add_scalar("acc", float(acc), it)
        with gzip.open(fname, "wb") as fp:
            pickle.dump(results, fp)
    elif mode == "eval":
        with gzip.open(fname, "rb") as fp:
            results = pickle.load(fp)
        r = torch.randint(Xtr.shape[0], size=(10 ** 4,))
        X, Y = Xtr[r, :], Ytr[r, :].to(Xtr.dtype)

        pdb.set_trace()

        def k_fn(z, lam):
            ret = JACOBIAN(
                lambda z: loss_fn(z, lam, X, Y),
                z,
                create_graph=True,
            )
            return ret

        lam = torch.tensor([LAM], device=device)
        its = list(results["iters"].keys())

        z = torch.tensor(results["iters"][its[-1]]).to(device)
        print("Acc: %3.1f" % (1e2 * acc_fn(z, Xts, Yts)))

        t = time.time()
        z, lam = z.requires_grad_(), lam.requires_grad_()
        g, J = k_fn(z, lam), []
        for i in tqdm(range(z.numel())):
            J.append(grad(g.reshape(-1)[i], z, retain_graph=True))
        print("Elapsed %9.4e" % (time.time() - t))

        t = time.time()
        J = JACOBIAN(lambda z: k_fn(z, lam), z)
        print("Elapsed %9.4e" % (time.time() - t))

        pdb.set_trace()

        Dpz = implicit_grads_1st(k_fn, z, lam)

        # pdb.set_trace()
    elif mode == "eval_slurm":
        torch.set_num_threads(4)
        assert len(sys.argv) >= 3
        CONFIG = dict(idx=int(sys.argv[2]))

        with gzip.open(fname, "rb") as fp:
            results = pickle.load(fp)
        X = torch.as_tensor(results["X"], device=device)
        Y = torch.as_tensor(results["Y"], device=device)
        # r = torch.randint(X.shape[0], size=(10 ** 1,))
        # X, Y = X[r, ...], Y[r, ...]

        def k_fn(z, lam):
            ret = JACOBIAN(
                lambda z: loss_fn(z, lam, X, Y),
                z,
                create_graph=True,
            )
            return ret

        fname = "data/Dpz_%03d.pkl.gz"
        lam = torch.tensor([LAM], device=device)
        idx, results_dpz = 0, odict()
        its = np.array(list(results["iters"].keys()))

        its1 = its[its <= first_phase_it]
        its2 = its[its > first_phase_it]

        # its = its[np.round(np.linspace(0, len(its) - 1, 50)).astype(int)]
        its = np.concatenate(
            [
                its1[np.round(np.linspace(0, len(its1) - 1, 25)).astype(int)],
                its2[np.round(np.linspace(0, len(its2) - 1, 25)).astype(int)],
            ]
        )
        print(its)

        for it in its:
            if idx == CONFIG["idx"]:
                results_dpz[it] = dict()
                z = torch.tensor(results["iters"][it]).to(device)
                print("Acc: %3.1f" % (1e2 * acc_fn(z, Xts, Yts)))
                print("||g|| = %9.4e" % k_fn(z, lam).norm().cpu())

                t = time.time()
                Dzk = JACOBIAN(lambda z: k_fn(z, lam), z)
                print("Elapsed %9.4e s" % (time.time() - t))
                # Dzk = torch_grad(lambda z: k_fn(z, lam), verbose=True)(z)

                results_dpz[it]["Dzk"] = Dzk.cpu().detach().numpy()
                optimizations = dict(Dzk=Dzk)
                Dpz = implicit_grads_1st(
                    k_fn, z, lam, optimizations=optimizations
                )
                results_dpz[it]["Dpz"] = Dpz.cpu().detach().numpy()
                with gzip.open(fname % idx, "wb") as fp:
                    pickle.dump(results_dpz, fp)
            idx += 1

    else:
        print("Nothing to be done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) >= 2 else "train")
