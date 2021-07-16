import os, pdb, sys, math, time, gzip, pickle

os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch, numpy as np
from tqdm import tqdm
from collections import OrderedDict as odict
import matplotlib.pyplot as plt

import include_implicit

from implicit import implicit_grads_1st
from implicit.diff import grad, JACOBIAN, HESSIAN_DIAG, torch_grad
from implicit.opt import minimize_agd
from implicit.nn_tools import nn_all_params, nn_structure, nn_forward
from implicit.nn_tools import nn_forward2

LAM = -3

def make_nn(device):
    return torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    ).to(device)


def loss_fn_gen(fwd_fn):
    l = torch.nn.MSELoss()

    def loss_fn(z, lam, X, Y):
        return l(fwd_fn(X, z), Y) + ((10.0 ** lam) * (z ** 2)).sum()

    return loss_fn


def main(mode):
    device = torch.device("cuda")
    nn = make_nn(device)
    Xtr = torch.linspace(-10, 0, 10 ** 3, device=device)[..., None]
    Ytr = torch.sin(Xtr)
    Xts = torch.linspace(0, 10, 10 ** 3, device=device)[..., None]
    Yts = torch.sin(Xts)

    params = nn_all_params(nn)
    fwd_fn = lambda X, params: nn_forward(nn, X, params)
    loss_fn = loss_fn_gen(fwd_fn)

    lam = LAM * torch.ones(params.numel(), device=device)

    fname = "data/sin_results.pkl.gz"
    if mode == "train":
        params0 = nn_all_params(nn).clone()
        params = torch.nn.Parameter(params0.clone())
        opt = torch.optim.Adam([params], lr=1e-3)
        results = dict(iters=odict(), X=None, Y=None)
        first_phase = True

        def step_fn(X, Y):
            l = loss_fn(params, lam, X, Y)
            opt.zero_grad()
            l.backward()
            return l

        max_it = 10 ** 4
        X, Y = Xtr, Ytr
        results["X"] = X.cpu().clone().numpy()
        results["Y"] = Y.cpu().clone().numpy()
        for it in tqdm(range(max_it)):
            if (it + 1) % (max_it // 10) == 0:
                results["iters"][it] = params.detach().clone().cpu().numpy()
            if (it + 1) % (max_it // 10) == 0:
                plt.figure(100)
                plt.clf()
                plt.plot(
                    *[z.reshape(-1).cpu().detach().numpy() for z in [Xtr, Ytr]]
                )
                Yp = nn_forward(nn, Xtr, params)
                plt.plot(
                    *[z.reshape(-1).cpu().detach().numpy() for z in [Xtr, Yp]]
                )
                plt.draw()
                plt.pause(1e-2)
            old_params = params.detach().clone()
            l = opt.step(lambda: step_fn(X, Y))
            g = params.grad.detach().norm()
            imprv = (params.detach() - old_params).norm()
            tqdm.write(
                "%05d %9.4e %9.4e %9.4e"
                % (it, float(l), float(g), float(imprv))
            )
        with gzip.open(fname, "wb") as fp:
            pickle.dump(results, fp)
    elif mode == "eval":
        with gzip.open(fname, "rb") as fp:
            results = pickle.load(fp)
        pdb.set_trace()
        X, Y = [torch.as_tensor(results[k], device=device) for k in ["X", "Y"]]
        def k_fn(z, lam):
            ret = JACOBIAN(
                lambda z: loss_fn(z, lam, X, Y),
                z,
                create_graph=True,
            )
            return ret

        its = list(results["iters"].keys())
        z = torch.tensor(results["iters"][its[-1]]).to(device)
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

        def k_fn(z, lam):
            ret = JACOBIAN(
                lambda z: loss_fn(z, lam, X, Y),
                z,
                create_graph=True,
            )
            return ret

        fname = "data/sin_Dpz_%03d.pkl.gz"
        idx, results_dpz = 0, odict()
        its = np.array(list(results["iters"].keys()))
        #its = its[np.round(np.linspace(0, len(its) - 1, 50)).astype(int)]

        for it in its:
            if idx == CONFIG["idx"]:
                results_dpz[it] = dict()
                z = torch.tensor(results["iters"][it]).to(device)
                print("||g|| = %9.4e" % k_fn(z, lam).norm().cpu())

                Dzk = JACOBIAN(lambda z: k_fn(z, lam), z)
                # Dzk = torch_grad(lambda z: k_fn(z, lam), verbose=True)(z)

                results_dpz[it]["Dzk"] = Dzk.cpu().detach().numpy()
                Dpz = implicit_grads_1st(k_fn, z, lam, Dzk=Dzk)
                results_dpz[it]["Dpz"] = Dpz.cpu().detach().numpy()
                with gzip.open(fname % idx, "wb") as fp:
                    pickle.dump(results_dpz, fp)
            idx += 1

    else:
        print("Nothing to be done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) >= 2 else "train")
