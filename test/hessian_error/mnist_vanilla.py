import os, pdb, sys, math, time, gzip, pickle

os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch, numpy as np
from tqdm import tqdm
from collections import OrderedDict as odict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import include_implicit

import implicit.interface as interface 

jaxm = interface.init(device="cpu", dtype=np.float64)

from implicit.implicit import implicit_jacobian
from implicit.diff import JACOBIAN, HESSIAN_DIAG
from implicit.opt import minimize_agd
from implicit.nn_tools import nn_all_params, nn_forward_gen

#from mnist import train, test
from fashion import train, test

cat = lambda *args: np.concatenate(list(args))

LAM = -6.0


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
    def loss_fn(z, lam, X, Y):
        # Cross-entropy
        return -jaxm.sum(
            Y * jaxm.softmax(fwd_fn(X, z), axis=-1) / Y.shape[-2]
        ) + (10.0 ** lam) * jaxm.sum(z ** 2)
        # MSE
        return jaxm.mean((Y - fwd_fn(X, z)) ** 2) + 10.0 ** lam * jaxm.sum(
            z ** 2
        )

    return jaxm.jit(loss_fn)


CONFIG = dict(
    n_train=10 ** 3,
)


MODE = sys.argv[1] if len(sys.argv) >= 2 else "train"
if __name__ == "__main__":
    nn = make_nn(torch.device("cpu"))
    dtype = jaxm.ones(()).dtype
    Xtr, Xts = [
        jaxm.array(z).reshape((-1, 1, 28, 28)).astype(dtype)
        for z in [train["images"], test["images"]]
    ]
    Ytr, Yts = [
        jaxm.nn.one_hot(jaxm.array(z), 10)
        for z in [train["labels"], test["labels"]]
    ]

    params = nn_all_params(nn)
    fwd_fn = jaxm.jit(nn_forward_gen(nn))

    loss_fn = loss_fn_gen(fwd_fn)
    l = loss_fn(params, LAM, Xtr, Ytr)
    grad_fn = jaxm.jit(jaxm.grad(loss_fn))

    dki_fn = jaxm.jit(
        jaxm.grad(
            lambda params, i: grad_fn(params, LAM, Xtr, Ytr).reshape(-1)[i]
        )
    )
    # dk = []
    # for i in tqdm(range(params.size)):
    #    dk.append(dki_fn(params, i))

    @jaxm.jit
    def acc_fn(params, X, Y):
        return jaxm.mean(
            jaxm.argmax(fwd_fn(X, params), -1) == jaxm.argmax(Y, -1)
        )

    ridx = jaxm.randint(0, Xtr.shape[0], (CONFIG["n_train"],))
    opt_idx = 0

    def cb_fn(*args):
        global ridx, opt_idx
        params = args[0]
        if (opt_idx + 1) % (10 ** 2) == 0:
            tqdm.write("%5.2f%%" % (acc_fn(params, Xts, Yts) * 100))
        ridx = jaxm.randint(0, Xtr.shape[0], (CONFIG["n_train"],))
        opt_idx += 1

    def f_fn(params):
        global ridx
        X, Y = Xtr[ridx, ...], Ytr[ridx, ...]
        return loss_fn(params, LAM, X, Y)

    def g_fn(params):
        global ridx
        X, Y = Xtr[ridx, ...], Ytr[ridx, ...]
        return grad_fn(params, LAM, X, Y)

    fname = "data/results.pkl.gz"
    first_phase_it = 2 * 10 ** 5
    if MODE == "train":
        params0 = nn_all_params(nn)
        lam = LAM
        results = dict(iters=odict(), X=None, Y=None)
        first_phase = True

        minimize_agd(
            f_fn,
            g_fn,
            params0,
            ai=1e-3,
            af=1e-4,
            callback_fn=cb_fn,
            verbose=True,
            max_it=5 * 10 ** 4,
            use_writer=True,
        )
        pdb.set_trace()

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
    elif MODE == "eval":
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

        Dpz = implicit_jacobian(k_fn, z, lam)

        # pdb.set_trace()
    elif MODE == "eval_slurm":
        assert len(sys.argv) >= 3
        CONFIG = dict(idx=int(sys.argv[2]))

        with gzip.open(fname, "rb") as fp:
            results = pickle.load(fp)
        X = torch.as_tensor(results["X"], device=device)
        Y = torch.as_tensor(results["Y"], device=device)
        r = torch.randint(X.shape[0], size=(10 ** 4,))
        X, Y = X[r, ...], Y[r, ...]

        def k_fn(z, lam):
            ret = JACOBIAN(
                lambda z: loss_fn(z, lam, X, Y),
                z,
                create_graph=True,
            )
            return ret

        fname = "data/Dpz_%03d_2.pkl.gz"
        lam = torch.tensor([LAM], device=device)
        idx, results_dpz = 0, odict()
        its = np.array(list(results["iters"].keys()))

        its1 = its[its <= first_phase_it]
        its2 = its[its > first_phase_it]

        its = cat(
            its1[np.round(np.linspace(0, len(its1) - 1, 25)).astype(int)],
            its2[np.round(np.linspace(0, len(its2) - 1, 25)).astype(int)],
        )

        for (idx, it) in enumerate(its):
            if idx == CONFIG["idx"]:
                results_dpz[it] = dict()
                z = torch.tensor(results["iters"][it]).to(device)
                print("Acc: %3.1f" % (1e2 * acc_fn(z, Xts, Yts)))
                print("||g|| = %9.4e" % k_fn(z, lam).norm().cpu())

                t = time.time()
                # Dzk = JACOBIAN(lambda z: k_fn(z, lam), z)
                # Dzk = torch_grad(lambda z: k_fn(z, lam), verbose=True)(z)

                lam_ = jnp.array(lam.numpy().detach().cpu())
                X_ = jnp.array(X.numpy().detach().cpu())
                Y_ = jnp.array(Y.numpy().detach().cpu())
                # gi_fn = jax.jit(jax.grad(lambda z: loss_fn(z, lam_, X_, Y_)

                lens = [
                    sum(p.numel() for p in m.parameters())
                    for m in list(nn.modules())[1:]
                ]
                lens = [l for l in lens if l != 0]
                zs = torch.split(z, lens)
                # Dzk = JACOBIAN(lambda *zs: k_fn(torch.cat(zs), lam), zs)
                # Dzk = torch_grad(lambda *zs: k_fn(torch.cat(zs), lam), verbose=True)(*zs)
                Dzk = HESSIAN_DIAG(
                    lambda *zs: loss_fn(torch.cat(zs), lam, X, Y), zs
                )
                print("Elapsed %9.4e s" % (time.time() - t))
                pdb.set_trace()

                results_dpz[it]["Dzk"] = Dzk.cpu().detach().numpy()
                optimizations = dict(Dzk=Dzk)
                Dpz = implicit_jacobian(
                    k_fn, z, lam, optimizations=optimizations
                )
                results_dpz[it]["Dpz"] = Dpz.cpu().detach().numpy()
                with gzip.open(fname % idx, "wb") as fp:
                    pickle.dump(results_dpz, fp)
    else:
        print("Nothing to be done")


if __name__ == "__main__":
    main()
