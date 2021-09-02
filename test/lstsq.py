import sys, os, math, pdb

import torch, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from objs import LS, OPT_with_diag

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
from implicit import generate_fns, JACOBIAN, HESSIAN, HESSIAN_DIAG
from implicit.opt import minimize_sqp

t = lambda x: x.transpose(-1, -2)


def feat_map(X):
    return torch.cat([X[..., :1] ** 0, X, X ** 2, torch.cos(X + 0.1)], -1)


def lfn(Z, Y, W, *params):
    (lam_diag,) = params
    return (
        torch.sum((Z @ W - Y) ** 2) / Z.shape[-2]
        + torch.sum((10.0 ** lam_diag.reshape(-1)[:, None]) * (W ** 2))
    ) / 2


def k_fn(W, *params):
    global Ztr, Ytr, OPT
    lfn_ = lambda W: lfn(Ztr, Ytr, W, *params)
    return JACOBIAN(lfn_, W, create_graph=True)


def opt_fn(*params):
    global Ztr, Ytr, OPT

    topts = dict(device=Ztr.device, dtype=Ztr.dtype)
    W = torch.zeros((Ztr.shape[-1], Ytr.shape[-1]), **topts)

    H = JACOBIAN(lambda W: k_fn(W, *params), W).reshape((W.numel(),) * 2)
    g = k_fn(W, *params).reshape(W.numel())
    W = -torch.linalg.solve(H, g).reshape(W.shape)
    return W


def loss_fn(W, *params):
    global Zts, Yts, OPT
    return lfn(Zts, Yts, W, *params)


if __name__ == "__main__":
    device, dtype = torch.device("cpu"), torch.float64
    torch.set_default_dtype(dtype)
    topts = dict(device=device, dtype=dtype)

    fn = lambda x: torch.cos(x)
    Xtr = torch.linspace(-10, 0, 10 ** 5, **topts)[..., None]
    Ytr = fn(Xtr)
    Xts = torch.linspace(-10, 10, 10 ** 5, **topts)[..., None]
    Yts = fn(Xts)
    Ztr, Zts = feat_map(Xtr), feat_map(Xts)

    OPT = OPT_with_diag(LS())

    f_fn, g_fn, h_fn = generate_fns(loss_fn, opt_fn, k_fn)

    params = (-0 * torch.ones(Ztr.shape[-1], **topts),)

    ps = minimize_sqp(
        f_fn, g_fn, h_fn, params[0], verbose=True, max_it=10, reg0=1e-9
    )
    params = (ps,)

    W = opt_fn(*params)
    W2 = opt_fn(params[0] + 1e0)
    pdb.set_trace()
    k = k_fn(W, *params)
    f, g, h = f_fn(*params), g_fn(*params), h_fn(*params)

    dW = torch.randn(W.shape, **topts)
    # dlam = torch.tensor([0.0, 0.0, 0.0, 1.0])
    dlam = torch.randn(params[0].shape, **topts)

    Yp = Zts @ W

    plt.figure()
    plt.plot(Xtr.reshape(-1), Ytr.reshape(-1), label="train")
    plt.plot(Xts.reshape(-1), Yts.reshape(-1), label="test")
    plt.plot(Xts.reshape(-1), Yp.reshape(-1).detach(), label="pred")
    plt.legend()
    plt.draw_all()
    plt.pause(1e-1)

    N = 30
    S, L = torch.meshgrid(
        torch.linspace(-0.01, 0.01, N), torch.linspace(-4, 4, N)
    )
    pts = torch.stack([S.reshape(-1), L.reshape(-1)], -1)
    f = torch.zeros(S.numel())
    for (i, pt) in enumerate(tqdm(pts)):
        s, l = pt
        f[i] = loss_fn(W + s * dW, params[0] + l * dlam)
    f = f.reshape(S.shape)
    plt.figure()
    plt.contourf(S.detach(), L.detach(), torch.log(f.detach()), 50)
    plt.colorbar()

    # N = 100
    # L = torch.linspace(-3, 3, N, **topts)
    # f = torch.zeros(L.numel(), **topts)
    # for (i, l) in enumerate(L):
    #    f[i] = loss_fn(opt_fn(params[0] + dlam * l), params[0] + dlam * l)
    # plt.figure()
    # plt.plot(L, f.detach())
    # plt.show()

    plt.draw_all()
    plt.pause(1e-1)
    pdb.set_trace()
