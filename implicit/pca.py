import time, os, sys, math, pdb

import matplotlib.pyplot as plt, numpy as np, torch
from mpl_toolkits import mplot3d
from tqdm import tqdm

from .utils import topts


def visualize_landscape(loss_fn, x_hist, N=30, log=True):
    param_shape = x_hist[0].shape
    if isinstance(x_hist, list) or isinstance(x_hist, tuple):
        X = torch.stack(x_hist, 0).reshape((-1, x_hist[0].numel()))
    else:
        X = x_hist.reshape((-1, x_hist[0].numel()))
    X_mean = torch.mean(X, -2)
    X = X - X_mean[None, :]

    U = torch.svd(X.T)[0][:, :2]  # find the first 2 dimensions
    X_projected = (U.T @ X.T).T.detach()  # project onto the first 2 dimensions

    scale = 30.0 * torch.mean(torch.std(X_projected, -2))
    Xp, Yp = torch.meshgrid(
        *((torch.linspace(-scale / 2, scale / 2, N, **topts(X)),) * 2)
    )
    pts = torch.stack([Xp.reshape(-1), Yp.reshape(-1)], -1)
    ls = []
    for i in tqdm(range(pts.shape[0])):
        pt = pts[i, :]
        ls.append(
            loss_fn(
                (U @ (pt + X_projected[-1, :]) + X_mean[None, :]).reshape(
                    param_shape
                )
            )
        )
    Zp = torch.stack(ls).reshape(Xp.shape)
    l_optimal = min(loss_fn(x_hist[-1]), torch.min(Zp))

    if log:
        Zp = torch.log10(Zp - l_optimal + 1e-7)
    else:
        Zp = Zp - l_optimal + 1e-7

    Xp, Yp, Zp = [z.detach().numpy() for z in [Xp, Yp, Zp]]

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    X_projected_loss = torch.stack(
        [
            loss_fn((U @ pt + X_mean[None, :]).reshape(param_shape))
            for pt in X_projected
        ]
    )

    if log:
        X_projected_loss = torch.log10(X_projected_loss - l_optimal + 1e-7)
    else:
        X_projected_loss = X_projected_loss - l_optimal + 1e-7

    ax.plot(
        X_projected[:, 0], X_projected[:, 1], X_projected_loss, "ro", alpha=0.5
    )
    ax.plot(
        X_projected[:, 0], X_projected[:, 1], X_projected_loss, "r", alpha=0.5
    )
    ax.plot_surface(Xp, Yp, Zp)
    fig.tight_layout()

    plt.figure()
    plt.contourf(Xp, Yp, Zp, 100)
    plt.colorbar()
    plt.plot(X_projected[:, 0], X_projected[:, 1], color="red")
    plt.scatter(X_projected[:, 0], X_projected[:, 1], color="red")
    plt.scatter(X_projected[-1, 0], X_projected[-1, 1], color="black")
    plt.tight_layout()

    plt.draw_all()
    plt.pause(1e-1)
    return np.stack([Xp, Yp], -1), Zp

#def fit_quadratic(X, f):
#    topts = dict(device=X.device, dtype=X.dtype)
#    n = X.shape[-1]
#    L = torch.randn((n, n), **topts)
#    p = torch.randn((n,), **topts)
#    loss_fn = lambda z: 

if __name__ == "__main__":
    import tensorly as tl

    torch.set_default_dtype(torch.float64)

    A = torch.randn((100, 3)) @ torch.randn((3, 1000))
    A = A + 1e-2 * torch.randn(A.shape)

    t = time.time()
    U, S, V = torch.svd(A)
    print("%020s: %9.4e" % ("SVD", time.time() - t))
    err = torch.norm(U[:, :3] @ torch.diag(S[:3]) @ V[:, :3].T - A)
    print("err = %9.4e" % err)
    print(tuple(U.shape), tuple(S.shape), tuple(V.shape))

    t = time.time()
    U, S, V = tl.truncated_svd(A, n_eigenvecs=3)
    print("%020s: %9.4e" % ("Truncated SVD", time.time() - t))
    U, S, V = [torch.tensor(z) for z in [U, S, V]]
    err = torch.norm(U @ torch.diag(S) @ V - A)
    print("err = %9.4e" % err)
    print(tuple(U.shape), tuple(S.shape), tuple(V.shape))

    t = time.time()
    U, S, V = torch.pca_lowrank(A.T, q=3)
    print("%020s: %9.4e" % ("PCA Lowrank", time.time() - t))
    err = torch.norm(U @ torch.diag(S) @ V.T - A.T)
    print(tuple(U.shape), tuple(S.shape), tuple(V.shape))
    print("err = %9.4e" % err)
