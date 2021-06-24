import time, os, sys, math, pdb

import matplotlib.pyplot as plt, numpy as np, torch
from mpl_toolkits import mplot3d
from tqdm import tqdm

import include_implicit
from implicit.utils import topts


def visualize_landscape(loss_fn, x_hist, N=30):
    param_shape = x_hist[0].shape
    if isinstance(x_hist, list) or isinstance(x_hist, tuple):
        X = torch.stack(x_hist, 0).reshape((-1, x_hist[0].numel()))
    else:
        X = x_hist.reshape((-1, x_hist[0].numel()))
    X_mean = torch.mean(X, -2)
    X = X - X_mean[None, :]

    U = torch.svd(X.T)[0][:, :2]  # find the first 2 dimensions
    Xt = (U.T @ X.T).T  # project onto the first 2 dimensions
    scale = 15.0 * torch.mean(torch.std(Xt, -2))
    Xp, Yp = torch.meshgrid(
        *((torch.linspace(-scale / 2, scale / 2, N, **topts(X)),) * 2)
    )
    pts = torch.stack([Xp.reshape(-1), Yp.reshape(-1)], -1)
    ls = []
    for i in tqdm(range(pts.shape[0])):
        pt = pts[i, :]
        ls.append(
            loss_fn(
                (U @ (pt + Xt[-1, :]) + X_mean[None, :]).reshape(param_shape)
            )
        )
    Zp = torch.stack(ls).reshape(Xp.shape)
    l_optimal = min(loss_fn(x_hist[-1]), torch.min(Zp))
    Zp = torch.log10(Zp - l_optimal + 1e-7)

    Xp, Yp, Zp = [z.detach().numpy() for z in [Xp, Yp, Zp]]

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    Xt_loss = torch.stack([loss_fn((U @ pt + X_mean[None,
        :]).reshape(param_shape)) for pt in Xt])
    Xt_loss = torch.log10(Xt_loss - l_optimal + 1e-7)
    ax.plot(Xt[:, 0], Xt[:, 1], Xt_loss, "ro", alpha=0.5)
    ax.plot(Xt[:, 0], Xt[:, 1], Xt_loss, "r", alpha=0.5)
    ax.plot_surface(Xp, Yp, Zp)
    fig.tight_layout()

    Xt = Xt.detach()

    plt.figure()
    plt.contourf(Xp, Yp, Zp, 100)
    plt.colorbar()
    plt.plot(Xt[:, 0], Xt[:, 1], color="red")
    plt.scatter(Xt[:, 0], Xt[:, 1], color="red")
    plt.scatter(Xt[-1, 0], Xt[-1, 1], color="black")
    plt.tight_layout()

    plt.draw_all()
    plt.pause(1e-1)
    pdb.set_trace()
    return


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
