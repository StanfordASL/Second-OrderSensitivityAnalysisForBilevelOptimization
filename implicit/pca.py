import time, os, sys, math, pdb, random

import matplotlib.pyplot as plt, numpy as np, torch
from mpl_toolkits import mplot3d
from tqdm import tqdm

from .interface import init
from .utils import j2n, n2j

jaxm = init()


rand_sign = lambda: random.randint(0, 1) * 2 - 1


def assess_convexity(Z):
    # assume square grid
    assert Z.shape[0] == Z.shape[-1]
    rand_pos = lambda: random.randint(1, Z.shape[-1] - 1)
    is_in = lambda idx: all(
        idx[i] >= 0 and idx[i] < Z.shape[-1] for i in range(2)
    )

    nb_checks = 1000
    nb_success = 0
    for _ in range(nb_checks):
        idx1 = (rand_pos(), rand_pos())
        while True:
            dir = rand_sign() * (2 * random.randint(0, Z.shape[-1] // 2))
            idx2 = (idx1[0] + dir, idx1[1] + dir)
            idx_middle = (idx1[0] + dir // 2, idx1[1] + dir // 2)
            if is_in(idx2):
                break
        fa, fb = Z[idx1[0], idx1[1]], Z[idx2[0], idx2[1]]
        fm = Z[idx_middle[0], idx_middle[1]]
        if (fa + fb) / 2 >= fm:
            nb_success += 1
    print("Convexity check succeeded in: %d/%d" % (nb_success, nb_checks))


def visualize_landscape(
    loss_fn, x_hist, N=30, log=True, verbose=False, zoom_scale=1.0
):
    param_shape = x_hist[0].shape
    if isinstance(x_hist, list) or isinstance(x_hist, tuple):
        X = jaxm.stack(x_hist, 0).reshape((-1, x_hist[0].size))
    else:
        X = x_hist.reshape((-1, x_hist[0].size))
    X_mean = jaxm.mean(X, -2)
    X = X - X_mean[None, :]

    # computational part ####################################
    if verbose:
        print("Taking the SVD")
    U = jaxm.linalg.svd(X.T)[0][:, :2]  # find the first 2 dimensions
    X_projected = (U.T @ X.T).T  # project onto the first 2 dimensions

    scale = 30.0 * zoom_scale * jaxm.mean(jaxm.std(X_projected, -2))
    Xp, Yp = jaxm.meshgrid(*((jaxm.linspace(-scale / 2, scale / 2, N),) * 2))
    pts = jaxm.stack([Xp.reshape(-1), Yp.reshape(-1)], -1)
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
    Zp = jaxm.stack(ls).reshape(Xp.shape)
    assess_convexity(Zp)
    l_optimal = min(loss_fn(x_hist[-1]), jaxm.min(Zp))

    if log:
        Zp = jaxm.log10(Zp - l_optimal + 1e-7)
    else:
        Zp = Zp - l_optimal + 1e-7

    Xp, Yp, Zp = [j2n(z) for z in [Xp, Yp, Zp]]

    # plotting part #########################################
    if verbose:
        print("Plotting the contour plot")
    plt.figure()
    plt.contourf(Xp, Yp, Zp, 100)
    plt.colorbar()
    plt.plot(X_projected[:, 0], X_projected[:, 1], color="red")
    plt.scatter(X_projected[:, 0], X_projected[:, 1], color="red")
    plt.scatter(X_projected[-1, 0], X_projected[-1, 1], color="black")
    plt.tight_layout()

    if verbose:
        print("Plotting the 3D surface")
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    X_projected_loss = jaxm.stack(
        [
            loss_fn((U @ pt + X_mean[None, :]).reshape(param_shape))
            for pt in X_projected
        ]
    )

    if log:
        X_projected_loss = jaxm.log10(X_projected_loss - l_optimal + 1e-7)
    else:
        X_projected_loss = X_projected_loss - l_optimal + 1e-7

    # ax.plot(
    #    X_projected[:, 0], X_projected[:, 1], X_projected_loss, "ro", alpha=0.5
    # )
    # ax.plot(
    #    X_projected[:, 0], X_projected[:, 1], X_projected_loss, "r", alpha=0.5
    # )
    cmap = plt.get_cmap("viridis")
    ax.plot_surface(Xp, Yp, Zp, cmap=cmap, alpha=0.8)
    ax.contourf(Xp, Yp, Zp, 100, zdir="z", offset=np.min(Zp), cmap=cmap)
    fig.tight_layout()

    # plt.draw_all()
    # plt.pause(1e-1)
    return np.stack([Xp, Yp], -1), Zp


if __name__ == "__main__":
    import tensorly as tl

    A = jaxm.randn((100, 3)) @ jaxm.randn((3, 1000))
    A = A + 1e-2 * jaxm.randn(A.shape)

    t = time.time()
    U, S, VT = jaxm.linalg.svd(A)
    V = V.T
    print("%020s: %9.4e" % ("SVD", time.time() - t))
    err = jaxm.norm(U[:, :3] @ jaxm.diag(S[:3]) @ V[:, :3].T - A)
    print("err = %9.4e" % err)
    print(tuple(U.shape), tuple(S.shape), tuple(V.shape))

    t = time.time()
    U, S, V = tl.truncated_svd(A, n_eigenvecs=3)
    print("%020s: %9.4e" % ("Truncated SVD", time.time() - t))
    U, S, V = [jaxm.array(z) for z in [U, S, V]]
    err = jaxm.norm(U @ jaxm.diag(S) @ V - A)
    print("err = %9.4e" % err)
    print(tuple(U.shape), tuple(S.shape), tuple(V.shape))

    t = time.time()
    U, S, V = jaxm.pca_lowrank(A.T, q=3)
    print("%020s: %9.4e" % ("PCA Lowrank", time.time() - t))
    err = jaxm.norm(U @ jaxm.diag(S) @ V.T - A.T)
    print(tuple(U.shape), tuple(S.shape), tuple(V.shape))
    print("err = %9.4e" % err)
