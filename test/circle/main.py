import pdb, os, sys, math

import numpy as np, matplotlib.pyplot as plt

# k(x, y, r) = x^2 + y^2 - r^2 = 0
def ky_fn(x, y, r):
    return 2 * y


def kx_fn(x, y, r):
    return 2 * x

def grad(x, y, r, reg=0.0):
    return -kx_fn(X, Y, r) / (ky_fn(X, Y, r) + reg)


def project_onto_circle(Y, X):
    pt = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    pt = pt / np.linalg.norm(pt, axis=-1)[..., None]
    assert np.mean(np.linalg.norm(pt, axis=1)) == 1.0
    X_, Y_ = pt[:, 0].reshape(X.shape), pt[:, 1].reshape(Y.shape)
    return X_, Y_

def quiver(X, Y, G):
    dx = 0.1
    U, V = (G * 0) + dx, G * dx
    #norm = np.sqrt(U ** 2 + V ** 2)
    norm = 1.0
    U, V = U / norm, V / norm
    plt.quiver(X, Y, U, V, scale=15)
    plt.show()


if __name__ == "__main__":
    th = np.linspace(0, 2 * math.pi, 10 ** 3)
    N = 30
    X, Y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
    #G = grad(X, Y, 1.0)
    G = -kx_fn(X, Y, 1.0) / 1.0
    G_ref = grad(*project_onto_circle(X, Y), 1.0)

    err = np.abs(G - G_ref) / np.abs(G_ref)
    serr = (G * G_ref) / (np.abs(G) * np.abs(G_ref))
    pdb.set_trace()

    plt.figure()
    plt.contourf(X, Y, err, 50)
    plt.colorbar()

    plt.figure()
    plt.contourf(X, Y, serr, 50)
    plt.colorbar()

    plt.show()
