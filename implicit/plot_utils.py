import sys, os, pdb, math, time

import matplotlib.pyplot as plt, torch

def plot_3d(M):
    assert M.ndim == 3
    m = round(math.sqrt(M.shape[0]))
    n = M.shape[0] // m
    fig, axes = plt.subplots(m, n)
    k = 0
    cmap = plt.get_cmap("RdBu")
    for i in range(m):
        for j in range(n):
            p = axes[i, j].imshow(M[n * i + j, :, :], cmap=cmap)
            fig.colorbar(p, ax=axes[i, j])
    fig.tight_layout()
    return fig
