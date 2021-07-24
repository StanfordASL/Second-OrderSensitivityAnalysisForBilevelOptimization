import sys, time, os, pdb, pickle, gzip, math

import torch, numpy as np
from sklearn.linear_model import ElasticNet
import cvxpy as cp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "implicit"))
from implicit.utils import t2n, n2t

####################### CVXPY ##################################################
def optimize_cvxpy(X, Y, bet1, bet2, verbose=False):
    opts = dict(device=X.device, dtype=X.dtype)
    X, Y, bet1, bet2 = t2n(X), t2n(Y), t2n(bet1), t2n(bet2)
    W = cp.Variable((X.shape[-1], Y.shape[-1]))
    obj = cp.Minimize(
        0.5 * cp.sum_squares(X @ W - Y) / X.shape[-2]
        + 0.5 * bet1 * cp.sum_squares(W)
        + bet2 * cp.norm(W, p=1)
    )
    prob = cp.Problem(obj)
    prob.solve(cp.GUROBI, verbose=verbose)
    assert prob.status in ["optimal", "optimal_inaccurate"]
    return n2t(W.value, **opts)


####################### sklearn ################################################
def optimize_sklearn(X, Y, bet1, bet2):
    opts = dict(device=X.device, dtype=X.dtype)
    X, Y, bet1, bet2 = t2n(X), t2n(Y), t2n(bet1), t2n(bet2)
    alpha, l1_ratio = bet2 + bet1, bet2 / (bet1 + bet2)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
    model.fit(X, Y)
    return n2t(model.coef_.T, **opts)


####################### ADMM ###################################################
def solve_l1(z, rho):
    return torch.where(
        torch.abs(z) < 1 / rho,
        torch.as_tensor(0.0, device=z.device, dtype=z.dtype),
        torch.where(z < 0.0, z + 1 / rho, z - 1 / rho),
    )


def solve_reg(X, Y, lam, rho, Z, cache=None):
    opts = dict(device=X.device, dtype=X.dtype)
    n = X.shape[-2]
    if cache is None:
        cache = dict(XTX=X.T @ X / n, XTY=X.T @ Y / n)
    rhs = cache["XTY"] + rho * Z
    H = cache["XTX"] + (lam + rho) * torch.eye(X.shape[-1], **opts)
    ret = torch.cholesky_solve(rhs, torch.linalg.cholesky(H))
    return ret, cache


def optimize_admm(X, Y, bet1, bet2, max_it=300, rho=1e0, verbose=False):
    opts = dict(device=X.device, dtype=X.dtype)
    n = X.shape[-2]
    U = torch.zeros((X.shape[-1], Y.shape[-1]), **opts)
    Z = torch.zeros((X.shape[-1], Y.shape[-1]), **opts)
    W, cache = solve_reg(X, Y, bet1, 0.0, Z - U)
    Z = solve_l1(W + U, rho / (bet2 + 1e-9))
    U = U + (W - Z)
    r_hist = []
    for it in range(max_it):
        W_prev = W
        W, cache = solve_reg(X, Y, bet1, rho, Z - U, cache=cache)
        Z = solve_l1(W + U, rho / (bet2 + 1e-9))
        U = U + (W - Z)
        if W_prev is not None:
            r_prim, r_dual = (Z - W).norm(), (W_prev - W).norm()
            if verbose:
                print("%9.4e --- %9.4e --- %9.4e" % (r_prim, r_dual, rho))
    assert r_prim < 1e-3 and r_dual < 1e-3
    return Z


####################### testing ################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import header
    from mnist import train, test

    def loss_fn(W, X, Y, bet1, bet2):
        return (
            torch.sum((X @ W - Y) ** 2) / X.shape[-2]
            + bet1 * torch.sum(W ** 2)
            + bet2 * torch.sum(W.abs())
        )

    def feat_map(X):
        return torch.cat([X[:, :1] ** 0, X], -1)

    device = "cuda"
    dtype = torch.float32
    opts = dict(device=device, dtype=dtype)
    n2t = lambda x: torch.as_tensor(x, **opts)
    t2n = lambda x: x.detach().cpu().numpy()

    Xtr = feat_map(n2t(train["images"]) / 255)
    # Ytr = n2t(train["labels"]).reshape((-1, 1))
    Ytr = torch.nn.functional.one_hot(
        n2t(train["labels"]).to(torch.long), 10
    ).to(dtype)

    W = torch.randn((Xtr.shape[-1], Ytr.shape[-1]), **opts)

    bet1, bet2 = -2.0, -5.0

    W = optimize_admm(Xtr, Ytr, bet1, bet2, rho=1e0, max_it=200, verbose=True)
    W.requires_grad = True
    l = loss_fn(W, Xtr, Ytr, bet1, bet2)
    g = grad(l, W)

    acc = ((Xtr @ W).argmax(-1) == Ytr.argmax(-1)).to(dtype).mean()
    print("Accuracy %5.2f%%" % (acc.cpu() * 1e2))
    print("Fill     %5.2f%%" % ((W.abs() > 1e-7).to(dtype).mean().cpu() * 1e2))

    pdb.set_trace()
