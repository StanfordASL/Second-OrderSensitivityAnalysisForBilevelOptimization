import sys, pdb, time, gzip, pickle

import torch, cvxpy as cp, numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import osqp

import header

from implicit.opt import minimize_lbfgs, minimize_sqp, minimize_agd

from objs import LS
import mnist

t2n = lambda x: np.copy(x.detach().cpu().clone().numpy().astype(np.float64))
n2t = lambda x, device=None, dtype=None: torch.as_tensor(
    x, device=device, dtype=dtype
)

# torch.set_printoptions(threshold=10 ** 9, linewidth=300)

spfmt = "csc"
sphcat = lambda xs: sp.hstack(xs, format=spfmt)
spvcat = lambda xs: sp.vstack(xs, format=spfmt)
spmat = lambda x: sp.csc_matrix(x)
speye = lambda n: sp.eye(n, format=spfmt)
spzeros = lambda m, n: spmat((m, n))
spdiags = lambda x: sp.diags(x, format=spfmt)
spkron = lambda a, b: sp.kron(a, b, format=spfmt)


class MSVM:
    def __init__(self, k=10, e=784):
        self.k, self.e = k, e

    def _split_vars(self, V, n):
        # assert V.numel() == self.k * self.e + 3 * n * self.k
        # W = V[: self.e * self.k].reshape((self.k, self.e)).transpose(-2, -1)
        # Zet = V[W.numel() : W.numel() + n * self.k]
        # Lam = V[W.numel() + Zet.numel() : W.numel() + 3 * Zet.numel()]

        assert V.numel() == self.k * self.e + 2 * n + n * self.k
        W = V[: self.e * self.k].reshape((self.k, self.e)).transpose(-2, -1)
        Zet = V[W.numel() : W.numel() + n]
        Lam = V[
            W.numel() + Zet.numel() : W.numel() + Zet.numel() + self.k * n + n
        ]
        return W, Zet, Lam

    def pred(self, V, Z, *params):
        assert Z.ndim == 2
        W = V[: self.e * self.k].reshape((self.k, self.e)).transpose(-2, -1)
        return Z @ W

    def _generate_problem_data(self, Z, Y, *params):
        assert Z.ndim == 2 and Y.ndim == 2
        (gam,) = params
        (n, e), k = Z.shape, Y.shape[-1]
        opts = dict(device=Z.device, dtype=Z.dtype)

        # version with a variable per example per class ########################
        # A1 = torch.kron(torch.eye(k, **opts), Z)
        # A2 = (
        #   (Y[..., None] * Z[..., None, :])
        #   .reshape((Y.shape[0], -1))
        #   .tile((k, 1))
        # )
        # D = (
        #   (Y[None, ...].argmax(-1) == torch.arange(k, **opts)[..., None])
        #   .reshape(-1)
        #   .to(Z.dtype)
        # )

        # A1_ = spmat(t2n(A1))
        # A2_ = spmat(t2n(A2))
        # Aa_ = sphcat([A1_ - A2_, -speye(n * k)])
        # Ab_ = sphcat([spzeros(n * k, e * k), -sp.eye(n * k)])
        # A_ = spvcat([Aa_, Ab_])
        # b_ = np.concatenate([t2n(D) - 1.0, np.zeros(n * k)])
        # P_ = spdiags(np.concatenate([np.ones(k * e), 1e-9 * np.ones(n * k)]))
        # q_ = np.concatenate([np.zeros(k * e), gam * np.ones(n * k)])

        # version with a single variable per example ###########################
        A1 = torch.kron(torch.eye(k, **opts), Z)
        A2 = (
            (Y[..., None] * Z[..., None, :])
            .reshape((Y.shape[0], -1))
            .tile((k, 1))
        )
        D = (
            (Y[None, ...].argmax(-1) == torch.arange(k, **opts)[..., None])
            .reshape(-1)
            .to(Z.dtype)
        )

        A1_ = spmat(t2n(A1))
        A2_ = spmat(t2n(A2))
        Aa_ = sphcat([A1_ - A2_, -spkron(np.ones((k, 1)), speye(n))])
        Ab_ = sphcat([spzeros(n, e * k), -sp.eye(n)])
        A_ = spvcat([Aa_, Ab_])
        b_ = np.concatenate([t2n(D) - 1.0, np.zeros(n)])
        P_ = spdiags(np.concatenate([np.ones(k * e), 0.0 * np.ones(n)]))
        q_ = np.concatenate([np.zeros(k * e), gam * np.ones(n)])

        return P_, q_, A_, b_

    def solve(self, Z, Y, *params, method="cvx", verbose=False):
        P, q, A, b = self._generate_problem_data(Z, Y, *params)
        if method == "cvx":
            x = cp.Variable(A.shape[-1])
            obj = cp.Minimize(0.5 * cp.sum(cp.quad_form(x, P)) + q @ x)
            cstr = [A @ x <= b]
            prob = cp.Problem(obj, cstr)
            prob.solve(cp.GUROBI, verbose=verbose)
            assert prob.status in ["optimal", "optimal_inaccurate"]
            x, lam = x.value, cstr[0].dual_value
        elif method == "osqp":
            P, A = sp.csc_matrix(P), sp.csc_matrix(A)
            l = -np.infty * np.ones(A.shape[0])
            m = osqp.OSQP()
            m.setup(P=P, q=q, l=l, u=b, A=A, verbose=verbose, polish=False)
            results = m.solve()
            assert results.info.status == "solved"
            x, lam = results.x, results.y
        return n2t(np.concatenate([x, lam]), device=Z.device, dtype=Z.dtype)

    def grad(self, W, Z, Y, *params):
        raise NotImplementedError

    def hess(self, W, Z, Y, *params):
        raise NotImplementedError

    def Dzk_solve(self, W, Z, Y, *params, rhs, T=False, diag_reg=None):
        pass


def loss_fn(V, *params):
    global OPT, Zts, Yts
    return torch.nn.CrossEntropyLoss()(OPT.pred(V, Zts), Yts.argmax(-1))


if __name__ == "__main__":
    global OPT, Zts, Yts

    device, dtype = "cpu", torch.float32
    Xtr, Ytr = mnist.train["images"], mnist.train["labels"]
    Xts, Yts = mnist.test["images"], mnist.test["labels"]

    r = np.random.randint(Xtr.shape[0], size=(100,))

    Xtr = n2t(Xtr[r, :], device=device).to(dtype)
    Ytr = torch.nn.functional.one_hot(
        n2t(Ytr[r], device=device).to(torch.long), 10
    ).to(dtype)

    Xts = n2t(Xts, device=device).to(dtype)
    Yts = torch.nn.functional.one_hot(
        n2t(Yts, device=device).to(torch.long), 10
    ).to(dtype)

    Ztr = torch.cat([Xtr[:, :1] ** 0, Xtr], -1)
    Zts = torch.cat([Xts[:, :1] ** 0, Xts], -1)

    OPT = MSVM(k=10, e=Ztr.shape[-1])

    lams, losses = torch.logspace(-9, 1, 100), []
    for lam in tqdm(lams):
        V = OPT.solve(Ztr, Ytr, lam, method="osqp")
        losses.append(float(t2n(loss_fn(V, lam))))
    losses = torch.as_tensor(losses, device=device, dtype=dtype)

    with gzip.open("data/losses.pkl.gz", "wb") as fp:
        pickle.dump((t2n(lams), t2n(losses)), fp)
    sys.exit()

    # PROFILE.print_stats(output_unit=1e-3)
    # W, Zet, Lam = OPT._split_vars(V, Ztr.shape[-2])

    print("#" * 80)
    print(W.norm())
    print(Zet.norm())
    print("#" * 80)

    acc_tr = (Ytr.argmax(-1) == OPT.pred(V, Ztr).argmax(-1)).to(dtype).mean()
    acc_ts = (Yts.argmax(-1) == OPT.pred(V, Zts).argmax(-1)).to(dtype).mean()

    print("Train acc = %5.1f%%" % (acc_tr * 1e2))
    print("Test acc =  %5.1f%%" % (acc_ts * 1e2))

    pdb.set_trace()

    # pdb.set_trace()
