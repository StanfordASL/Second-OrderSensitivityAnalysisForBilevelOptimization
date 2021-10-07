import pdb, os, sys, time, gzip, pickle, math

import torch, numpy as np

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

from implicit.interface import init

jaxm = init()

from implicit.opt import minimize_lbfgs, minimize_sqp
from implicit.utils import t2j
from implicit.nn_tools import conv, nn_all_params, nn_forward_gen


def Z2Za(Z, sig, d=None):
    d = Z.shape[-1] // 2 if d is None else d
    dist = Z[:, -d:]
    return jaxm.cat(
        [Z[:, :-d], jaxm.softmax(-dist / (10.0 ** sig), axis=1)], -1
    )


def poly_feat(X, n=1, centers=None):
    Z = jaxm.cat([X[..., 0:1] ** 0] + [X ** i for i in range(1, n + 1)], -1)
    if centers is not None:
        t_ = time.time()
        dist = jaxm.norm(X[..., None, :] - centers, axis=-1) / X.shape[-1]
        Z = jaxm.cat([Z, dist], -1)
    return Z


class OBJ:
    def __init__(self):
        pass

    def grad(self, *args):
        self.__dict__.setdefault("grad_fn", jaxm.jit(jaxm.grad(self.fval)))
        g = self.grad_fn(*args)
        return g

    def hess(self, *args):
        self.__dict__.setdefault("hess_fn", jaxm.jit(jaxm.jacobian(self.grad)))
        H = self.hess_fn(*args)
        return H

    def Dzk_solve(self, *args, rhs=None, T=False):
        H = self.hess(*args).reshape((args[0].size, args[0].size))
        if T:
            H = jaxm.t(H)
        rhs_ = rhs.reshape((H.shape[-1], -1))
        return jaxm.linalg.solve(H, rhs).reshape(rhs.shape)


class LS(OBJ):
    def pred(self, W, Z, lam=None):
        return Z @ W

    def solve(self, Z, Y, lam):
        n = Z.shape[-2]
        A = jaxm.t(Z) @ Z / n + (10.0 ** lam) * jaxm.eye(Z.shape[-1])
        return jaxm.linalg.cholesky_solve(
            jaxm.linalg.cholesky(A), jaxm.t(Z) @ Y / n
        )

    def fval(self, W, Z, Y, lam):
        n = Z.shape[-2]
        return (
            jaxm.sum((Z @ W - Y) ** 2) / n + (10.0 ** lam) * jaxm.sum(W ** 2)
        ) / 2

    def grad(self, W, Z, Y, lam):
        n = Z.shape[-2]
        return jaxm.t(Z) @ (Z @ W - Y) / n + (10.0 ** lam) * W

    def Dzk_solve(self, W, Z, Y, lam, rhs=None, T=False):
        n = Z.shape[-2]
        A = jaxm.t(Z) @ Z / n + (10.0 ** lam) * jaxm.eye(Z.shape[-1])
        rhs_shape = rhs.shape
        rhs = rhs.reshape((A.shape[-1], -1))
        if T == True:
            F = jaxm.linalg.cholesky(A)
        else:
            F = jaxm.linalg.cholesky(jaxm.t(A))
        return jaxm.linalg.cholesky_solve(F, rhs).reshape(rhs_shape)


class CE(OBJ):
    def __init__(self, max_it=8, verbose=False, method=""):
        self.max_it, self.verbose, self.method = max_it, verbose, method

    @staticmethod
    def _Yp_aug(W, X):
        Yp = X @ W
        zeros = jaxm.zeros(Yp.shape[:-1] + (1,))
        return jaxm.cat([Yp, zeros], -1)

    @staticmethod
    def pred(W, X, lam=None):
        Yp = CE._Yp_aug(W, X)
        return jaxm.softmax(Yp, -1)

    def solve(self, X, Y, lam):
        if self.method == "cvx":
            import cvxpy as cp

            W = cp.Variable((X.shape[-1], Y.shape[-1] - 1))
            Yp = np.array(X) @ W
            Yp_aug = cp.hstack([Yp, np.zeros((Yp.shape[-2], 1))])
            obj = (
                -cp.sum(cp.multiply(np.array(Y)[..., :-1], Yp))
                + cp.sum(cp.log_sum_exp(Yp_aug, 1))
            ) / X.shape[-2] + 0.5 * (10.0 ** np.array(lam)) * cp.sum_squares(W)
            prob = cp.Problem(cp.Minimize(obj))
            prob.solve(cp.MOSEK)
            assert prob.status in ["optimal", "optimal_inaccurate"]
            W = jaxm.array(W.value)
        else:
            f_fn = lambda W: self.fval(W, X, Y, lam)
            g_fn = lambda W: self.grad(W, X, Y, lam)
            h_fn = lambda W: self.hess(W, X, Y, lam)
            Y_ls = Y * 6 - 5

            W = LS().solve(X, Y_ls, lam)[..., :-1]
            kw = dict(max_it=self.max_it, verbose=self.verbose)
            W = minimize_lbfgs(f_fn, g_fn, W, lr=1e-1, **kw)
            # W = minimize_sqp(f_fn, g_fn, h_fn, W, max_it=1, verbose=False)
        return W

    @staticmethod
    def fval(W, X, Y, lam):
        Yp = X @ W
        Yp_aug = CE._Yp_aug(W, X)
        return (
            -jaxm.sum(Y[..., :-1] * Yp)
            + jaxm.sum(jaxm.scipy.special.logsumexp(Yp_aug, 1))
        ) / X.shape[-2] + 0.5 * jaxm.sum((10.0 ** lam) * (W ** 2))


class OPT_with_centers:
    def __init__(self, OPT, d):
        self.OPT = OPT
        self.d = d

    def get_params(self, param):
        sig, lam = param[0], param[1]
        return jaxm.clip(sig, -2, 2), jaxm.clip(lam, -5, 5)

    def pred(self, W, Z, param):
        sig, lam = self.get_params(param)
        Za = Z2Za(Z, sig, self.d)
        return self.OPT.pred(W, Za, lam)

    def solve(self, Z, Y, param):
        sig, lam = self.get_params(param)
        Za = Z2Za(Z, sig, self.d)
        return self.OPT.solve(Za, Y, lam)

    def fval(self, W, Z, Y, param):
        sig, lam = self.get_params(param)
        Za = Z2Za(Z, sig, self.d)
        return self.OPT.fval(W, Za, Y, lam)

    def grad(self, W, Z, Y, param):
        sig, lam = self.get_params(param)
        Za = Z2Za(Z, sig, self.d)
        return self.OPT.grad(W, Za, Y, lam)

    def hess(self, W, Z, Y, param):
        sig, lam = self.get_params(param)
        Za = Z2Za(Z, sig, self.d)
        return self.OPT.hess(W, Za, Y, lam)

    def Dzk_solve(self, W, Z, Y, param, rhs, T=False):
        sig, lam = self.get_params(param)
        Za = Z2Za(Z, sig, self.d)
        return self.OPT.Dzk_solve(W, Za, Y, lam, rhs, T=T)


class LS_with_diag(OBJ):
    def __init__(self):
        pass

    def get_params(self, param):
        return jaxm.clip(param, -5, 5)

    def pred(self, W, Z, param):
        return Z @ W

    def solve(self, Z, Y, param):
        lam_diag = self.get_params(param)
        n = Z.shape[-2]

        L = lam_diag.reshape((Z.shape[-1], Y.shape[-1]))
        ws = [None for i in range(Y.shape[-1])]
        A_base = jaxm.t(Z) @ Z / n 
        for i in range(Y.shape[-1]):
            A = A_base + jaxm.diag(10.0 ** L[:, i])
            ws[i] = jaxm.linalg.solve(A, jaxm.t(Z) @ Y[:, i] / n)
        ret = jaxm.stack(ws, -1)

        #A = jaxm.t(Z) @ Z / n + jaxm.diag(10.0 ** lam_diag.reshape(-1))
        #ret = jaxm.linalg.solve(A, jaxm.t(Z) @ Y / n)

        return ret

    def fval(self, W, Z, Y, param):
        lam_diag = self.get_params(param)
        n = Z.shape[-2]
        return (
            jaxm.sum((Z @ W - Y) ** 2) / n
            + jaxm.sum((10.0 ** lam_diag.reshape(W.shape)) * (W ** 2))
            #+ jaxm.sum((10.0 ** lam_diag.reshape((W.shape[-2], 1))) * (W ** 2))
        ) / 2


class OPT_with_diag(OBJ):
    def __init__(self, OPT):
        self.OPT = OPT

    def get_params(self, param):
        return jaxm.clip(param, -5, 5)

    def pred(self, W, Z, param):
        lam_diag = self.get_params(param)
        return self.OPT.pred(W, Z, lam_diag)

    def solve(self, Z, Y, param):
        f_fn = lambda W: self.fval(W, Z, Y, param)
        g_fn = lambda W: self.grad(W, Z, Y, param)
        h_fn = lambda W: self.hess(W, Z, Y, param)
        W = self.OPT.solve(Z, Y, 1e-3)
        #ret = minimize_sqp(
        #    f_fn,
        #    g_fn,
        #    h_fn,
        #    W,
        #    verbose=True,
        #    max_it=10,
        #    reg0=1e-9,
        #    force_step=True,
        #)
        ret = minimize_lbfgs(f_fn, g_fn, W, verbose=True, max_it=50, lr=1e-1)
        return ret

    def fval(self, W, Z, Y, param):
        lam_diag = self.get_params(param)
        fval_ = self.OPT.fval(W, Z, Y, 0.0)
        return fval_ + 0.5 * jaxm.sum(
            (10.0 ** lam_diag).reshape(W.shape) * W ** 2
        )


class OPT_conv:
    def __init__(
        self, OPT, in_channels=1, out_channels=1, kernel_size=3, stride=2
    ):
        self.OPT = OPT
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def generate_parameter(self):
        c = torch.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=(self.stride, self.stride),
        )
        params = [t2j(z) for z in c.parameters()]
        assert len(params) == 2
        C0_idx = 0 if params[0].size == self.out_channels else 1
        C0, C = params[C0_idx], params[(C0_idx + 1) % 2]
        assert C0.size == self.out_channels
        return jaxm.cat([1e-3 * jaxm.ones(1), C0.reshape(-1), C.reshape(-1)])

    def get_params(self, params):
        params = params.reshape(-1)
        lam = params[0]
        C0 = params[1 : 1 + self.out_channels]
        C = params[C0.size + 1 :]
        n = self.kernel_size
        lam = jaxm.clip(lam, -5, 5)
        return lam, C0, C.reshape((self.out_channels, self.in_channels, n, n))

    def conv(self, Z, C0, C):
        assert Z.ndim == 2
        m = round(math.sqrt(Z.shape[-1] / self.in_channels))
        n = round(Z.shape[-1] / self.in_channels / m)
        assert Z.shape[-1] == m * n * self.in_channels
        Za = Z.reshape((Z.shape[0], self.in_channels, m, n))
        # Za = torch.nn.functional.conv2d(Za, C, stride=self.stride)
        Za = conv(Za, C, C0, stride=(self.stride, self.stride))
        Za = Za.reshape((-1, Za[0, ...].size))
        ret = jaxm.cat([Za[..., 0:1] ** 0, jaxm.tanh(Za)], -1)
        return ret

    def pred(self, W, Z, param):
        lam, C0, C = self.get_params(param)
        Za = self.conv(Z, C0, C)
        return self.OPT.pred(W, Za, lam)

    def solve(self, Z, Y, param):
        lam, C0, C = self.get_params(param)
        Za = self.conv(Z, C0, C)
        return self.OPT.solve(Za, Y, lam)

    def fval(self, W, Z, Y, param):
        lam, C0, C = self.get_params(param)
        Za = self.conv(Z, C0, C)
        return self.OPT.fval(W, Za, Y, lam)

    def grad(self, W, Z, Y, param):
        lam, C0, C = self.get_params(param)
        Za = self.conv(Z, C0, C)
        return self.OPT.grad(W, Za, Y, lam)

    def hess(self, W, Z, Y, param):
        lam, C0, C = self.get_params(param)
        Za = self.conv(Z, C0, C)
        return self.OPT.hess(W, Za, Y, lam)

    def Dzk_solve(self, W, Z, Y, param, rhs, T=False):
        lam, C0, C = self.get_params(param)
        Za = self.conv(Z, C0, C)
        return self.OPT.Dzk_solve(W, Za, Y, lam, rhs, T=T)
