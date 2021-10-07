import pdb, os, sys, time, gzip, pickle, math

import torch, numpy as np

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
from implicit.utils import t, topts, is_equal, to_tuple
from implicit.opt import minimize_lbfgs, minimize_sqp
from implicit.diff import JACOBIAN


def Z2Za(Z, sig, d=None):
    d = Z.shape[-1] // 2 if d is None else d
    dist = Z[:, -d:]
    return torch.cat(
        [Z[:, :-d], torch.softmax(-dist / (10.0 ** sig), dim=1)], -1
    )


def poly_feat(X, n=1, centers=None):
    Z = torch.cat([X[..., 0:1] ** 0] + [X ** i for i in range(1, n + 1)], -1)
    if centers is not None:
        t_ = time.time()
        dist = torch.norm(X[..., None, :] - centers, dim=-1) / X.shape[-1]
        Z = torch.cat([Z, dist], -1)
    return Z


class LS:
    def pred(self, W, Z, lam=None):
        return Z @ W

    def solve(self, Z, Y, lam):
        n = Z.shape[-2]
        A = t(Z) @ Z / n + (10.0 ** lam) * torch.eye(Z.shape[-1], **topts(Z))
        return torch.cholesky_solve(t(Z) @ Y / n, torch.linalg.cholesky(A))

    def fval(self, W, Z, Y, lam):
        return (
            torch.sum((Z @ W - Y) ** 2) + (10.0 ** lam) * torch.sum(W ** 2)
        ) / 2

    def grad(self, W, Z, Y, lam):
        n = Z.shape[-2]
        return t(Z) @ (Z @ W - Y) / n + (10.0 ** lam) * W

    def hess(self, W, Z, Y, lam):
        H = JACOBIAN(lambda W: self.grad(W, Z, Y, lam), W)
        return H

    def Dzk_solve(self, W, Z, Y, lam, rhs, T=False, diag_reg=None):
        lam = lam.detach()
        n = Z.shape[-2]
        A = t(Z) @ Z / n + (10.0 ** lam) * torch.eye(Z.shape[-1], **topts(Z))
        rhs_shape = rhs.shape
        rhs = rhs.reshape((A.shape[-1], -1))
        if diag_reg is not None:
            l = diag_reg.numel() // A.shape[-1]
            xs = [None for _ in range(l)]
            rhs = rhs.reshape((A.shape[-1], l, -1))
            for i in range(l):
                diag_reg_ = diag_reg[i * A.shape[-1] : (i + 1) * A.shape[-1]]
                F = torch.linalg.cholesky(A + torch.diag(diag_reg_.reshape(-1)))
                xs[i] = torch.cholesky_solve(rhs[:, i, ...], F)
            sol = torch.stack(xs, 1)
            return sol.reshape(rhs_shape)
        else:
            F = torch.linalg.cholesky(A)
            return torch.cholesky_solve(rhs, F).reshape(rhs_shape)


class CE:
    def __init__(self, max_it=8, verbose=False):
        self.hess_map = dict()
        self.fact_map = dict()
        self.max_it, self.verbose = max_it, verbose

    @staticmethod
    def _Yp_aug(W, X):
        Yp = X @ W
        zeros = torch.zeros(Yp.shape[:-1] + (1,), **topts(Yp))
        return torch.cat([Yp, zeros], -1)

    @staticmethod
    def pred(W, X, lam=None):
        Yp = CE._Yp_aug(W, X)
        return torch.softmax(Yp, -1)

    def solve(self, X, Y, lam, method=""):
        if method == "cvx":
            import cvxpy as cp

            W = cp.Variable((X.shape[-1], Y.shape[-1] - 1))
            Yp = X.detach().numpy() @ W
            Yp_aug = cp.hstack([Yp, np.zeros((Yp.shape[-2], 1))])
            obj = (
                -cp.sum(cp.multiply(Y[..., :-1].detach().numpy(), Yp))
                + cp.sum(cp.log_sum_exp(Yp_aug, 1))
            ) / X.shape[-2] + 0.5 * (10.0 ** lam) * cp.sum_squares(W)
            prob = cp.Problem(cp.Minimize(obj))
            prob.solve(cp.MOSEK)
            assert prob.status in ["optimal", "optimal_inaccurate"]
            W = torch.tensor(W.value, **topts(X))
        else:
            f_fn = lambda W: self.fval(W, X, Y, lam)
            g_fn = lambda W: self.grad(W, X, Y, lam)
            h_fn = lambda W: self.hess(W, X, Y, lam)
            mask = torch.argmax(Y, -1)[..., None]
            Y_ls = Y.clone() * 0 + -1
            Y_ls.scatter_(-1, mask, torch.ones(mask.shape, **topts(Y)))

            W = LS().solve(X, Y_ls, lam)[..., :-1]
            W = minimize_lbfgs(
                f_fn,
                g_fn,
                W,
                max_it=self.max_it,
                verbose=self.verbose,
                lr=1e-1,
                use_tqdm=False,
            )
            # W = minimize_sqp(f_fn, g_fn, h_fn, W, max_it=1, verbose=False)
        return W

    @staticmethod
    def fval(W, X, Y, lam):
        Yp = X @ W
        Yp_aug = CE._Yp_aug(W, X)
        return (
            -torch.sum(Y[..., :-1] * Yp) + torch.sum(torch.logsumexp(Yp_aug, 1))
        ) / X.shape[-2] + 0.5 * (10.0 ** lam) * torch.sum(W ** 2)

    @staticmethod
    def grad(W, X, Y, lam):
        Yp_aug = CE._Yp_aug(W, X)
        Yp_softmax = torch.softmax(Yp_aug, -1)[..., :-1]
        return (
            t(X) @ (Yp_softmax - Y[..., :-1]) / X.shape[-2] + (10.0 ** lam) * W
        )

    def hess(self, W, X, Y, lam):
        key = to_tuple(W, X, Y, lam)
        if key in self.hess_map:
            print("reusing hessian")
            return self.hess_map[key]

        Yp_aug = CE._Yp_aug(W, X)
        s = torch.softmax(Yp_aug, -1)
        Ds = -s[..., :-1, None] @ s[..., None, :-1] + torch.diag_embed(
            s[..., :-1], 0
        )

        # t_ = time.time()
        H = (
            torch.einsum(
                "bijkl,bijkl,bijkl->ijkl",
                Ds[..., None, :, None, :],
                X[..., :, None, None, None],
                X[..., None, None, :, None],
            )
            / X.shape[-2]
        )
        # print("CPU time: %9.4e" % (time.time() - t_))

        # t_ = time.time()
        # Ds, X = Ds.cuda(), X.cuda()
        # H = (
        #    torch.einsum(
        #        "bijkl,bijkl,bijkl->ijkl",
        #        Ds[..., None, :, None, :],
        #        X[..., :, None, None, None],
        #        X[..., None, None, :, None],
        #    )
        #    / X.shape[-2]
        # ).cpu()
        # print("GPU time: %9.4e" % (time.time() - t_))

        H = H + (10.0 ** lam) * torch.eye(W.numel(), **topts(H)).reshape(
            H.shape
        )

        # self.hess_map[key] = H.detach()
        return H

    def Dzk_solve(self, W, X, Y, lam, rhs=None, T=False):
        eps = 1e-7
        # key = to_tuple(W, X, Y, lam)
        # if key in self.fact_map:
        #    print("reusing F")
        #    F = self.fact_map[key]
        # else:
        #    H = self.hess(W, X, Y, lam).reshape((W.numel(),) * 2)
        #    F = torch.linalg.cholesky(H)
        #    # self.fact_map[key] = F.detach()
        H = self.hess(W, X, Y, lam).reshape((W.numel(),) * 2)
        F = torch.linalg.cholesky(H)
        rhs_ = rhs.reshape((F.shape[-1], -1))
        return torch.cholesky_solve(rhs_, F).reshape(rhs.shape)


class OPT_with_centers:
    def __init__(self, OPT, d):
        self.OPT = OPT
        self.d = d

    def get_params(self, param):
        sig, lam = param[0], param[1]
        return torch.clamp(sig, -1, 1), torch.clamp(lam, -5, 4)

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


class OPT_with_diag:
    def __init__(self, OPT):
        self.OPT = OPT

    def get_params(self, param):
        lam_diag, lam = param[1:], param[0]
        return torch.clamp(lam_diag, -5, 4), torch.clamp(lam, -5, 4)

    def pred(self, W, Z, param):
        lam_diag, lam = self.get_params(param)
        return self.OPT.pred(W, Z, lam)

    def solve(self, Z, Y, param):
        lam_diag, lam = self.get_params(param)
        f_fn = lambda W: self.fval(W, Z, Y, param)
        g_fn = lambda W: self.grad(W, Z, Y, param)
        h_fn = lambda W: self.hess(W, Z, Y, param)
        W = self.OPT.solve(Z, Y, lam)
        ret = minimize_lbfgs(f_fn, g_fn, W, verbose=False, max_it=50, lr=1e-1)
        #ret = minimize_sqp(
        #    f_fn,
        #    g_fn,
        #    h_fn,
        #    W,
        #    verbose=True,
        #    max_it=100,
        #    reg0=1e-9,
        #    force_step=True,
        #)
        return ret
        # return self.OPT.solve(Z, Y, lam)

    def fval(self, W, Z, Y, param):
        lam_diag, lam = self.get_params(param)
        fval_ = self.OPT.fval(W, Z, Y, lam)
        return fval_ + 0.5 * torch.sum(
            (10.0 ** lam_diag).reshape(W.shape) * W ** 2
        )

    def grad(self, W, Z, Y, param):
        lam_diag, lam = self.get_params(param)
        grad_ = self.OPT.grad(W, Z, Y, lam)
        return grad_ + (10 ** lam_diag).reshape(W.shape) * W

    def hess(self, W, Z, Y, param):
        lam_diag, lam = self.get_params(param)
        hess_ = self.OPT.hess(W, Z, Y, lam)
        return hess_ + torch.diag(10 ** lam_diag).reshape(hess_.shape)

    def Dzk_solve(self, W, Z, Y, param, rhs, T=False):
        lam_diag, lam = self.get_params(param)
        try:
            diag_reg = 10 ** lam_diag
            return self.OPT.Dzk_solve(W, Z, Y, lam, rhs, T=T, diag_reg=diag_reg)
        except TypeError:
            H = self.OPT.hess(W, Z, Y, lam).reshape((W.numel(),) * 2)
            F = torch.linalg.cholesky(
                H + torch.diag((10 ** lam_diag).reshape(W.numel()))
            )
            rhs_ = rhs.reshape((F.shape[-1], -1))
            return torch.cholesky_solve(rhs_, F).reshape(rhs.shape)


class OPT_conv:
    def __init__(self, OPT, in_channels=1, out_channels=1, stride=2):
        self.OPT = OPT
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def get_params(self, params):
        params = params.reshape(-1)
        lam = params[0]
        C0 = params[1 : 1 + self.out_channels]
        C = params[C0.numel() + 1 :]
        n = round(math.sqrt(C.numel() / self.in_channels / self.out_channels))
        lam = torch.clamp(lam, -4, 1)
        return lam, C0, C.reshape((self.out_channels, self.in_channels, n, n))

    def conv(self, Z, C0, C):
        n = round(math.sqrt(Z.shape[-1] / self.in_channels))
        Za = Z.reshape((-1, self.in_channels, n, n))
        Za = torch.nn.functional.conv2d(Za, C, stride=self.stride)
        Za = Za + C0[..., None, None]
        Za = Za.reshape((-1, Za[0, ...].numel()))
        return torch.cat([torch.tanh(Za), Za[..., 0:1] ** 0], -1)

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


class OPT_conv_poly:
    def __init__(self, OPT, in_channels=1, out_channels=1, stride=2, d=2):
        self.OPT = OPT
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.d = d

    def get_params(self, params):
        params = params.reshape(-1)
        lam, C0, C = params[0], params[1], params[2:]
        n = round(math.sqrt(C.numel() / self.in_channels / self.out_channels))
        lam = torch.clamp(lam, -4, 1)
        return lam, C0, C.reshape((self.out_channels, self.in_channels, n, n))

    def conv(self, Z, C0, C):
        n = round(math.sqrt(Z.shape[-1] / self.in_channels))
        Za = Z.reshape((-1, self.in_channels, n, n))
        Za = torch.nn.functional.conv2d(Za, C, stride=self.stride)
        Za = Za.reshape((-1, Za[0, ...].numel())) + C0
        return poly_feat(Za, self.d)

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


if False and __name__ == "__main__":
    import torch, matplotlib.pyplot as plt

    torch.set_printoptions(threshold=10 ** 7, precision=3)
    torch.set_default_dtype(torch.double)
    W = torch.randn((3, 2))
    X = torch.randn((10, 3))
    Y = torch.randn((10, 2 + 1))
    lam = -99

    # loss_fn = lambda W: torch.sum(
    #    torch.logsumexp(
    #        torch.cat([X @ W, torch.zeros(X.shape[:-1] + (1,))], -1), -1
    #    )
    # )
    # h = hessian(loss_fn)(W)
    # s = torch.softmax(
    #    torch.cat([X @ W, torch.zeros(X.shape[:-1] + (1,))], -1), -1
    # )
    # temp = (-s[..., :, None] @ s[..., None, :] + torch.diag_embed(s, 0))
    # temp = temp[..., :-1, :-1]
    # XTX = X[..., None] @ X[..., None, :]
    # temp_ = temp[..., None, :, None, :]
    # XTX_ = XTX[..., :, None, :, None]
    # temp2 = torch.sum(temp_ * XTX_, 0)
    # temp3 = torch.einsum("bijkl,bijkl->ijkl", temp_, XTX_)
    # pdb.set_trace()

    loss = lambda W: CE.fval(W, X, Y, lam)
    g = grad(loss)(W)
    g2 = CE.grad(W, X, Y, lam)

    h = hessian(lambda W: CE.fval(W, X, Y, lam))(W)
    h2 = CE.hess(W, X, Y, lam).detach()

    plt.imshow(torch.log10(torch.abs(h - h2).reshape((W.numel(),) * 2)))
    plt.colorbar()
    plt.draw()
    plt.pause(1e-2)

    pdb.set_trace()
