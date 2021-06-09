import math, pdb, os, sys, time
from functools import reduce
from operator import mul

import matplotlib.pyplot as plt, torch

from utils import t, topts, ss, fn_with_sol_cache
from diff import torch_grad as grad, torch_hessian as hessian
from diff import torch_fwd_grad as fwd_grad

torch.set_default_dtype(torch.float64)
prod = lambda zs: reduce(mul, zs)

CHECK_GRADS = False

JACOBIAN = torch.autograd.functional.jacobian
HESSIAN = torch.autograd.functional.hessian


def implicit_grads_1st(
    k_fn, z, *params, Dg=None, Dzk=None, Dzk_solve_fn=None, full_output=False
):
    zlen, plen = prod(z.shape), [prod(param.shape) for param in params]
    cache = dict()
    if Dg is not None:
        if Dzk_solve_fn is None:
            if Dzk is None:
                Dzk = grad(k_fn, argnums=0)(z, *params)
            Dzk = Dzk.reshape((zlen, zlen))
            FT = torch.lu(t(Dzk))
            cache["FT"] = FT
            v = -torch.lu_solve(Dg.reshape((zlen, 1)), *FT)
        else:
            v = -Dzk_solve_fn(z, *params, Dg.reshape((zlen, 1)), T=True)
        v = v.detach()
        fn = lambda *params: torch.sum(
            v.reshape(zlen) * k_fn(z, *params).reshape(zlen)
        )
        Dp = JACOBIAN(fn, *params)
        Dp = [Dp] if len(params) == 1 else Dp
        # Dp = grad(fn, argnums=range(len(params)))(*params)
        Dp_shaped = [Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)]
        ret = Dp_shaped[0] if len(params) == 1 else Dp_shaped

        if CHECK_GRADS:
            print("Checking 1st order grad")
            Dpz_ = implicit_grads_1st(k_fn, z, *params)
            Dpz_ = [Dpz_] if len(params) == 1 else Dpz_
            Df = [
                Dpz_.reshape((zlen, plen))
                for (Dpz_, param, plen) in zip(Dpz_, params, plen)
            ]
            ret_ = [
                (Dg.reshape((1, zlen)) @ Df).reshape(param.shape)
                for (Df, param) in zip(Df, params)
            ]
            errs = [
                torch.norm(ret_ - Dp_shaped) / (torch.norm(Dp_shaped) + 1e-7)
                for (ret_, Dp_shaped) in zip(ret_, Dp_shaped)
            ]
            try:
                assert all(err < 1e-7 for err in errs)
            except:
                print("Gradient 1st order check failed")
                pdb.set_trace()
    else:
        # t_ = time.time()
        # Dpk_ = fwd_grad(k_fn, argnums=range(1, len(params) + 1))(z, *params)
        # print("Dpk_ took %9.4e" % (time.time() - t_))
        # t_ = time.time()
        # Dpk = grad(k_fn, argnums=range(1, len(params) + 1))(z, *params)
        # print("Dpk took %9.4e" % (time.time() - t_))
        # pdb.set_trace()
        # Dpk = fwd_grad(k_fn, argnums=range(1, len(params) + 1))(z, *params)
        Dpk = JACOBIAN(lambda *params: k_fn(z, *params), *params)
        Dpk = [Dpk] if len(params) == 1 else Dpk
        Dpk = [Dpk.reshape((zlen, plen)) for (Dpk, plen) in zip(Dpk, plen)]
        if Dzk_solve_fn is None:
            if Dzk is None:
                Dzk = grad(k_fn, argnums=0)(z, *params)
            F = torch.lu(Dzk.reshape((zlen, zlen)))
            Dpz = [-torch.lu_solve(Dpk, *F) for Dpk in Dpk]
        else:
            Dpz = [-Dzk_solve_fn(z, *params, Dpk, T=False) for Dpk in Dpk]
        Dpz_shaped = [
            Dpz.reshape(z.shape + param.shape)
            for (Dpz, param) in zip(Dpz, params)
        ]
        if Dzk_solve_fn is None:
            cache["F"] = F
        ret = Dpz_shaped if len(params) != 1 else Dpz_shaped[0]
    if Dzk_solve_fn is None:
        cache["Dzk"] = Dzk
    return (ret, cache) if full_output else ret


def implicit_grads_2nd(
    k_fn, z, *params, Dg=None, Hg=None, Dzk=None, Dzzk=None, Dzk_solve_fn=None
):
    """
    compute implicit gradients of z wrt params (or p)
    Args:
        k_fn: the implicit function k(z, *params) = 0, dk_dz(z, *params) != 0
        z: variable which we differentiate
        *params: variables wrt which we differentiate
    Keyword args:
        only_1st: whether to only compute the 1st order implicit gradients

    """
    zlen, plen = prod(z.shape), [prod(param.shape) for param in params]

    # compute the full first order 1st gradients
    Dpz, cache = implicit_grads_1st(
        k_fn, z, *params, Dzk=Dzk, Dzk_solve_fn=Dzk_solve_fn, full_output=True
    )
    Dpz = [Dpz] if len(params) == 1 else Dpz
    Dpz = [Dpz.reshape((zlen, plen)).detach() for (Dpz, plen) in zip(Dpz, plen)]

    # compute 2nd implicit gradients
    if Dg is not None:
        assert Dg.numel() == zlen
        assert Hg is None or Hg.numel() == zlen ** 2

        Dg_ = Dg.reshape((zlen, 1))
        Hg_ = Hg.reshape((zlen, zlen)) if Hg is not None else Hg
        #H1 = [t(Dpz) @ Hg_ @ Dpz for Dpz in Dpz]

        # compute the left hand vector in the VJP
        if Dzk_solve_fn is None:
            Dzk = cache["Dzk"].reshape((zlen, zlen))
            FT = torch.lu(t(Dzk.reshape((zlen, zlen))))
            v = -torch.lu_solve(Dg_.reshape((zlen, 1)), *FT).reshape((zlen, 1))
        else:
            v = -Dzk_solve_fn(z, *params, Dg_.reshape((zlen, 1)), T=True)
        v = v.detach()
        fn = lambda z, *params: torch.sum(
            v.reshape(zlen) * k_fn(z, *params).reshape(zlen)
        )

        # compute the 2nd order derivatives consisting of 4 terms
        # Dpp1 = [
        #    hessian(fn, argnums=i)(z, *params).reshape((plen, plen))
        #    for (i, plen) in zip(range(1, len(params) + 1), plen)
        # ]
        Dpp1 = HESSIAN(
            lambda *params: fn(z, *params), *params, create_graph=True
        )
        Dpp1 = [[Dpp1]] if len(params) == 1 else Dpp1
        Dpp1 = [
            Dpp1[i].reshape((plen, plen))
            for (Dpp1, i, plen) in zip(Dpp1, range(len(params)), plen)
        ]
        # temp = grad(
        #    grad(fn, argnums=0, create_graph=True),
        #    argnums=range(1, len(params) + 1),
        # )(z, *params)
        # temp = [
        #    t(
        #        grad(grad(fn, argnums=i, create_graph=True), argnums=0)(
        #            z, *params
        #        )
        #    )
        #    for i in range(1, len(params) + 1)
        # ]
        temp = JACOBIAN(
            lambda z: JACOBIAN(
                lambda *params: fn(z, *params), *params, create_graph=True
            ),
            z,
        )
        temp = [temp] if len(params) == 1 else temp
        temp = [temp.reshape((plen, zlen)) for (temp, plen) in zip(temp, plen)]
        Dpp2 = [
            (temp @ Dpz).reshape((plen, plen))
            for (temp, Dpz, plen) in zip(temp, Dpz, plen)
        ]
        Dpp3 = [t(Dpp2) for Dpp2 in Dpp2]

        #############################################################
        # version 1 ###########
        t_ = time.time()
        # Dzz = hessian(fn, argnums=0)(z, *params).reshape((zlen, zlen))
        Dzz = HESSIAN(lambda z: fn(z, *params), z).reshape((zlen, zlen))
        if Hg is not None:
            Dpp4 = [t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
        else:
            Dpp4 = [t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]
        #print("Time 1: %9.4e" % (time.time() - t_))

        # version 2 ###########
        #pdb.set_trace()
        #t_ = time.time()
        #Dpp4_ = [
        #    JACOBIAN(
        #        lambda z: JACOBIAN(
        #            lambda z: fn(z, *params), z, create_graph=True
        #        ).reshape(zlen)
        #        @ Dpz,
        #        z,
        #    ).reshape((plen, zlen))
        #    @ Dpz
        #    for (Dpz, plen) in zip(Dpz, plen)
        #]
        #print("Time 2: %9.4e" % (time.time() - t_))

        ## comparison ##########
        # assert all(
        #    torch.norm(Dpp4_ - Dpp4) < 1e-5
        #    for (Dpp4, Dpp4_) in zip(Dpp4, Dpp4_)
        # )
        #Dpp4 = Dpp4_
        #############################################################

        Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]
        #if Hg is not None:
        #    Dpp = [Dpp + t(Dpz) @ Hg_ @ Dpz for (Dpp, Dpz) in zip(Dpp, Dpz)]
        Dp = [Dg_.reshape((1, zlen)) @ Dpz for Dpz in Dpz]

        # return the results
        Dp_shaped = [Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)]
        Dpp_shaped = [
            Dpp.reshape(param.shape + param.shape)
            for (Dpp, param) in zip(Dpp, params)
        ]
        if CHECK_GRADS:
            print("Checking 2nd order grad")
            Dpz, Dppz = implicit_grads_2nd(k_fn, z, *params)
            Dpz = [Dpz] if len(params) == 1 else Dpz
            Dppz = [Dppz] if len(params) == 1 else Dppz

            Df = [Dpz.reshape((zlen, plen)) for (Dpz, plen) in zip(Dpz, plen)]
            H1 = [t(Df) @ Hg.reshape((zlen, zlen)) @ Df for Df in Df]

            Hf = [
                Dppz.reshape((zlen, plen, plen))
                for (Dppz, plen) in zip(Dppz, plen)
            ]
            H2 = [torch.sum(Dg.reshape((zlen, 1, 1)) * Hf, -3) for Hf in Hf]

            H = [
                (H1 + H2).reshape(param.shape + param.shape)
                for (H1, H2, param) in zip(H1, H2, params)
            ]
            errs = [
                torch.norm(Dpp_shaped - H) / torch.norm(Dpp_shaped)
                for (Dpp_shaped, H) in zip(Dpp_shaped, H)
            ]
            try:
                assert all(err < 1e-7 for err in errs)
            except AssertionError:
                print("Gradient 2nd order check failed")
                pdb.set_trace()
        return (
            (Dp_shaped[0], Dpp_shaped[0])
            if len(params) == 1
            else (Dp_shaped, Dpp_shaped)
        )
    else:
        # compute derivatives
        if Dzzk is None:
            Hk = [
                hessian(k_fn, argnums=i)(z, *params)
                for i in range(len(params) + 1)
            ]
            Dzzk, Dppk = Hk[0], Hk[1:]
        else:
            Dppk = [
                hessian(k_fn, argnums=i)(z, *params)
                for i in range(1, len(params) + 1)
            ]
        Dpzk = grad(
            grad(k_fn, argnums=0, create_graph=True),
            argnums=range(1, len(params) + 1),
        )(z, *params)
        Dppk = [
            Dppk.reshape((zlen, plen, plen)) for (Dppk, plen) in zip(Dppk, plen)
        ]
        Dzzk = Dzzk.reshape((zlen, zlen, zlen))
        Dpzk = [
            Dpzk.reshape((zlen, zlen, plen)) for (Dpzk, plen) in zip(Dpzk, plen)
        ]
        Dzpk = [Dpzk.transpose(-1, -2) for Dpzk in Dpzk]

        # solve the IFT equation
        lhs = [
            Dppk
            + Dzpk @ Dpz[None, ...]
            + t(Dpz)[None, ...] @ t(Dzpk)
            + (t(Dpz)[None, ...] @ Dzzk) @ Dpz[None, ...]
            for (Dpz, Dzpk, Dpzk, Dppk) in zip(Dpz, Dzpk, Dpzk, Dppk)
        ]
        F = cache["F"]
        Dppz = [
            -torch.lu_solve(lhs.reshape((zlen, plen * plen)), *F).reshape(
                (zlen, plen, plen)
            )
            for (lhs, plen) in zip(lhs, plen)
        ]

        # return computed values
        Dpz_shaped = [
            Dpz.reshape(z.shape + param.shape)
            for (Dpz, param) in zip(Dpz, params)
        ]
        Dppz_shaped = [
            Dppz.reshape(z.shape + param.shape + param.shape)
            for (Dppz, param) in zip(Dppz, params)
        ]
        return (
            (Dpz_shaped[0], Dppz_shaped[0])
            if len(params) == 1
            else (Dpz_shaped, Dppz_shaped)
        )


def detach_args(*args):
    return tuple(
        arg.detach() if isinstance(arg, torch.Tensor) else arg for arg in args
    )


def generate_fns(
    loss_fn, opt_fn, k_fn, Dzk_solve_fn=None, normalize_grad=False
):
    sol_cache = dict()
    opt_fn_ = lambda *args, **kwargs: opt_fn(*args, **kwargs).detach()

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def f_fn(z, *params):
        z = z.detach() if isinstance(z, torch.Tensor) else z
        params = detach_args(*params)
        return loss_fn(z, *params)

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def g_fn(z, *params):
        z = z.detach() if isinstance(z, torch.Tensor) else z
        params = detach_args(*params)
        g = grad(loss_fn, argnums=range(len(params) + 1))(z, *params)
        Dp = implicit_grads_1st(
            k_fn, z, *params, Dg=g[0], Dzk_solve_fn=Dzk_solve_fn
        )
        Dp = Dp if len(params) != 1 else [Dp]
        ret = [Dp + g for (Dp, g) in zip(Dp, g[1:])]
        if normalize_grad:
            ret = [z / (torch.norm(z) + 1e-7) for z in ret]
        ret = [ret.detach() for ret in ret]
        return ret[0] if len(ret) == 1 else ret

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def h_fn(z, *params):
        z = z.detach() if isinstance(z, torch.Tensor) else z
        params = detach_args(*params)
        g = grad(loss_fn, argnums=range(len(params) + 1))(z, *params)
        # H = [
        #    hessian(loss_fn, argnums=i)(z, *params)
        #    for i in range(len(params) + 1)
        # ]
        params_ = (z, *params)
        H = [
            torch.autograd.functional.hessian(
                lambda arg: loss_fn(*params_[:i], arg, *params_[i + 1 :]), arg
            )
            for (i, arg) in enumerate(params_)
        ]
        # H = torch.autograd.functional.hessian(loss_fn, (z, *params))
        # H = [H[i][i] for i in range(len(H))]
        Dp, Dpp = implicit_grads_2nd(
            k_fn, z, *params, Dg=g[0], Hg=H[0], Dzk_solve_fn=Dzk_solve_fn
        )
        Dpp = Dpp if len(params) != 1 else [Dpp]
        ret = [Dpp + H for (Dpp, H) in zip(Dpp, H[1:])]
        ret = [ret.detach() for ret in ret]
        return ret[0] if len(ret) == 1 else ret

    return f_fn, g_fn, h_fn
