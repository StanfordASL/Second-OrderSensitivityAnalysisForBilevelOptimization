import math, pdb, os, sys, time
from functools import reduce
from operator import mul

import matplotlib.pyplot as plt, torch

from .utils import t, topts, ss, fn_with_sol_cache
from .diff import JACOBIAN, HESSIAN, HESSIAN_DIAG, fwd_grad, grad

prod = lambda zs: reduce(mul, zs)

CHECK_GRADS = False


def ensure_list(a):
    return a if isinstance(a, list) or isinstance(a, tuple) else [a]


def implicit_grads_1st(
    k_fn,
    z,
    *params,
    Dg=None,
    Dzk=None,
    Dzk_solve_fn=None,
    full_output=False,
    jvp_vec=None
):
    zlen, plen = prod(z.shape), [prod(param.shape) for param in params]
    jvp_vec = ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec

    cache = dict()
    if Dg is not None:
        if Dzk_solve_fn is None:
            if Dzk is None:
                Dzk = JACOBIAN(lambda z: k_fn(z, *params), z)
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
        Dp = ensure_list(JACOBIAN(fn, *params))
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
        if jvp_vec is not None:
            for param in params:
                param.requires_grad = True
            f_ = k_fn(z.detach(), *params)
            Dp = [
                fwd_grad(f_, param, grad_inputs=jvp_vec)
                for (param, jvp_vec) in zip(params, jvp_vec)
            ]
            Dp = [Dp.reshape((zlen, 1)) for (Dp, plen) in zip(Dp, plen)]
            Dpk = Dp
        else:
            Dpk = ensure_list(
                JACOBIAN(lambda *params: k_fn(z, *params), *params)
            )
            Dpk = [Dpk.reshape((zlen, plen)) for (Dpk, plen) in zip(Dpk, plen)]
        if Dzk_solve_fn is None:
            if Dzk is None:
                Dzk = JACOBIAN(lambda z: k_fn(z, *params), z)
            F = torch.lu(Dzk.reshape((zlen, zlen)))
            Dpz = [-torch.lu_solve(Dpk, *F) for Dpk in Dpk]
        else:
            Dpz = [-Dzk_solve_fn(z, *params, Dpk, T=False) for Dpk in Dpk]
        if jvp_vec is not None:
            Dpz_shaped = [Dpz.reshape(z.shape) for Dpz in Dpz]
        else:
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
    k_fn,
    z,
    *params,
    Dg=None,
    Hg=None,
    Dzk=None,
    Dzzk=None,
    Dzk_solve_fn=None,
    jvp_vec=None
):
    """
    compute implicit gradients of z wrt params (or p)
    Args:
        k_fn: the implicit function k(z, *params) = 0, dk_dz(z, *params) != 0
        z: variable which we differentiate
        *params: variables wrt which we differentiate
    """
    zlen, plen = prod(z.shape), [prod(param.shape) for param in params]
    jvp_vec = ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec
    if jvp_vec is not None:
        assert Dg is not None

    # compute 2nd implicit gradients
    if Dg is not None:
        assert Dg.numel() == zlen
        assert Hg is None or Hg.numel() == zlen ** 2

        Dg_ = Dg.reshape((zlen, 1))
        Hg_ = Hg.reshape((zlen, zlen)) if Hg is not None else Hg

        # compute the left hand vector in the VJP
        if Dzk_solve_fn is None:
            Dzk = JACOBIAN(lambda z: k_fn(z, *params), z).reshape((zlen, zlen))
            FT = torch.lu(t(Dzk.reshape((zlen, zlen))))
            v = -torch.lu_solve(Dg_.reshape((zlen, 1)), *FT).reshape((zlen, 1))
        else:
            v = -Dzk_solve_fn(z, *params, Dg_.reshape((zlen, 1)), T=True)
        v = v.detach()
        fn = lambda z, *params: torch.sum(
            v.reshape(zlen) * k_fn(z, *params).reshape(zlen)
        )

        if jvp_vec is not None:
            for param in params:
                param.requires_grad = True
            z.requires_grad = True

            Dpz_jvp = ensure_list(
                implicit_grads_1st(
                    k_fn,
                    z,
                    *params,
                    Dzk=Dzk,
                    Dzk_solve_fn=Dzk_solve_fn,
                    jvp_vec=jvp_vec,
                )
            )
            Dpz_jvp = [Dpz_jvp.reshape(-1).detach() for Dpz_jvp in Dpz_jvp]

            # compute the 2nd order derivatives consisting of 4 terms
            # term 1 ##############################
            # Dpp1 = HESSIAN(lambda *params: fn(z, *params), *params)
            g_ = grad(fn(z, *params), params, create_graph=True)
            Dpp1 = [
                fwd_grad(g_, param, grad_inputs=jvp_vec).reshape(-1)
                for (g_, param, jvp_vec) in zip(g_, params, jvp_vec)
            ]

            # term 2 ##############################
            # temp = JACOBIAN(
            #    lambda z: JACOBIAN(
            #        lambda *params: fn(z, *params), *params, create_graph=True
            #    ),
            #    z,
            # )
            # temp = [temp] if len(params) == 1 else temp
            # temp = [
            #    temp.reshape((plen, zlen)) for (temp, plen) in zip(temp, plen)
            # ]
            # Dpp2 = [
            #    (temp @ Dpz).reshape((plen, plen))
            #    for (temp, Dpz, plen) in zip(temp, Dpz, plen)
            # ]
            g_ = grad(fn(z, *params), params, create_graph=True)
            Dpp2 = [
                fwd_grad(g_, z, grad_inputs=Dpz_jvp.reshape(z.shape)).reshape(
                    -1
                )
                for (g_, Dpz_jvp) in zip(g_, Dpz_jvp)
            ]

            # term 3 ##############################
            # Dpp3 = [t(Dpp2) for Dpp2 in Dpp2]
            g_ = grad(fn(z, *params), z, create_graph=True)
            g_ = [
                fwd_grad(g_, param, grad_inputs=jvp_vec)
                for (param, jvp_vec) in zip(params, jvp_vec)
            ]
            Dpp3 = [
                ensure_list(
                    implicit_grads_1st(
                        k_fn,
                        z,
                        *params,
                        Dzk=Dzk,
                        Dzk_solve_fn=Dzk_solve_fn,
                        Dg=g_,
                    )
                )[i].reshape(-1)
                for (i, g_) in enumerate(g_)
            ]

            # term 4 ##############################
            # Dzz = HESSIAN(lambda z: fn(z, *params), z).reshape((zlen, zlen))
            # if Hg is not None:
            #    Dpp4 = [t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
            # else:
            #    Dpp4 = [t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]
            g_ = grad(fn(z, *params), z, create_graph=True)
            g_ = [
                fwd_grad(g_, z, grad_inputs=Dpz_jvp.reshape(z.shape))
                for Dpz_jvp in Dpz_jvp
            ]
            if Hg is not None:
                g_ = [
                    g_.reshape(zlen) + Hg_ @ Dpz_jvp.reshape(zlen)
                    for (g_, Dpz_jvp) in zip(g_, Dpz_jvp)
                ]
            Dpp4 = [
                ensure_list(
                    implicit_grads_1st(
                        k_fn,
                        z,
                        *params,
                        Dzk=Dzk,
                        Dzk_solve_fn=Dzk_solve_fn,
                        Dg=g_,
                    )
                )[i].reshape(plen)
                for ((i, g_), plen) in zip(enumerate(g_), plen)
            ]
            Dp = [
                Dg_.reshape((1, zlen)) @ Dpz_jvp.reshape(zlen)
                for Dpz_jvp in Dpz_jvp
            ]
            Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]

            # return the results
            Dp_shaped = [Dp.reshape(()) for Dp in Dp]
            Dpp_shaped = [
                Dpp.reshape(param.shape)
                for (Dpp, param) in zip(Dpp, params)
            ]
        else:
            # compute the full first order 1st gradients
            Dpz = implicit_grads_1st(
                k_fn,
                z,
                *params,
                Dzk=Dzk,
                Dzk_solve_fn=Dzk_solve_fn,
            )
            Dpz = [Dpz] if len(params) == 1 else Dpz
            Dpz = [
                Dpz.reshape((zlen, plen)).detach()
                for (Dpz, plen) in zip(Dpz, plen)
            ]

            # compute the 2nd order derivatives consisting of 4 terms
            Dpp1 = HESSIAN(lambda *params: fn(z, *params), *params)
            Dpp1 = [[Dpp1]] if len(params) == 1 else Dpp1
            Dpp1 = [
                Dpp1[i].reshape((plen, plen))
                for (Dpp1, i, plen) in zip(Dpp1, range(len(params)), plen)
            ]
            temp = JACOBIAN(
                lambda z: JACOBIAN(
                    lambda *params: fn(z, *params), *params, create_graph=True
                ),
                z,
            )
            temp = [temp] if len(params) == 1 else temp
            temp = [
                temp.reshape((plen, zlen)) for (temp, plen) in zip(temp, plen)
            ]
            Dpp2 = [
                (temp @ Dpz).reshape((plen, plen))
                for (temp, Dpz, plen) in zip(temp, Dpz, plen)
            ]
            Dpp3 = [t(Dpp2) for Dpp2 in Dpp2]
            Dzz = HESSIAN(lambda z: fn(z, *params), z).reshape((zlen, zlen))
            if Hg is not None:
                Dpp4 = [t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
            else:
                Dpp4 = [t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]
            Dp = [Dg_.reshape((1, zlen)) @ Dpz for Dpz in Dpz]
            Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]

            # return the results
            Dp_shaped = [
                Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)
            ]
            Dpp_shaped = [
                Dpp.reshape(param.shape + param.shape)
                for (Dpp, param) in zip(Dpp, params)
            ]
            if CHECK_GRADS:
                print("Checking 2nd order grad")
                Dpz, Dppz = implicit_grads_2nd(k_fn, z, *params)
                Dpz = [Dpz] if len(params) == 1 else Dpz
                Dppz = [Dppz] if len(params) == 1 else Dppz

                Df = [
                    Dpz.reshape((zlen, plen)) for (Dpz, plen) in zip(Dpz, plen)
                ]
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
        Dpz, cache = implicit_grads_1st(
            k_fn,
            z,
            *params,
            Dzk=Dzk,
            Dzk_solve_fn=Dzk_solve_fn,
            full_output=True,
        )
        Dpz = ensure_list(Dpz)
        Dpz = [Dpz.reshape(zlen, plen) for (Dpz, plen) in zip(Dpz, plen)]

        # compute derivatives
        if Dzzk is None:
            Hk = HESSIAN_DIAG(k_fn, z, *params)
            Dzzk, Dppk = Hk[0], Hk[1:]
        else:
            Dppk = HESSIAN_DIAG(lambda *params: k_fn(z, *params), *params)
        Dzpk = JACOBIAN(
            lambda *params: JACOBIAN(
                lambda z: k_fn(z, *params), z, create_graph=True
            ),
            params,
        )
        Dppk = [
            Dppk.reshape((zlen, plen, plen)) for (Dppk, plen) in zip(Dppk, plen)
        ]
        Dzzk = Dzzk.reshape((zlen, zlen, zlen))
        Dzpk = [
            Dzpk.reshape((zlen, zlen, plen)) for (Dzpk, plen) in zip(Dzpk, plen)
        ]
        Dpzk = [Dzpk.transpose(-1, -2) for Dzpk in Dzpk]

        # solve the IFT equation
        lhs = [
            Dppk
            + Dpzk @ Dpz[None, ...]
            + t(Dpz)[None, ...] @ Dzpk
            + (t(Dpz)[None, ...] @ Dzzk) @ Dpz[None, ...]
            for (Dpz, Dzpk, Dpzk, Dppk) in zip(Dpz, Dzpk, Dpzk, Dppk)
        ]
        if Dzk_solve_fn is not None:
            Dppz = [
                -Dzk_solve_fn(
                    z, *params, lhs.reshape((zlen, plen * plen)), T=False
                ).reshape((zlen, plen, plen))
                for (lhs, plen) in zip(lhs, plen)
            ]
        else:
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
        g = JACOBIAN(loss_fn, (z, *params))
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
        g = JACOBIAN(loss_fn, (z, *params))
        params_ = (z, *params)
        H = [
            HESSIAN(
                lambda arg: loss_fn(*params_[:i], arg, *params_[i + 1 :]), arg
            )
            for (i, arg) in enumerate(params_)
        ]
        Dp, Dpp = implicit_grads_2nd(
            k_fn, z, *params, Dg=g[0], Hg=H[0], Dzk_solve_fn=Dzk_solve_fn
        )
        Dpp = Dpp if len(params) != 1 else [Dpp]
        ret = [Dpp + H for (Dpp, H) in zip(Dpp, H[1:])]
        ret = [ret.detach() for ret in ret]
        return ret[0] if len(ret) == 1 else ret

    return f_fn, g_fn, h_fn
