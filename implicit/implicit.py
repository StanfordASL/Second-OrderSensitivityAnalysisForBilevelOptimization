import math, pdb, os, sys, time
from functools import reduce
from operator import mul
from typing import Mapping
from copy import copy


import matplotlib.pyplot as plt, torch

from .interface import init

jaxm = init()

from .utils import fn_with_sol_cache
from .diff import JACOBIAN, HESSIAN, HESSIAN_DIAG
from .inverse import solve_gmres, solve_cg, solve_cg_torch

prod = lambda zs: reduce(mul, zs) if zs != () else 1

CHECK_GRADS = False


def dealloc_dict(d):
    # boggles my mind why that would be necessaryu
    # but it absolutely prevents memory leaks
    keys = copy(list(d.keys()))
    for k in keys:
        del d[k]


def generate_default_Dzk_solve_fn(optimizations: Mapping, k_fn):
    def Dzk_solve_fn(z, *params, rhs=None, T=False):
        zlen = z.size
        if optimizations.get("Dzk", None) is None:
            optimizations["Dzk"] = JACOBIAN(k_fn)(z, *params)
        Dzk = optimizations["Dzk"]
        if T == True:
            if optimizations.get("FT", None) is None:
                optimizations["FT"] = jaxm.linalg.lu_factor(
                    jaxm.t(Dzk.reshape((zlen, zlen)))
                )
            return jaxm.linalg.lu_solve(optimizations["FT"], rhs)
        else:
            if optimizations.get("F", None) is None:
                optimizations["F"] = jaxm.linalg.lu_factor(
                    jaxm.t(Dzk.reshape((zlen, zlen)))
                )
            return jaxm.linalg.lu_solve(optimizations["F"], rhs)

    optimizations["Dzk_solve_fn"] = Dzk_solve_fn
    return


def ensure_list(a):
    return a if isinstance(a, list) or isinstance(a, tuple) else [a]


def implicit_jacobian(
    k_fn,
    z,
    *params,
    Dg=None,
    jvp_vec=None,
    matrix_free_inverse: bool = False,
    full_output=False,
    optimizations: Mapping = {},
):
    optimizations = {k: v for (k, v) in optimizations.items()}  # local copy
    zlen, plen = prod(z.shape), [prod(param.shape) for param in params]
    jvp_vec = ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        generate_default_Dzk_solve_fn(optimizations, k_fn)
    #############################################################

    if Dg is not None:
        if matrix_free_inverse:
            A_fn = lambda x: JACOBIAN(
                lambda z: jaxm.sum(k_fn(z, *params).reshape(-1) * x.reshape(-1))
            )(z).reshape(x.shape)
            raise NotImplementedError
            v = -solve_gmres(A_fn, Dg.reshape((zlen, 1)), max_it=300)
        else:
            Dzk_solve_fn = optimizations["Dzk_solve_fn"]
            v = -Dzk_solve_fn(z, *params, rhs=Dg.reshape((zlen, 1)), T=True)
        fn = lambda *params: jaxm.sum(
            v.reshape(zlen) * k_fn(z, *params).reshape(zlen)
        )
        Dp = JACOBIAN(fn, argnums=range(len(params)))(*params)
        Dp_shaped = [Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)]
        ret = Dp_shaped[0] if len(params) == 1 else Dp_shaped

        if CHECK_GRADS:
            ##^
            print("Checking 1st order grad")
            Dpz_ = implicit_jacobian(
                k_fn, z, *params, optimizations=optimizations
            )
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
                jaxm.norm(ret_ - Dp_shaped) / (jaxm.norm(Dp_shaped) + 1e-7)
                for (ret_, Dp_shaped) in zip(ret_, Dp_shaped)
            ]
            try:
                assert all(err < 1e-7 for err in errs)
            except:
                print("Gradient 1st order check failed")
                pdb.set_trace()
            ##$
    else:
        if jvp_vec is not None:
            fn = lambda *params: k_fn(z, *params)
            Dp = ensure_list(jaxm.jvp(fn, params, tuple(jvp_vec))[1])
            Dp = [Dp.reshape((zlen, 1)) for (Dp, plen) in zip(Dp, plen)]
            Dpk = Dp
        else:
            Dpk = JACOBIAN(
                lambda *params: k_fn(z, *params), argnums=range(len(params))
            )(*params)
            Dpk = [Dpk.reshape((zlen, plen)) for (Dpk, plen) in zip(Dpk, plen)]

        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        Dpz = [-Dzk_solve_fn(z, *params, rhs=Dpk, T=False) for Dpk in Dpk]

        if jvp_vec is not None:
            Dpz_shaped = [Dpz.reshape(z.shape) for Dpz in Dpz]
        else:
            Dpz_shaped = [
                Dpz.reshape(z.shape + param.shape)
                for (Dpz, param) in zip(Dpz, params)
            ]
        ret = Dpz_shaped if len(params) != 1 else Dpz_shaped[0]
    if full_output:
        return (ret, optimizations)
    else:
        dealloc_dict(optimizations)
        return ret


def implicit_hessian(
    k_fn,
    z,
    *params,
    Dg=None,
    Hg=None,
    jvp_vec=None,
    optimizations: Mapping = {},
):
    """
    compute implicit gradients of z wrt params (or p)
    Args:
        k_fn: the implicit function k(z, *params) = 0, dk_dz(z, *params) != 0
        z: variable which we differentiate
        *params: variables wrt which we differentiate
    """
    optimizations = {k: v for (k, v) in optimizations.items()}  # local copy
    zlen, plen = prod(z.shape), [prod(param.shape) for param in params]
    jvp_vec = ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec
    if jvp_vec is not None:
        assert Dg is not None

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        generate_default_Dzk_solve_fn(optimizations, k_fn)
    #############################################################

    # compute 2nd implicit gradients
    if Dg is not None:
        assert Dg.size == zlen
        assert Hg is None or Hg.size == zlen ** 2

        Dg_ = Dg.reshape((zlen, 1))
        Hg_ = Hg.reshape((zlen, zlen)) if Hg is not None else Hg

        # compute the left hand vector in the VJP
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        v = -Dzk_solve_fn(z, *params, rhs=Dg_.reshape((zlen, 1)), T=True)
        fn = lambda z, *params: jaxm.sum(
                v.reshape(zlen) * k_fn(z, *params).reshape(zlen)
            )

        if jvp_vec is not None:
            Dpz_jvp = ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    jvp_vec=jvp_vec,
                    optimizations=optimizations,
                )
            )
            Dpz_jvp = [Dpz_jvp.reshape(-1) for Dpz_jvp in Dpz_jvp]

            # compute the 2nd order derivatives consisting of 4 terms
            # term 1 ##############################
            # Dpp1 = HESSIAN_DIAG(lambda *params: fn(z, *params), *params)
            # g_ = grad(fn(z, *params), params, create_graph=True)
            # Dpp1 = [
            #    fwd_grad(g_, param, grad_inputs=jvp_vec).reshape(plen)
            #    for (g_, param, jvp_vec) in zip(g_, params, jvp_vec)
            # ]
            dfn_params = jaxm.grad(
                lambda *params: fn(z, *params), argnums=range(len(params))
            )
            Dpp1 = ensure_list(jaxm.jvp(dfn_params, params, tuple(jvp_vec))[1])
            Dpp1 = [Dpp1.reshape(plen) for (Dpp1, plen) in zip(Dpp1, plen)]

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
            # g_ = grad(fn(z, *params), params, create_graph=True)
            # Dpp2 = [
            #    fwd_grad(g_, z, grad_inputs=Dpz_jvp.reshape(z.shape)).reshape(
            #        -1
            #    )
            #    for (g_, Dpz_jvp) in zip(g_, Dpz_jvp)
            # ]
            Dpp2 = [
                jaxm.jvp(
                    lambda z: jaxm.grad(fn, argnums=i + 1)(z, *params),
                    (z,),
                    (Dpz_jvp.reshape(z.shape),),
                )[1].reshape(plen)
                for (i, (Dpz_jvp, plen)) in enumerate(zip(Dpz_jvp, plen))
            ]

            # term 3 ##############################
            # Dpp3 = [jaxm.t(Dpp2) for Dpp2 in Dpp2]
            # g_ = grad(fn(z, *params), z, create_graph=True)
            # g_ = [
            #    fwd_grad(g_, param, grad_inputs=jvp_vec)
            #    for (param, jvp_vec) in zip(params, jvp_vec)
            # ]
            g_ = ensure_list(
                jaxm.jvp(
                    lambda *params: jaxm.grad(fn)(z, *params),
                    params,
                    tuple(jvp_vec),
                )[1]
            )
            Dpp3 = [
                ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        Dg=g_,
                        optimizations=optimizations,
                    )
                )[i].reshape(-1)
                for (i, g_) in enumerate(g_)
            ]

            # term 4 ##############################
            # Dzz = HESSIAN(lambda z: fn(z, *params), z).reshape((zlen, zlen))
            # if Hg is not None:
            #    Dpp4 = [jaxm.t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
            # else:
            #    Dpp4 = [jaxm.t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]

            # g_ = grad(fn(z, *params), z, create_graph=True)
            # g_ = [
            #    fwd_grad(g_, z, grad_inputs=Dpz_jvp.reshape(z.shape))
            #    for Dpz_jvp in Dpz_jvp
            # ]
            g_ = [
                jaxm.jvp(
                    lambda z: jaxm.grad(fn)(z, *params),
                    (z,),
                    (Dpz_jvp.reshape(z.shape),),
                )[1]
                for (i, Dpz_jvp) in enumerate(Dpz_jvp)
            ]
            if Hg is not None:
                g_ = [
                    g_.reshape(zlen) + Hg_ @ Dpz_jvp.reshape(zlen)
                    for (g_, Dpz_jvp) in zip(g_, Dpz_jvp)
                ]
            Dpp4 = [
                ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        Dg=g_,
                        optimizations=optimizations,
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
                Dpp.reshape(param.shape) for (Dpp, param) in zip(Dpp, params)
            ]
        else:
            # compute the full first order 1st gradients
            Dpz = ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    optimizations=optimizations,
                )
            )
            Dpz = [Dpz.reshape((zlen, plen)) for (Dpz, plen) in zip(Dpz, plen)]

            # compute the 2nd order derivatives consisting of 4 terms
            Dpp1 = HESSIAN_DIAG(lambda *params: fn(z, *params))(*params)
            Dpp1 = [
                Dpp1.reshape((plen, plen)) for (Dpp1, plen) in zip(Dpp1, plen)
            ]

            # temp = JACOBIAN(
            #    lambda z: JACOBIAN(
            #        lambda *params: fn(z, *params), params, create_graph=True
            #    ),
            #    z,
            # )
            temp = JACOBIAN(
                lambda *params: JACOBIAN(fn)(z, *params),
                argnums=range(len(params)),
            )(*params)
            temp = [
                jaxm.t(temp.reshape((zlen, plen)))
                for (temp, plen) in zip(temp, plen)
            ]
            Dpp2 = [
                (temp @ Dpz).reshape((plen, plen))
                for (temp, Dpz, plen) in zip(temp, Dpz, plen)
            ]
            Dpp3 = [jaxm.t(Dpp2) for Dpp2 in Dpp2]
            Dzz = HESSIAN(lambda z: fn(z, *params))(z).reshape((zlen, zlen))
            if Hg is not None:
                Dpp4 = [jaxm.t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
            else:
                Dpp4 = [jaxm.t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]
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
                ##^
                print("Checking 2nd order grad")
                Dpz, Dppz = implicit_hessian(
                    k_fn, z, *params, optimizations=optimizations
                )
                Dpz = [Dpz] if len(params) == 1 else Dpz
                Dppz = [Dppz] if len(params) == 1 else Dppz

                Df = [
                    Dpz.reshape((zlen, plen)) for (Dpz, plen) in zip(Dpz, plen)
                ]
                H1 = [jaxm.t(Df) @ Hg.reshape((zlen, zlen)) @ Df for Df in Df]

                Hf = [
                    Dppz.reshape((zlen, plen, plen))
                    for (Dppz, plen) in zip(Dppz, plen)
                ]
                H2 = [jaxm.sum(Dg.reshape((zlen, 1, 1)) * Hf, -3) for Hf in Hf]

                H = [
                    (H1 + H2).reshape(param.shape + param.shape)
                    for (H1, H2, param) in zip(H1, H2, params)
                ]
                errs = [
                    jaxm.norm(Dpp_shaped - H) / jaxm.norm(Dpp_shaped)
                    for (Dpp_shaped, H) in zip(Dpp_shaped, H)
                ]
                try:
                    assert all(err < 1e-7 for err in errs)
                except AssertionError:
                    print("Gradient 2nd order check failed")
                    pdb.set_trace()
                ##$
        dealloc_dict(optimizations)
        return (
            (Dp_shaped[0], Dpp_shaped[0])
            if len(params) == 1
            else (Dp_shaped, Dpp_shaped)
        )
    else:
        Dpz, optimizations = implicit_jacobian(
            k_fn,
            z,
            *params,
            full_output=True,
            optimizations=optimizations,
        )
        Dpz = ensure_list(Dpz)
        Dpz = [Dpz.reshape(zlen, plen) for (Dpz, plen) in zip(Dpz, plen)]

        # compute derivatives
        if optimizations.get("Dzzk", None) is None:
            Hk = HESSIAN_DIAG(k_fn)(z, *params)
            Dzzk, Dppk = Hk[0], Hk[1:]
            optimizations["Dzzk"] = Dzzk
        else:
            Dppk = HESSIAN_DIAG(lambda *params: k_fn(z, *params))(*params)
        Dzpk = JACOBIAN(
            lambda *params: JACOBIAN(k_fn)(z, *params),
            argnums=range(len(params)),
        )(*params)
        Dppk = [
            Dppk.reshape((zlen, plen, plen)) for (Dppk, plen) in zip(Dppk, plen)
        ]
        Dzzk = Dzzk.reshape((zlen, zlen, zlen))
        Dzpk = [
            Dzpk.reshape((zlen, zlen, plen)) for (Dzpk, plen) in zip(Dzpk, plen)
        ]
        Dpzk = [jaxm.t(Dzpk) for Dzpk in Dzpk]

        # solve the IFT equation
        lhs = [
            Dppk
            + Dpzk @ Dpz[None, ...]
            + jaxm.t(Dpz)[None, ...] @ Dzpk
            + (jaxm.t(Dpz)[None, ...] @ Dzzk) @ Dpz[None, ...]
            for (Dpz, Dzpk, Dpzk, Dppk) in zip(Dpz, Dzpk, Dpzk, Dppk)
        ]
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        Dppz = [
            -Dzk_solve_fn(
                z, *params, rhs=lhs.reshape((zlen, plen * plen)), T=False
            ).reshape((zlen, plen, plen))
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
        dealloc_dict(optimizations)
        return (
            (Dpz_shaped[0], Dppz_shaped[0])
            if len(params) == 1
            else (Dpz_shaped, Dppz_shaped)
        )


def detach_args(*args):
    return tuple(
        np.array(arg) if isinstance(arg, jaxm.DeviceArray) else arg
        for arg in args
    )


def generate_fns(
    loss_fn, opt_fn, k_fn, normalize_grad=False, optimizations: Mapping = {}
):
    sol_cache = dict()
    opt_fn_ = lambda *args, **kwargs: opt_fn(*args, **kwargs)
    optimizations = {k: v for (k, v) in optimizations.items()}

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def f_fn(z, *params):
        return loss_fn(z, *params)

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def g_fn(z, *params):
        g = JACOBIAN(loss_fn, argnums=range(len(params) + 1))(z, *params)
        Dp = implicit_jacobian(
            k_fn, z, *params, Dg=g[0], optimizations=optimizations
        )
        Dp = Dp if len(params) != 1 else [Dp]
        # opts = dict(device=z.device, dtype=z.dtype)
        # Dp = [
        #    jaxm.zeros(param.shape, **opts) for param in params
        # ]
        ret = [Dp + g for (Dp, g) in zip(Dp, g[1:])]
        if normalize_grad:
            ret = [(z / (jaxm.norm(z) + 1e-7)) for z in ret]
        return ret[0] if len(ret) == 1 else ret

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def h_fn(z, *params):
        g = JACOBIAN(loss_fn, argnums=range(len(params) + 1))(z, *params)

        if optimizations.get("Hz_fn", None) is None:
            optimizations["Hz_fn"] = lambda z, *params: jaxm.hessian(loss_fn)(
                z, *params
            )
        Hz_fn = optimizations["Hz_fn"]
        Hz = Hz_fn(z, *params)
        H = [Hz] + HESSIAN_DIAG(lambda *params: loss_fn(z, *params))(*params)

        Dp, Dpp = implicit_hessian(
            k_fn,
            z,
            *params,
            Dg=g[0],
            Hg=H[0],
            optimizations=optimizations,
        )
        Dpp = Dpp if len(params) != 1 else [Dpp]
        ret = [Dpp + H for (Dpp, H) in zip(Dpp, H[1:])]
        return ret[0] if len(ret) == 1 else ret

    return (f_fn, g_fn, h_fn)
