import pdb, time, math
from functools import partial

import torch
import numpy as np
import scipy.sparse.linalg as spla

from .cg_batch import cg_batch


def asLinearOperator(fn, n, device=None, dtype=None):
    return spla.LinearOperator(
        (n, n),
        matvec=lambda x: fn(torch.as_tensor(x, device=device, dtype=dtype))
        .cpu()
        .detach()
        .numpy(),
    )


def solve_spla(
    spla_solve_fn, A_fn, rhs, x0=None, M_fn=None, tol=1e-5, max_it=None
):
    n, device, dtype = rhs.shape[-2], rhs.device, rhs.dtype
    A = asLinearOperator(A_fn, n, device=device, dtype=dtype)
    b = rhs.detach().cpu().numpy()
    if M_fn is not None:
        M = asLinearOperaotr(M_fn, n, device=device, dtype=dtype)
    else:
        M = None
    x0 = x0.detach().cpu().numpy() if x0 is not None else None

    x_prev = None

    def callback_fn(*args):
        x = args[0]
        nonlocal x_prev
        if x_prev is not None:
            # print(np.linalg.norm(x_prev - x) / np.linalg.norm(x))
            print(np.linalg.norm(A(x).reshape(-1) - b.reshape(-1)))
        x_prev = np.copy(x)

    kwargs = dict(callback_type="x") if spla_solve_fn == spla.gmres else dict()
    x, info = spla_solve_fn(
        A,
        b,
        x0=x0,
        M=M,
        tol=tol,
        maxiter=max_it,
        callback=callback_fn,
        **kwargs
    )
    pdb.set_trace()
    return x


solve_cg = partial(solve_spla, spla.cg)
solve_gmres = partial(solve_spla, spla.gmres)


def solve_cg_torch(A_fn, rhs, x0=None, M_fn=None, tol=1e-5, max_it=None):
    x, info = cg_batch(
        A_fn, rhs, M_bmm=M_fn, X0=x0, rtol=tol, maxiter=max_it, verbose=True
    )
    return x
