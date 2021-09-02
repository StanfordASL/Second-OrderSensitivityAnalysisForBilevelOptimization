import functools, operator, pdb
import torch
from tqdm import tqdm

from .interface import init

jaxm = init()

JACOBIAN = jaxm.jacobian
HESSIAN = jaxm.hessian


def HESSIAN_DIAG(fn):
    def h_fn(*args, **kwargs):
        args = (
            (args,)
            if not (isinstance(args, list) or isinstance(args, tuple))
            else tuple(args)
        )
        ret = [
            jaxm.hessian(
                lambda arg: fn(*args[:i], arg, *args[i + 1 :], **kwargs)
            )(arg)
            for (i, arg) in enumerate(args)
        ]
        return ret

    return h_fn
