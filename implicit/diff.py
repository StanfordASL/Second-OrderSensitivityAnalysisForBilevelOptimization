import torch

JACOBIAN = torch.autograd.functional.jacobian
HESSIAN_ = torch.autograd.functional.hessian


def HESSIAN(fn, *args_, **kwargs):
    args = args_
    f = fn(*args, **kwargs)
    n = f.numel()
    if n == 1:
        H = HESSIAN_(fn, *args, **kwargs)
        return H
    else:
        Hs = [
            HESSIAN_(lambda *args: fn(*args, **kwargs).reshape(-1)[i], args)
            for i in range(n)
        ]
        Hs = [
            [
                torch.stack([Hs[k][i][j] for k in range(n)], 0).reshape(
                    f.shape + Hs[0][i][j].shape
                )
                for i in range(len(args))
            ]
            for j in range(len(args))
        ]
        if len(args) == 1:
            Hs = Hs[0][0]
        return Hs


def HESSIAN_DIAG(fn, *args, **kwargs):
    return [
        HESSIAN(lambda arg: fn(*args[:i], arg, *args[i + 1 :], **kwargs), arg)
        for (i, arg) in enumerate(args)
    ]


