import functools, operator
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


# fwd mode grads ###############################################################
def prod(x):
    return functools.reduce(operator.mul, x, 1)


def write_list_at(xs, idxs, els):
    assert len(idxs) == len(els)
    k, xs = 0, [x for x in xs]
    for idx in idxs:
        xs[idx] = els[k]
        k += 1
    return xs


def fwd_grad(ys, xs, grad_inputs=None, **kwargs):
    # we only support a single input, otherwise undesirable accumulation occurs
    if isinstance(xs, list) or isinstance(xs, tuple):
        assert len(xs) == 1

    # select only the outputs which have a gradient path/graph
    ys_ = [ys] if not (isinstance(ys, list) or isinstance(ys, tuple)) else ys
    idxs = [i for (i, y) in enumerate(ys_) if y.grad_fn is not None]
    ys_select = [ys_[i] for i in idxs]
    if len(ys_) == 0:
        return [torch.zeros_like(y) for y in ys_]

    # perform the first step of forward mode emulation in reverse mode AD
    vs_ = [torch.ones_like(y, requires_grad=True) for y in ys_select]
    gs_ = torch.autograd.grad(
        ys_select, xs, grad_outputs=vs_, create_graph=True, allow_unused=True
    )

    # perform the second step of reverse mode emulation in reverse mode AD
    if grad_inputs is not None:
        # apply the JVP if necessary
        gs_ = torch.autograd.grad(
            gs_, vs_, grad_outputs=grad_inputs, allow_unused=True
        )
    else:
        gs_ = torch.autograd.grad(gs_, vs_, allow_unused=True)

    # fill in the unused outputs with zeros (rather than None)
    gs_ = [
        torch.zeros_like(y) if g is None else g
        for (g, y) in zip(gs_, ys_select)
    ]

    # fill in the outputs which did not have gradient paths
    ret = write_list_at([torch.zeros_like(y) for y in ys_], idxs, gs_)

    # return a single output if the output was a single element
    return ret if isinstance(ys, list) or isinstance(ys, tuple) else ret[0]

def grad(y, xs, **kwargs):
    gs = torch.autograd.grad(y, xs, **kwargs)
    return gs[0] if not (isinstance(xs, list) or isinstance(xs, tuple)) else gs
