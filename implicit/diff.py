import functools, operator, pdb
import torch
from tqdm import tqdm

JACOBIAN = torch.autograd.functional.jacobian
HESSIAN_ = torch.autograd.functional.hessian


def HESSIAN(fn, args_, **kwargs):
    args = args_
    args = (
        (args,)
        if not (isinstance(args, list) or isinstance(args, tuple))
        else tuple(args)
    )

    f = fn(*args)
    n = f.numel()
    if n == 1:
        H = HESSIAN_(fn, args, **kwargs)
        return H
    else:
        Hs = [
            HESSIAN_(lambda *args: fn(*args).reshape(-1)[i], args, **kwargs)
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
        # if len(args) == 1:
        #    Hs = Hs[0][0]
        return Hs


def HESSIAN_DIAG(fn, args, **kwargs):
    args = (
        (args,)
        if not (isinstance(args, list) or isinstance(args, tuple))
        else tuple(args)
    )
    ret = [
       HESSIAN(lambda arg: fn(*args[:i], arg, *args[i + 1 :]), arg, **kwargs)[
           0
       ][0]
       for (i, arg) in enumerate(args)
    ]

    #from tqdm import tqdm
    #ret = []
    #for i in tqdm(range(len(args))):
    #    arg = args[i]
    #    ret.append(
    #        HESSIAN(
    #            lambda arg: fn(*args[:i], arg, *args[i + 1 :]), arg, **kwargs
    #        )[0][0]
    #    )

    return ret


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


##^# torch #####################################################################
def prod(x):
    return functools.reduce(operator.mul, x, 1)


def write_list_at(xs, idxs, els):
    assert len(idxs) == len(els)
    k, xs = 0, [x for x in xs]
    for idx in idxs:
        xs[idx] = els[k]
        k += 1
    return xs


def reshape_linear(fs):
    if not (isinstance(fs, tuple) or isinstance(fs, list)):
        return [fs], {"size": 1, "nb": 0}
    else:
        ret = [reshape_linear(f) for f in fs]
        vals, trees = [el[0] for el in ret], [el[1] for el in ret]
        tree = {i: trees[i] for i in range(len(trees))}
        tree["size"] = sum([tree["size"] for tree in trees])
        tree["nb"] = len(trees)
        return sum(vals, []), tree


def reshape_struct(fs_flat, tree):
    if tree["nb"] == 0:
        return fs_flat[0]
    k = 0
    fs = [None for i in range(tree["nb"])]
    for i in range(tree["nb"]):
        if tree[i]["nb"] == 0:
            fs[i] = fs_flat[k]
        elif tree[i]["nb"] > 0:
            fs[i] = reshape_struct(fs_flat[k : k + tree[i]["size"]], tree[i])
        else:
            fs[i] = fs_flat[k : k + tree[i]["size"]]
        k += tree[i]["size"]
    return tuple(fs)


def torch_grad(
    fn,
    argnums=0,
    bdims=0,
    create_graph=False,
    retain_graph=False,
    verbose=False,
):
    def g_fn(*args, **kwargs):
        args = list(args)
        argnums_ = list(argnums) if hasattr(argnums, "__iter__") else [argnums]
        # req_grads = [
        #    torch.as_tensor(arg).requires_grad if i in argnums_ else False
        #    for (i, arg) in enumerate(args)
        # ]
        for i in range(len(args)):
            if i in argnums_:
                args[i] = torch.as_tensor(args[i])
                args[i].requires_grad = True
        gargs = [arg for (i, arg) in enumerate(args) if i in argnums_]
        fs = fn(*args, **kwargs)
        fs, tree = reshape_linear(fs)
        G = [None for _ in range(len(fs))]
        for j, f in enumerate(fs):
            f_org_shape = f.shape
            f = f.reshape(f.shape[:bdims] + (-1,))
            Js = [
                torch.zeros(
                    (f.shape[-1],) + garg.shape, dtype=f.dtype, device=f.device
                )
                for garg in gargs
            ]
            rng = tqdm(range(f.shape[-1])) if verbose else range(f.shape[-1])
            for i in rng:
                f_ = torch.sum(f[..., i])
                if f_.grad_fn is not None:
                    gs = torch.autograd.grad(
                        f_,
                        gargs,
                        create_graph=create_graph,
                        retain_graph=(
                            create_graph
                            or j < len(fs) - 1
                            or i < f.shape[-1] - 1
                            or retain_graph
                        ),
                        allow_unused=True,
                    )
                else:
                    gs = [None for garg in gargs]
                gs = [
                    g
                    if g is not None
                    else torch.zeros(
                        gargs[l].shape,
                        dtype=gargs[l].dtype,
                        device=gargs[l].device,
                    )
                    for (l, g) in enumerate(gs)
                ]
                for k in range(len(Js)):
                    Js[k][i, ...] = gs[k].reshape((-1,) + gs[k].shape)
            for (k, J) in enumerate(Js):
                lshp = f_org_shape[bdims:]
                bshp = gargs[k].shape[:bdims]
                rshp = gargs[k].shape[bdims:]
                Js[k] = (
                    J.reshape((prod(lshp), prod(bshp), prod(rshp)))
                    .transpose(-2, -3)
                    .reshape(bshp + lshp + rshp)
                )
            if len(Js) > 1 or hasattr(argnums, "__iter__"):
                G[j] = tuple(Js)
            else:
                G[j] = Js[0]
        # for i in range(len(args)):
        #    if i in argnums_:
        #        args[i].requires_grad = req_grads[i]

        ret = reshape_struct(G, tree)
        return ret

    return g_fn


def torch_hessian(fn, argnums=0, bdims=0, create_graph=False):
    g_fn = torch_grad(fn, argnums=argnums, bdims=bdims, create_graph=True)
    g2_fn = torch_grad(
        g_fn, argnums=argnums, bdims=bdims, create_graph=create_graph
    )
    return g2_fn


##$#############################################################################
