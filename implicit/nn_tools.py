import pdb, time
from collections import OrderedDict as odict
from functools import partial

import torch


# utilities ####################################################################
def nn_all_params(nn):
    return torch.cat([param.detach().reshape(-1) for param in nn.parameters()])


# main forward network via functional interface ################################
def linear(self, input, *args, **kwargs):
    return torch.nn.functional.linear(input, *args, **kwargs)


def conv(fn, self, input, *args, **kwargs):
    return fn(
        input,
        *args,
        **kwargs,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups,
    )


conv1d = partial(conv, torch.nn.functional.conv1d)
conv2d = partial(conv, torch.nn.functional.conv2d)
conv3d = partial(conv, torch.nn.functional.conv3d)


def batch_norm(fn, self, input, *args, **kwargs):
    return torch.nn.functional.batch_norm(
        input,
        self.running_mean,
        self.running_var,
        *args,
        **kwargs,
        momentum=self.momentum,
        eps=self.eps,
    )


NN_MAP = {
    torch.nn.Linear: linear,
    torch.nn.Conv1d: conv1d,
    torch.nn.Conv2d: conv2d,
    torch.nn.Conv3d: conv3d,
    torch.nn.BatchNorm1d: batch_norm,
    torch.nn.BatchNorm2d: batch_norm,
    torch.nn.BatchNorm3d: batch_norm,
}

def nn_forward(nn, X, params, debug=False):
    assert isinstance(nn, torch.nn.Sequential)
    layers = torch.split(
        params, [param.detach().numel() for param in nn.state_dict().values()]
    )
    layers_list, k = [], 0
    for mod in list(nn.modules())[1:]:
        params = list(mod.parameters())
        layer_params = [
            z.reshape(param.shape)
            for (z, param) in zip(layers[k : k + len(params)], params)
        ]
        layers_list.append(layer_params)
        k += len(params)
    Z = X
    for (layer, mod) in zip(layers_list, list(nn.modules())[1:]):
        if debug:
            pdb.set_trace()
        Z = NN_MAP[type(mod)](mod, Z, *layer) if type(mod) in NN_MAP else mod(Z)
    return Z


# alternative forward network via functional interface #########################
NN_FN_MAP = {
    torch.nn.Flatten: lambda x: x.reshape((x.shape[0], -1)),
    torch.nn.ReLU: torch.nn.functional.relu,
    torch.nn.Tanh: torch.tanh,
    torch.nn.Softmax: torch.nn.functional.softmax,
    torch.nn.Linear: torch.nn.functional.linear,
    torch.nn.Conv1d: torch.nn.functional.conv1d,
    torch.nn.Conv2d: torch.nn.functional.conv2d,
    torch.nn.Conv3d: torch.nn.functional.conv3d,
    torch.nn.BatchNorm1d: torch.nn.functional.batch_norm,
    torch.nn.BatchNorm2d: torch.nn.functional.batch_norm,
    torch.nn.BatchNorm3d: torch.nn.functional.batch_norm,
}

NN_ARGS_KWARGS_MAP = {
    torch.nn.Softmax: lambda x: ([], dict(dim=x.dim)),
    torch.nn.Conv1d: lambda x: (
        [],
        dict(
            stride=x.stride,
            padding=x.padding,
            dilation=x.dilation,
            groups=x.groups,
        ),
    ),
    torch.nn.Conv2d: lambda x: (
        [],
        dict(
            stride=x.stride,
            padding=x.padding,
            dilation=x.dilation,
            groups=x.groups,
        ),
    ),
    torch.nn.Conv3d: lambda x: (
        [],
        dict(
            stride=x.stride,
            padding=x.padding,
            dilation=x.dilation,
            groups=x.groups,
        ),
    ),
    torch.nn.BatchNorm1d: lambda x: (
        [x.running_mean, x.running_var],
        dict(momentum=x.momentum, eps=x.eps),
    ),
    torch.nn.BatchNorm2d: lambda x: (
        [x.running_mean, x.running_var],
        dict(momentum=x.momentum, eps=x.eps),
    ),
    torch.nn.BatchNorm3d: lambda x: (
        [x.running_mean, x.running_var],
        dict(momentum=x.momentum, eps=x.eps),
    ),
}

def nn_structure(nn):
    assert isinstance(nn, torch.nn.Sequential)
    numels = [param.numel() for param in nn.parameters()]
    shapes = [param.shape for param in nn.parameters()]
    mods = list(nn.modules())[1:]
    mod_types = [type(z) for z in mods]
    layer_args_kwargs = [
        NN_ARGS_KWARGS_MAP.get(type(mod), lambda x: ([], dict()))(mod)
        for mod in mods
    ]
    layer_param_nbs = [len(list(mod.parameters())) for mod in mods]
    return dict(
        numels=numels,
        shapes=shapes,
        layer_args_kwargs=layer_args_kwargs,
        mod_types=mod_types,
        layer_param_nbs=layer_param_nbs,
    )


#def nn_forward2(nn_struct, X, params, debug=False) -> torch.Tensor:
def nn_forward2(nn_struct, X, params, debug=False):
    params = torch.split(params, [numel for numel in nn_struct["numels"]])
    params = [
        param.reshape(shape)
        for (param, shape) in zip(params, nn_struct["shapes"])
    ]
    layers, k = [], 0
    for layer_param_nb in nn_struct["layer_param_nbs"]:
        layer_params = [z for z in params[k : k + layer_param_nb]]
        layers.append(layer_params)
        k += layer_param_nb
    Z = X
    for (layer, mod_type, args_kwargs) in zip(
        layers, nn_struct["mod_types"], nn_struct["layer_args_kwargs"]
    ):
        args, kwargs = args_kwargs
        Z = NN_FN_MAP[mod_type](Z, *layer, *args, **kwargs)
    return Z



# testing ######################################################################
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import include_implicit
    from implicit.diff import grad, fwd_grad

    def fn(nn, X, Y, params):
        Yp = nn_forward(nn, X, params)
        return torch.nn.MSELoss()(Yp, Y)
        # return torch.nn.CrossEntropyLoss()(Yp, torch.argmax(Y, -1))

    in_dim, out_dim = 784, 10
    embed_dim = 32

    # X, Y = torch.randn((1000, in_dim)), torch.randn((1000, out_dim))

    in_dim, out_dim = 1, 1
    X = torch.linspace(-10, 10, 1000)[..., None]
    Y = torch.sin(X)

    nn = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, embed_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(embed_dim, embed_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(embed_dim, embed_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(embed_dim, out_dim),
        # torch.nn.Softmax(-1),
    )

    device = torch.device("cpu")
    params = nn_all_params(nn).to(device)
    X, Y = X.to(device), Y.to(device)
    print(params.shape)

    params.requires_grad = True

    t = time.perf_counter()
    g = grad(fn(nn, X, Y, params), params, create_graph=True)
    t = time.perf_counter() - t
    print("Elapsed: %9.4e" % t)

    r = torch.randn(params.shape).to(device)

    t = time.perf_counter()
    g2 = fwd_grad(g, params, grad_inputs=r)
    t = time.perf_counter() - t
    print("Elapsed: %9.4e" % t)

    t = time.perf_counter()
    g3 = grad(torch.sum(r * g), params)
    t = time.perf_counter() - t
    print("Elapsed: %9.4e" % t)

    pdb.set_trace()
    assert torch.norm(g) > 1e-5
    assert torch.norm(g2) > 1e-5
    assert torch.norm(g3) > 1e-5

    pdb.set_trace()

    # devices = torch.device("cpu"), torch.device("cuda")
    # ns = torch.logspace(1, 4, 10).to(torch.long)
    # ts = {device: [] for device in devices}
    # for device in devices:
    #    for i in tqdm(range(len(ns))):
    #        n = ns[i].to(device)
    #        X = torch.randn((n, in_dim)).to(device)
    #        Y = torch.randn((n, out_dim)).to(device)
    #        params = params.to(device)

    #        t = time.time()
    #        H = torch.autograd.functional.hessian(
    #            lambda params: fn(X, Y, params), params
    #        )
    #        H = H.cpu()
    #        t = time.time() - t

    #        ts[device].append(t)
    #        # print("Elapsed %9.4e" % t)

    # for device in devices:
    #    plt.loglog(ns.cpu(), ts[device], label=str(device))
    # plt.legend()
    # plt.ylabel("time (s)")
    # plt.xlabel("data size (1)")
    # plt.show()

    # pdb.set_trace()
