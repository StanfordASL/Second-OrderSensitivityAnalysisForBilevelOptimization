import torch, pdb, time, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

in_dim, out_dim = 784, 10
embed_dim = 32

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
    torch.nn.Softmax(-1),
)


def all_params(nn):
    return torch.cat(
        [nn.state_dict()[key].detach().reshape(-1) for key in nn.state_dict()]
    )


def eval_nn(X, params):
    layers = torch.split(params, [param.numel() for param in nn.parameters()])
    Z = X
    for i in range(len(layers) // 2):
        W, b = layers[2 * i].reshape((-1, Z.shape[-1])), layers[2 * i + 1]
        Z = Z @ W.T + b
        if i < (len(layers) // 2) - 1:
            Z = torch.nn.Tanh()(Z)
    return torch.nn.Softmax(-1)(Z)


X, Y = torch.randn((1000, in_dim)), torch.randn((1000, out_dim))


def fn(X, Y, params):
    Yp = eval_nn(X, params)
    return torch.nn.CrossEntropyLoss()(Yp, torch.argmax(Y, -1))


device = torch.device("cuda")
params = all_params(nn).to(device)
print(params.shape)
# eval_nn(X, 1e9 * params)
# l = fn(params)
# print(l)


devices = torch.device("cpu"), torch.device("cuda")
ns = torch.logspace(1, 4, 10).to(torch.long)
ts = {device: [] for device in devices}
for device in devices:
    for i in tqdm(range(len(ns))):
        n = ns[i].to(device)
        X = torch.randn((n, in_dim)).to(device)
        Y = torch.randn((n, out_dim)).to(device)
        params = params.to(device)

        t = time.time()
        H = torch.autograd.functional.hessian(
            lambda params: fn(X, Y, params), params
        )
        H = H.cpu()
        t = time.time() - t

        ts[device].append(t)
        # print("Elapsed %9.4e" % t)

for device in devices:
    plt.loglog(ns.cpu(), ts[device], label=str(device))
plt.legend()
plt.ylabel("time (s)")
plt.xlabel("data size (1)")
plt.show()

pdb.set_trace()
