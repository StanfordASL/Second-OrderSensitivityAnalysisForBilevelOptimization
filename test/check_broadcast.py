import torch, sys, pdb

sys.path.append("auto_tuning")

from objs import CE
from implicit import implicit_grads_2nd

torch.set_default_dtype(torch.double)

n, e, d = 10, 5, 3
X, Y, lam = torch.randn((n, e)), torch.randn((n, d)), 1e-1
ce = CE()


def kron(A, B):
    return torch.kron(A.contiguous(), B.contiguous())


def opt_fn(param):
    global X, ce, lam
    Z = torch.sin(param * X) / (torch.sum(param) ** 2 + 1e-7)
    return ce.solve(Z, Y, lam, method="", max_it=100, verbose=True)


def k_fn(W, param):
    global X, ce, lam
    Z = torch.sin(param * X) / (torch.sum(param) ** 2 + 1e-7)
    return ce.grad(W, Z, Y, lam)


param = torch.randn(e)

W = opt_fn(param)
k = k_fn(W, param)
print(k)

g, H = implicit_grads_2nd(k_fn, W, param)
H = H.reshape((W.numel(),) + (param.numel(),) * 2)
v = torch.randn(W.numel())
temp1 = kron(v.reshape((1, -1)), torch.eye(param.numel()))
H1 = kron(v.reshape((1, -1)), torch.eye(param.numel())) @ torch.cat(
    [z for z in H], -2
)
H2 = torch.sum(v[..., None, None] * H, -3)

g_, H3 = implicit_grads_2nd(k_fn, W, param, Dg=v)

pdb.set_trace()
