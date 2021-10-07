import pdb
import torch
from tqdm import tqdm

A = torch.randn((5, 5))
#A = torch.diag(torch.randn((5,)))
A = A + A.T
U, S, V = torch.svd(A)

def fn(bet):
    return torch.linalg.norm(A - torch.diag(bet), 2)

bet = torch.nn.Parameter(torch.randn(A.shape[-1]))
optim = torch.optim.Adam([bet], lr=1e-2)

def closure():
    l = fn(bet)
    bet.grad = torch.autograd.grad(l, bet)[0]
    return l

for i in tqdm(range(10 ** 4)):
    optim.step(closure)
    l = fn(bet)
    tqdm.write("%9.4e" % float(fn(bet)))
pdb.set_trace()