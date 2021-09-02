import os, sys, pdb, gzip, pickle, math

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from implicit import implicit_jacobian
from implicit.diff import JACOBIAN

torch.set_default_dtype(torch.float64)

# class MixedProblem:
#    def __init__(self, a=1.0, b=0.0):
#        self.a, self.b = torch.tensor(a), torch.tensor(b)
#
#    def obj(self, z):
#        return z**2 + self.a * z + self.b
#
#    def solve(self):
#        zs = torch.arange(2)
#        fs = torch.tensor([self.obj(zs[i]) for i in range(zs.numel())])
#        zopt = zs[torch.argmin(fs)]
#        nu = -(2 * zopt + self.a) / (2 * zopt - 1)
#        return zopt, nu
#
#    def k_fn(self, x, p):
#        z, nu = x[0], x[1]
#        a, b = p[0], p[1]
#        return torch.stack([nu * z + 2 * z + nu * (z - 1) + a, z * (z - 1)])
#
#    def derv(self):
#        z, nu = self.solve()
#        x, p = torch.stack([z, nu]), torch.stack([self.a, self.b])
#        k = self.k_fn(x, p)
#        assert k.norm() < 1e-6
#        Dzk = JACOBIAN(lambda x: self.k_fn(x, p), x)
#        Dpk = JACOBIAN(lambda a: self.k_fn(x, torch.stack([a, self.b])), self.a)
#        g = -torch.linalg.solve(Dzk, Dpk)
#        return g


#class MixedProblem:
#    def __init__(self, a=1.0, b=0.0):
#        self.a, self.b = torch.tensor(a), torch.tensor(b)
#
#    def obj(self, z):
#        return z ** 2 + self.a * z + self.b
#
#    def solve(self):
#        zs = torch.arange(2)
#        fs = torch.tensor([self.obj(zs[i]) for i in range(zs.numel())])
#        zopt = zs[torch.argmin(fs)]
#        if zopt > 0:
#            lam = 2 * zopt + self.a
#        else:
#            lam = -2 * zopt - self.a
#        return zopt, lam
#
#    def k_fn(self, x, p):
#        z, lam = x[0], x[1]
#        a, b = p[0], p[1]
#        if z > 0:
#            k1 = 2 * z - lam + a
#            k2 = lam * (-z + 1)
#        else:
#            k1 = 2 * z + lam + a
#            k2 = lam * z
#        return torch.stack([k1, k2])
#
#    def derv(self):
#        z, lam = self.solve()
#        x, p = torch.stack([z, lam]), torch.stack([self.a, self.b])
#        k = self.k_fn(x, p)
#        assert k.norm() < 1e-6
#        Dzk = JACOBIAN(lambda x: self.k_fn(x, p), x)
#        Dpk = JACOBIAN(lambda a: self.k_fn(x, torch.stack([a, self.b])), self.a)
#        Dzk.diagonal()[:] += 0.0
#        g = -torch.linalg.solve(Dzk, Dpk)
#        return g

class MixedProblem:
    def __init__(self, a=1.0, b=0.0):
        self.a, self.b = torch.tensor(a), torch.tensor(b)

    def obj(self, z):
        return z ** 2 + self.a * z + self.b

    def solve(self):
        zs = torch.arange(2)
        fs = torch.tensor([self.obj(zs[i]) for i in range(zs.numel())])
        zopt = zs[torch.argmin(fs)]
        nu = -2 * zopt - a
        return zopt, nu

    def k_fn(self, x, p):
        z, nu = x[0], x[1]
        a, b = p[0], p[1]
        k1 = 2 * z + nu + a
        if z > 0:
            k2 = nu * (z - 1)
        else:
            k2 = nu * z
        return torch.stack([k1, k2])

    def derv(self):
        z, nu = self.solve()
        x, p = torch.stack([z, nu]), torch.stack([self.a, self.b])
        k = self.k_fn(x, p)
        assert k.norm() < 1e-6
        Dzk = JACOBIAN(lambda x: self.k_fn(x, p), x)
        Dpk = JACOBIAN(lambda a: self.k_fn(x, torch.stack([a, self.b])), self.a)
        Dzk.diagonal()[:] += 0.0
        pdb.set_trace()
        g = -torch.linalg.solve(Dzk, Dpk)
        return g


a_list = torch.linspace(-2.1, 0.1, 10 ** 1)
lam_list, z_list, g_list = [], [], []
for a in a_list:
    prob = MixedProblem(a=a)
    z_list.append(prob.solve()[0])
    lam_list.append(prob.solve()[1])
    g_list.append(prob.derv()[0])
x_list = -a_list / 2
z_list, g_list = torch.tensor(z_list), torch.tensor(g_list)
lam_list = torch.tensor(lam_list)
plt.plot(a_list, lam_list)
plt.plot([-1, -1], [torch.min(lam_list), torch.max(lam_list)])
plt.show()
pdb.set_trace()
