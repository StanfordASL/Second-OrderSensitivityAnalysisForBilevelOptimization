import torch, pdb

JACOBIAN = torch.autograd.functional.jacobian


def kron(A, B):
    return torch.kron(A.contiguous(), B.contiguous())


op = lambda A, B: (A @ B.reshape((B.shape[0], -1))).reshape(
    (A.shape[0],) + B.shape[-2:]
)


torch.set_default_dtype(torch.float64)

z, p = 5, 2
z_, p_ = torch.randn(z), torch.randn(p)
Dpz = torch.randn((z, p))

############# d(A B) / dA = B[None, :, :].T @ dA = (I (x) Dpz.T) dA ############
def convoluted_matrix(p):
    global z
    self = convoluted_matrix
    self.Dzz = self.Dzz if hasattr(self, "Dzz") else torch.randn((z, z))
    z_ = torch.cat([p, torch.ones(1), p.flip(-1)])
    Dzk = (
        (self.Dzz @ z_.reshape((-1, 1)))
        @ (z_.reshape((1, -1)) / (torch.sin(torch.sum(p) ** 2) ** 2 + 1e-3))
    ) @ self.Dzz
    return Dzk


def f(p_):
    Dzk = convoluted_matrix(p_)
    return Dzk @ Dpz


p_ = torch.randn(p)
D = JACOBIAN(f, p_)
Dzpk = JACOBIAN(convoluted_matrix, p_)
err = (Dpz.T[None, :, :] @ Dzpk - D).norm()
print("err = %9.4e" % err)
assert err <= 1e-7
Dzpk_ = torch.cat([z for z in Dzpk], -2)
ret = torch.stack((kron(torch.eye(z), Dpz.T) @ Dzpk_).split(p), 0)
err = (ret - D).norm()
print("err = %9.4e" % err)
assert err <= 1e-7
################################################################################

##### d(A B) / dB = (A @ dB.reshape((B1, B2 * B3))).reshape((A1, B2, B3)) ######
################# = (A (x) I) dB ###############################################
def convoluted_matrix(p):
    global z
    self = convoluted_matrix
    self.Dzz = self.Dzz if hasattr(self, "Dzz") else torch.randn((z, z))
    z_ = torch.cat([p, torch.ones(1), p.flip(-1)])
    Dzk = (
        (self.Dzz @ z_.reshape((-1, 1)))
        @ (z_.reshape((1, -1)) / (torch.sin(torch.sum(p) ** 2) ** 2 + 1e-3))
    ) @ self.Dzz
    return Dzk


def f(p_):
    Dzk = convoluted_matrix(p_)
    return Dpz.T @ Dzk


p_ = torch.randn(p)
D = JACOBIAN(f, p_)
Dzpk = JACOBIAN(convoluted_matrix, p_)
#err = (
#    (Dpz.T @ Dzpk.reshape((Dzpk.shape[0], -1))).reshape(
#        (Dpz.T.shape[0],) + Dzpk.shape[-2:]
#    )
#    - D
#).norm()
err = (op(Dpz.T, Dzpk) - D).norm()
print("err = %9.4e" % err)
assert err <= 1e-7
Dzpk_ = torch.cat([z for z in Dzpk], -2)
ret = torch.stack((kron(Dpz.T, torch.eye(z)) @ Dzpk_).split(z), 0)
err = (ret - D).norm()
print("err = %9.4e" % err)
assert err <= 1e-7
################################################################################

################################################################################
r = torch.randn((3, 2))
A = torch.randn((2, 4))
ret1 = kron(r @ A, torch.eye(5))
ret2 = kron(r, torch.eye(5)) @ kron(A, torch.eye(5))
err = (ret1 - ret2).norm()
print("err = %9.4e" % err)
assert err <= 1e-7
################################################################################

################################################################################
ret1 = op(Dpz.T, Dzpk)
ret2 = torch.einsum("ik,kjl->ijl", Dpz.T, Dzpk)
################################################################################

pdb.set_trace()
