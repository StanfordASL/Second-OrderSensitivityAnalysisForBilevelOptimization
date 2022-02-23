import unittest, pdb

import torch, numpy as np

import header
from implicit.interface import init

jaxm = init(dtype=np.float64, device="cpu")
from implicit.implicit import implicit_jacobian
import objs

OPT = objs.CE()
X = jaxm.randn((100, 3))
Y = jaxm.randn((100, 5))
lam = 1e-3
p = jaxm.randn((3, 6))
W = OPT.solve(X @ p, Y, lam)

# we test here 1st order implicit gradients
class DpzTest(unittest.TestCase):
    def test_shape(self):
        k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
        Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(
            W, X @ p, Y, lam, rhs=rhs, T=T
        )
        optimizations = {
            "Dzk_solve_fn": Dzk_solve_fn,
        }
        Dpz = implicit_jacobian(k_fn, W, p, optimizations=optimizations)
        Dpz2 = implicit_jacobian(k_fn, W, p)
        self.assertEqual(Dpz.shape, (W.shape + p.shape))
        self.assertEqual(Dpz2.shape, (W.shape + p.shape))

        err_Dpz = jaxm.norm(Dpz - Dpz2)
        eps = 1e-5
        self.assertTrue(err_Dpz < eps)

    def test_shape_jvp_with_Dzk_solve(self):
       k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
       Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(
           W, X @ p, Y, lam, rhs=rhs, T=T
       )
       jvp_vec = jaxm.randn(p.shape)
       optimizations = {
           "Dzk_solve_fn": Dzk_solve_fn,
       }
       Dpz1 = implicit_jacobian(k_fn, W, p, optimizations=optimizations)
       Dpz2 = implicit_jacobian(
           k_fn, W, p, jvp_vec=jvp_vec, optimizations=optimizations
       )
       self.assertEqual(Dpz2.shape, W.shape)
       eps = 1e-5
       err = jaxm.norm(
           Dpz1.reshape((W.size, p.size)) @ jvp_vec.reshape(-1)
           - Dpz2.reshape(-1)
       )
       self.assertTrue(err < eps)

    def test_shape_jvp_without_Dzk_solve(self):
       k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
       jvp_vec = jaxm.randn(p.shape)
       Dpz1 = implicit_jacobian(k_fn, W, p)
       Dpz2 = implicit_jacobian(k_fn, W, p, jvp_vec=jvp_vec)
       self.assertEqual(Dpz2.shape, W.shape)
       eps = 1e-5
       err = jaxm.norm(
           Dpz1.reshape((W.size, p.size)) @ jvp_vec.reshape(-1)
           - Dpz2.reshape(-1)
       )
       self.assertTrue(err < eps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
