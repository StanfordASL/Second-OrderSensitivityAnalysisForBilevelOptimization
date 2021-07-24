import unittest, pdb

import torch

import header
from implicit import implicit_grads_1st
from auto_tuning import objs

CE = objs.CE()
X = torch.randn((100, 3))
Y = torch.randn((100, 5))
lam = 1e-3
p = torch.randn((3, 6))
W = CE.solve(X @ p, Y, lam)

# we test here 1st order implicit gradients
class DpzTest(unittest.TestCase):
    def test_shape(self):
        k_fn = lambda W, p: CE.grad(W, X @ p, Y, lam)
        Dzk_solve_fn = lambda W, p, rhs, T=False: CE.Dzk_solve(
            W, X @ p, Y, lam, rhs, T=T
        )
        optimizations = {
            "Dzk_solve_fn": Dzk_solve_fn,
        }
        Dpz = implicit_grads_1st(k_fn, W, p, optimizations=optimizations)
        Dpz2 = implicit_grads_1st(k_fn, W, p)
        self.assertEqual(Dpz.shape, (W.shape + p.shape))
        self.assertEqual(Dpz2.shape, (W.shape + p.shape))

        err_Dpz = torch.norm(Dpz - Dpz2)
        eps = 1e-5
        self.assertTrue(err_Dpz < eps)

    def test_shape_jvp_with_Dzk_solve(self):
        k_fn = lambda W, p: CE.grad(W, X @ p, Y, lam)
        Dzk_solve_fn = lambda W, p, rhs, T=False: CE.Dzk_solve(
            W, X @ p, Y, lam, rhs, T=T
        )
        jvp_vec = torch.randn(p.shape)
        optimizations = {
            "Dzk_solve_fn": Dzk_solve_fn,
        }
        Dpz1 = implicit_grads_1st(k_fn, W, p, optimizations=optimizations)
        Dpz2 = implicit_grads_1st(
            k_fn, W, p, jvp_vec=jvp_vec, optimizations=optimizations
        )
        self.assertEqual(Dpz2.shape, W.shape)
        eps = 1e-5
        err = torch.norm(
            Dpz1.reshape((W.numel(), p.numel())) @ jvp_vec.reshape(-1)
            - Dpz2.reshape(-1)
        )
        self.assertTrue(err < eps)

    def test_shape_jvp_without_Dzk_solve(self):
        k_fn = lambda W, p: CE.grad(W, X @ p, Y, lam)
        jvp_vec = torch.randn(p.shape)
        Dpz1 = implicit_grads_1st(k_fn, W, p)
        Dpz2 = implicit_grads_1st(k_fn, W, p, jvp_vec=jvp_vec)
        self.assertEqual(Dpz2.shape, W.shape)
        eps = 1e-5
        err = torch.norm(
            Dpz1.reshape((W.numel(), p.numel())) @ jvp_vec.reshape(-1)
            - Dpz2.reshape(-1)
        )
        self.assertTrue(err < eps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
