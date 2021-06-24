import unittest, pdb, time

import torch

import header
from implicit import implicit_grads_2nd
from auto_tuning import objs

CE = objs.CE()
X = torch.randn((100, 3))
Y = torch.randn((100, 5))
lam = 1e-3
p = torch.randn((3, 6))
W = CE.solve(X @ p, Y, lam)

VERBOSE = False

# we test here 2nd order implicit gradients
class DpzTest(unittest.TestCase):
    def test_shape_and_val(self):
        k_fn = lambda W, p: CE.grad(W, X @ p, Y, lam)
        Dzk_solve_fn = lambda W, p, rhs, T=False: CE.Dzk_solve(
            W, X @ p, Y, lam, rhs, T=T
        )
        Dpz, Dppz = implicit_grads_2nd(k_fn, W, p, Dzk_solve_fn=Dzk_solve_fn)
        self.assertEqual(Dpz.shape, (W.shape + p.shape))
        self.assertEqual(Dppz.shape, (W.shape + p.shape + p.shape))

        Dpz2, Dppz2 = implicit_grads_2nd(k_fn, W, p)
        self.assertEqual(Dpz2.shape, (W.shape + p.shape))
        self.assertEqual(Dppz2.shape, (W.shape + p.shape + p.shape))

        err_Dpz = torch.norm(Dpz - Dpz2)
        err_Dppz = torch.norm(Dppz - Dppz2)
        self.assertTrue(err_Dpz < 1e-5)
        self.assertTrue(err_Dppz < 1e-5)

    def test_shape_jvp_without_Dzk_solve(self, Dzk_solve_fn=None):
        if VERBOSE:
            print()

        k_fn = lambda W, p: CE.grad(W, X @ p, Y, lam)
        jvp_vec = torch.randn(p.shape)
        v = torch.randn(W.shape)

        t_ = time.time()
        Dpz1, Dppz1 = implicit_grads_2nd(
            k_fn, W, p, Dg=v, Dzk_solve_fn=Dzk_solve_fn
        )
        if VERBOSE:
            print("Elapsed %9.4e" % (time.time() - t_))

        t_ = time.time()
        Dpz2, Dppz2 = implicit_grads_2nd(
            k_fn, W, p, Dg=v, Dzk_solve_fn=Dzk_solve_fn, jvp_vec=jvp_vec
        )
        if VERBOSE:
            print("Elapsed %9.4e" % (time.time() - t_))

        self.assertEqual(Dpz2.shape, ())
        self.assertEqual(Dppz2.shape, p.shape)

        eps = 1e-5
        err_Dpz = torch.norm(
            Dpz1.reshape(p.numel()) @ jvp_vec.reshape(p.numel()) - Dpz2
        )
        err_Dppz = torch.norm(
            Dppz1.reshape((p.numel(), p.numel())) @ jvp_vec.reshape(p.numel())
            - Dppz2.reshape(p.numel())
        )
        self.assertTrue(err_Dpz < eps)
        self.assertTrue(err_Dppz < eps)

    def test_shape_jvp_with_Dzk_solve(self):
        Dzk_solve_fn = lambda W, p, rhs, T=False: CE.Dzk_solve(
            W, X @ p, Y, lam, rhs, T=T
        )
        self.test_shape_jvp_without_Dzk_solve(Dzk_solve_fn=Dzk_solve_fn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
