import os, sys, unittest, pdb

import torch, numpy as np

import header

from implicit.interface import init

jaxm = init(dtype=np.float64, device="cpu")

from objs import LS, CE, OPT_conv

N = 100
X = jaxm.randn((N, 3, 6, 5)).reshape((N, -1))
Y = jaxm.randn((N, 10))

OPT = OPT_conv(LS(), 3, 5, 3, 2)
param = OPT.generate_parameter()
params = (param,)

class DpzTest(unittest.TestCase):
    def test_conv(self):
        W = OPT.solve(X, Y, *params)
        g = OPT.grad(W, X, Y, *params)
        self.assertTrue(jaxm.norm(g) < 1e-5)

        # gradient quality
        g_ = jaxm.grad(OPT.fval)(W, X, Y, *params)
        err = jaxm.norm(g_ - g)
        self.assertTrue(err < 1e-9)

        # hessian quality
        H = OPT.hess(W, X, Y, *params)
        err = jaxm.norm(jaxm.jacobian(OPT.grad)(W, X, Y, *params) - H)
        self.assertTrue(err < 1e-9)

        # Dzk_solve
        H = OPT.hess(W, X, Y, *params).reshape((W.size, W.size))
        rhs = jaxm.randn((W.size, 3))
        err = jaxm.norm(
            jaxm.linalg.solve(H, rhs)
            - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=False)
        )
        self.assertTrue(err < 1e-9)
        err = jaxm.norm(
            jaxm.linalg.solve(jaxm.t(H), rhs)
            - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=True)
        )
        self.assertTrue(err < 1e-9)

if __name__ == "__main__":
    unittest.main(verbosity=2)
