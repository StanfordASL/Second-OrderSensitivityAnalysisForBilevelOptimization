import os, sys, unittest, pdb

import torch, numpy as np

import header

from implicit.interface import init

jaxm = init(dtype=np.float64, device="cpu")

import objs

X = jaxm.randn((100, 3))
Y = jaxm.randn((100, 5))


def generate_test(OPT, *params, name=""):
    def fn(self):
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

    # fn.__name__ = OPT.__class__.__name__
    fn.__name__ = name
    return fn


class DpzTest(unittest.TestCase):
    pass


names = ["LS", "CE", "LS_with_centers", "LS_with_diag", "CE_with_diag"]
OPTs = [
    objs.LS(),
    objs.CE(verbose=True, max_it=30, method="cvx"),
    objs.OPT_with_centers(objs.LS(), 2),
    objs.OPT_with_diag(objs.LS()),
    #objs.LS_with_diag(),
    objs.OPT_with_diag(
        objs.CE(verbose=True, max_it=30, method="cvx"),
    ),
]
param_list = [
    (1e-1,),
    (1e-1,),
    (jaxm.array([1e-1, 1e-1]),),
    (1e-1 * jaxm.ones(X.shape[-1] * Y.shape[-1]),),
    #(1e-1 * jaxm.ones(X.shape[-1]),),
    (1e-1 * jaxm.ones(X.shape[-1] * (Y.shape[-1] - 1)),),
]

for (OPT, params, name) in zip(OPTs, param_list, names):
    fn = generate_test(OPT, *params, name=name)
    setattr(DpzTest, "test_%s" % fn.__name__, fn)

if __name__ == "__main__":
    unittest.main(verbosity=2)
