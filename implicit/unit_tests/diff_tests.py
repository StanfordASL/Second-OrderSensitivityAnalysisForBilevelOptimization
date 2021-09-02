import unittest, pdb

import torch

import header
from implicit import diff

# we test here success/failure behavior and formatting, not actual AutoDiff
class FwdDiffTest(unittest.TestCase):
    def test_siso_format(self):
        x = torch.randn(1, requires_grad=True)
        y = torch.cat([x ** 2, x ** 0, x, x - x, x + x])
        g = diff.fwd_grad(y, x)
        self.assertFalse(isinstance(g, list) or isinstance(g, tuple))

    def test_mimo_fail(self):
        x1 = torch.randn(1, requires_grad=True)
        x2 = torch.randn(1, requires_grad=True)
        y = torch.cat([x1 ** 2, x2 ** 0, x2, x1 - x2, x1 * x2])
        with self.assertRaises(AssertionError):
            g = diff.fwd_grad(y, [x1, x2])

    def test_simo_succ(self):
        x = torch.randn(1, requires_grad=True)
        y1 = torch.cat([x ** 2, x ** 0, x, x - x, x * x])
        y2 = torch.cat([x ** 3, x ** 1, x, -x - x, 2 * x * x])
        g = diff.fwd_grad([y1, y2], x)
        self.assertTrue(isinstance(g, list) or isinstance(g, tuple))
        self.assertTrue(len(g) == 2)

    def test_jvp(self):
        x = torch.randn(6, requires_grad=True)
        Dyx = torch.randn((4, x.numel()))
        y = Dyx @ x
        jvp_vec = torch.randn(x.numel())
        eps = 1e-7

        J = diff.fwd_grad(y, x, grad_inputs=jvp_vec)
        self.assertTrue(torch.norm(J - Dyx @ jvp_vec) < eps)

        J = diff.fwd_grad(y, x, grad_inputs=[jvp_vec])
        self.assertTrue(torch.norm(J - Dyx @ jvp_vec) < eps)

        J = diff.fwd_grad(y, [x], grad_inputs=jvp_vec)
        self.assertTrue(torch.norm(J - Dyx @ jvp_vec) < eps)

        J = diff.fwd_grad(y, [x], grad_inputs=[jvp_vec])
        self.assertTrue(torch.norm(J - Dyx @ jvp_vec) < eps)

        J = diff.fwd_grad([y], [x], grad_inputs=jvp_vec)
        self.assertTrue(torch.norm(J[0] - Dyx @ jvp_vec) < eps)

    def test_simo_jvp(self):
        x = torch.randn(6, requires_grad=True)
        Dyx1, Dyx2 = torch.randn((4, x.numel())), torch.randn((5, x.numel()))
        y1, y2 = Dyx1 @ x, Dyx2 @ x
        jvp_vec = torch.randn(x.numel())
        eps = 1e-7

        J = diff.fwd_grad([y1, y2], x, grad_inputs=jvp_vec)
        self.assertTrue(torch.norm(J[0] - Dyx1 @ jvp_vec) < eps)
        self.assertTrue(torch.norm(J[1] - Dyx2 @ jvp_vec) < eps)

if __name__ == "__main__":
    unittest.main(verbosity=2)
