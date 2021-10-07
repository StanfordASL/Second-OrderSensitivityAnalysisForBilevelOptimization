import sys, os, pdb, pickle, gzip, math

import torch, numpy as np, matplotlib.pyplot as plt

dirname = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(dirname, "..", ".."))
from implicit.interface import init

jaxm = init(dtype=np.float64, device="cpu")

from implicit.implicit import implicit_jacobian, implicit_hessian, generate_fns
from implicit.opt import minimize_agd, minimize_sqp, minimize_lbfgs

f_nw = jaxm.jit(lambda x, p: (x - (jaxm.exp(p * x) - 2)))
df_nw = jaxm.jit(lambda x, p: (1 - p * jaxm.exp(p * x)))

CONFIG = dict(a=1, b=100)


def visualize(p):
    x = jaxm.linspace(0, 3, 10 ** 5)
    plt.plot(x, x)
    plt.plot(x, jaxm.exp(p * x) - 2)
    plt.show()


def solve_exp(p):
    x = jaxm.array(1.0)
    for i in range(100):
        x = x - f_nw(x, p) / df_nw(x, p)
    return x


@jaxm.jit
def rosenbrock(z):
    a, b = CONFIG["a"], CONFIG["b"]
    x, y = z[0], z[1]
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def opt_fn(param):
    px, py = param
    x = solve_exp(px)
    y = py
    return jaxm.stack([x, y])


def k_fn(z, param):
    px, py = param
    x, y = z
    return jaxm.stack([f_nw(x, px), y - py])


def loss_fn(z, *params):
    return rosenbrock(z)


if __name__ == "__main__":
    param = jaxm.array([CONFIG["a"], CONFIG["a"] ** 2]) + 1e-1 * jaxm.randn(
        (2,)
    )
    z = opt_fn(param)
    k = k_fn(z, param)
    print(loss_fn(z, param))
    print(k)

    f_fn, g_fn, h_fn = generate_fns(loss_fn, opt_fn, k_fn)
    f, g, h = f_fn(param), g_fn(param), h_fn(param)

    g_fn_ = jaxm.grad(lambda param: loss_fn(opt_fn(param), param))
    h_fn_ = jaxm.hessian(lambda param: loss_fn(opt_fn(param), param))
    g_, h_ = g_fn_(param), h_fn_(param)
    # pdb.set_trace()

    ps = minimize_sqp(
        f_fn,
        g_fn,
        h_fn,
        param,
        verbose=True,
        reg0=1e-15,
        force_step=True,
        ls_pts_nb=1,
        max_it=300,
    )
    ps = minimize_agd(
        f_fn, g_fn, param, verbose=True, max_it=10 ** 3, ai=1e-1, af=1e-1
    )
    ps = minimize_lbfgs(
       f_fn, g_fn, param, verbose=True, lr=1e-1, max_it=10 ** 2
    )
    pdb.set_trace()
