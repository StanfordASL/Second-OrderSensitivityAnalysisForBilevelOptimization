import math, pdb, os, sys, time
from functools import partial

import numpy as np, torch

jaxm = None
jax, jnp, jsp, jrandom = None, None, None, None


def init(device=None, dtype=None, seed=None):
    if dtype is not None:
        os.environ["JAX_ENABLE_X64"] = str(
            dtype == np.float64 or dtype == torch.float64
        )
    if device is not None:
        if (
            isinstance(device, str)
            and (device.lower() == "cuda" or device.lower() == "gpu")
        ) or device == torch.device("cuda"):
            os.environ["JAX_PLATFORM_NAME"] = "GPU"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(False)
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        else:
            os.environ["JAX_PLATFORM_NAME"] = "CPU"
            keys = [
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
            ]
            for k in keys:
                if k in os.environ:
                    del os.environ[k]

    global jax, jnp, jsp, jrandom
    import jax, jax.numpy as jnp, jax.scipy as jsp, jax.random as jrandom

    # binding main derivatives and jit
    jaxm = jnp
    jaxm.grad, jaxm.jacobian = jax.grad, jax.jacobian
    jaxm.hessian = jax.hessian
    jaxm.jit = jax.jit

    # binding random numbers
    global key
    key = jrandom.PRNGKey(int(time.time()) if seed is None else seed)

    def random_fn(fn):
        def jaxm_fn(*args, **kwargs):
            global key
            key1, key2 = jrandom.split(key)
            ret = fn(key1, *args, **kwargs)
            key = key2
            return ret

        return jaxm_fn

    jaxm.randn = random_fn(jrandom.normal)
    jaxm.rand = random_fn(jrandom.uniform)
    jaxm.randint = random_fn(
        lambda key, low, high, size: jrandom.randint(key, size, low, high)
    )

    # jaxm.randn = partial(jrandom.normal, key)
    # jaxm.rand = partial(jrandom.uniform, key)
    # jaxm.randint = lambda low, high, size: jrandom.randint(key, size, low, high)

    # LA factorizations and solves
    jaxm.linalg.cholesky = jsp.linalg.cho_factor
    jaxm.linalg.cholesky_solve = jsp.linalg.cho_solve
    jaxm.linalg.lu_factor = jsp.linalg.lu_factor
    jaxm.linalg.lu_solve = jsp.linalg.lu_solve

    # some utility bindings
    jaxm.norm = jnp.linalg.norm
    jaxm.softmax = jax.nn.softmax
    jaxm.cat = jnp.concatenate
    jaxm.t = lambda x: jaxm.swapaxes(x, -1, -2)
    jaxm.nn = jax.nn

    # module bindings
    jaxm.jax = jax
    jaxm.lax = jax.lax
    jaxm.xla = jax.xla
    jaxm.scipy = jsp
    jaxm.random = jrandom

    return jaxm


if __name__ == "__main__":
    load_jax(device="cpu", dtype=torch.float32)
    d = jax.randn((10,))
    print(d.dtype, d.device())

    pdb.set_trace()
