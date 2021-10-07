import header
from implicit.interface import init

jaxm = init()


def scale_down(X, size=2):
    kernel = jaxm.ones((1, 1, size, size)) / (size ** 2)
    Z = jaxm.lax.conv(X.reshape((-1, 1, 28, 28)), kernel, (size, size), "VALID")
    Z = Z.reshape((X.shape[0], -1))
    return Z
