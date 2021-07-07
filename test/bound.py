import pdb, sys, time, math

import matplotlib.pyplot as plt, numpy as np, sympy as sp
from sympy.utilities.lambdify import lambdify

sp.init_printing(unicode=True)

alf, bet, sig, eps, gam = sp.symbols("alf bet sig eps gam")
bound = bet * sig / (alf + eps) + (gam * sig + eps) / (alf + eps) / alf
exp = sp.ratsimp(sp.diff(bound, eps))
num, denom = sp.numer(exp), sp.denom(exp)

ret = sp.solve(num, eps)

fn = lambdify((alf, bet, sig, eps, gam), bound)
print("#" * 80)
print(bound)
print("#" * 80)
#eps1_fn = lambdify((alf, bet, sig, gam), ret[0])
#eps2_fn = lambdify((alf, bet, sig, gam), ret[1])

plt.figure()
x = np.logspace(-11, 3, 10**3)
for s in np.logspace(-6, 2, 10):
    alf, sig = s, 1e-1
    bet, gam = 1e1, 1e1
    y = [fn(alf, bet, sig, eps, gam) for eps in x]

    #eps1, eps2 = eps1_fn(alf, bet, sig, gam), eps2_fn(alf, bet, sig, gam)

    plt.loglog(x, y, label="s = %9.4e" % s)
    #if not math.isnan(eps1) and eps1 > 0.0:
    #    plt.scatter(eps1, fn(alf, bet, sig, eps1, gam))
    #if not math.isnan(eps2) and eps2 > 0.0:
    #    plt.scatter(eps2, fn(alf, bet, sig, eps2, gam))

plt.legend()
plt.show()

pdb.set_trace()
