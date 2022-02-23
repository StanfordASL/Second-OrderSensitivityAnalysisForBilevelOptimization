import torch, numpy as np

from implicit.interface import init
jaxm = init(dtype=np.float64, device="cpu")

from implicit.utils import j2t, t2j

def jloss_fn(Yp, Yts):
    Yp, Yts = t2j(Yp), t2j(Yts)
    ret = jaxm.mean(-Yp[..., jaxm.argmax(Yts, -1)] + jaxm.nn.logsumexp(Yp, -1))
    return j2t(ret)

def tloss_fn(Yp, Yts):
    return torch.nn.CrossEntropyLoss()(Yp, torch.argmax(Yts, -1))

torch.set_default_dtype(torch.float64)

for i in range(1):
    n = 330
    Yp = 1e3 * torch.randn((n, 10))
    Yts = torch.nn.functional.one_hot(torch.randint(10, size=(n,)))
    print(tloss_fn(Yp, Yts))
    print(jloss_fn(Yp, Yts))
