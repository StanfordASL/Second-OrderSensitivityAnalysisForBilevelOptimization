import pdb, time, math, pickle, gzip

import torch
from tqdm import tqdm

def tv(A):
    assert A.ndim == 3
    device, dtype = A.device, A.dtype
    wx = torch.tensor([1, -1], device=device, dtype=dtype)[None, None, None, :] 
    wy = torch.tensor([1, -1], device=device, dtype=dtype)[None, None, :, None]
    A = A[..., None, :, :]
    lx = torch.sum(torch.nn.functional.conv2d(A, wx) ** 2, (-1, -2, -3))
    ly = torch.sum(torch.nn.functional.conv2d(A, wy) ** 2, (-1, -2, -3))
    l = lx + ly
    return l

def tv_l1(A):
    assert A.ndim == 3
    device, dtype = A.device, A.dtype
    wx = torch.tensor([1, -1], device=device, dtype=dtype)[None, None, None, :] 
    wy = torch.tensor([1, -1], device=device, dtype=dtype)[None, None, :, None]
    A = A[..., None, :, :]
    lx = torch.sum(torch.nn.functional.conv2d(A, wx).abs(), (-1, -2, -3))
    ly = torch.sum(torch.nn.functional.conv2d(A, wy).abs(), (-1, -2, -3))
    l = lx + ly
    return l


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    A = torch.nn.Parameter(torch.randn((1000, 28, 28), device="cuda"))
    l = tv(A)
    opt = torch.optim.Adam([A], lr=1e-2)

    loss_fn = lambda: tv(A).mean()
    for it in range(10 ** 4):
        l = loss_fn()
        opt.zero_grad()
        l.backward()
        opt.step()
        tqdm.write("%05d %9.4e" % (it, float(l)))
        if (it + 1) % 100 == 0:
            plt.figure(100)
            plt.imshow(A[0, ...].detach().cpu().numpy())
            plt.draw()
            plt.pause(1e-1)
    pdb.set_trace()
