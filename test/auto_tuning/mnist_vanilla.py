import os, pdb, sys, math, time, gzip, pickle

import torch
from tqdm import tqdm

import header

from implicit.opt import minimize_agd
from mnist import train, test


def loss_fn(W, Ztr, Ytr):
    pass


def main():
    device = torch.device("cuda")
    try:
        raise FileNotFoundError
        with gzip.open("data/mnist_weights.pkl.gz", "rb") as fp:
            nn = pickle.load(fp).to(device)
    except FileNotFoundError:
        nn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, (6, 6), 3),
            torch.nn.Tanh(),
            torch.nn.Conv2d(16, 16, (2, 2), 2),
            torch.nn.Tanh(),
            torch.nn.Conv2d(16, 16, (2, 2), 1),
            torch.nn.Tanh(),
            torch.nn.Conv2d(16, 16, (2, 2), 1),
            torch.nn.Tanh(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 4, 16 * 4),
            torch.nn.Tanh(),
            torch.nn.Linear(16 * 4, 10),
        ).to(device)

    Xtr, Xts = [
        torch.tensor(z, dtype=torch.float32, device=device).reshape(
            (-1, 1, 28, 28)
        )
        for z in [train["images"], test["images"]]
    ]
    Ytr, Yts = [
        torch.nn.functional.one_hot(
            torch.tensor(z, device=device, dtype=torch.long)
        )
        for z in [train["labels"], test["labels"]]
    ]

    def f_fn():
        r = torch.randint(Xtr.shape[0], size=(10 ** 4,))
        X, Y = Xtr[r, ...], Ytr[r, ...]
        return torch.nn.CrossEntropyLoss()(nn(X), torch.argmax(Y, -1))

    def acc_fn(X, Y):
        return torch.mean(
            (torch.argmax(nn(X), -1) == torch.argmax(Y, -1)).to(torch.float32)
        )

    opt = torch.optim.Adam(nn.parameters(), lr=1e-4)
    for it in tqdm(range(10 ** 5)):
        l = f_fn()
        opt.zero_grad()
        l.backward()
        opt.step()
        tqdm.write("%05d %9.4e" % (it, float(l)))
        if (it + 1) % (10 ** 2) == 0:
            print("-------------------------- %3.1f" % (acc_fn(Xts, Yts) * 1e2))
    with gzip.open("data/mnist_weights2.pkl.gz", "wb") as fp:
        pickle.dump(nn.to("cpu"), fp)

if __name__ == "__main__":
    main()
