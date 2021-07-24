import sys, os, gzip, pickle, pdb
import numpy as np, matplotlib.pyplot as plt
import torch

fname = "data/dataset_cache.pkl.gz"
try:
    with gzip.open(fname, "rb") as fp:
        X, Y = pickle.load(fp)
except FileNotFoundError:
    import pandas as pd
    df = pd.read_csv("data/survival", delimiter="\t")

    lived = (df["Survival"] > (5 * 365)) & (df["Death"] == 0.0)
    died = (
        (df["Survival"] < (5 * 365))
        & (df["Death"] == 1.0)
        & (df["Survival"] > 0.0)
    )
    lived, died = df[lived], df[died]
    lived["PatientID"] = lived["PatientID"].str.upper()
    died["PatientID"] = died["PatientID"].str.upper()

    mirna = pd.read_csv("data/mirna", delimiter=" ")
    exp = pd.read_csv("data/exp", delimiter=" ")

    exp = exp.filter(regex="\.01$")
    mirna = mirna.filter(regex="\.01$")

    exp.columns = [".".join(z[:-1]) for z in exp.columns.str.split(".")]
    mirna.columns = [".".join(z[:-1]) for z in mirna.columns.str.split(".")]

    lived_keys = (
        pd.Index(lived["PatientID"])
        .intersection(exp.columns)
        .intersection(mirna.columns)
    )
    died_keys = (
        pd.Index(died["PatientID"])
        .intersection(exp.columns)
        .intersection(mirna.columns)
    )
    lived_data = exp[lived_keys]
    died_data = exp[died_keys]
    X = np.concatenate([lived_data.values, died_data.values], -1).T
    Y = np.concatenate([np.ones(len(lived_keys)), np.zeros(len(died_keys))])

    X, Y = torch.as_tensor(X), torch.as_tensor(Y)
    #A = torch.randn((X.shape[1], 1000), dtype=X.dtype) / X.shape[0] ** 2
    #X = X @ A
    norm = torch.std(X, -2)[None, :]
    X = X / torch.where(norm < 1e-3, 1.0, norm)

    X, Y = X.cpu().numpy(), Y.cpu().numpy()

    with gzip.open(fname, "wb") as fp:
        pickle.dump((X, Y), fp)

if __name__ == "__main__":
    pass
