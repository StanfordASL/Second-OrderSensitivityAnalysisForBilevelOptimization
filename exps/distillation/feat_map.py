import sys, os, pdb

import torch

def feat_map(X):
    X = torch.sigmoid(X)
    return X
    # return torch.cat([X[..., 0:1] ** 0, X], -1)

