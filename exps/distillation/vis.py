import gzip, pickle, pdb, math
import matplotlib.pyplot as plt, numpy as np, torch

# feat map #####################################################################
from feat_map import feat_map

feat_map_ = feat_map

def feat_map(X):
    return feat_map_(torch.tensor(X)).numpy()
################################################################################

def main(Xs=None, save=False):
    self = main

    if Xs is None:
        with gzip.open("data/data.pkl.gz", "rb") as fp:
            Xs = pickle.load(fp)
    elif isinstance(Xs, torch.Tensor):
        Xs = Xs.cpu().detach().numpy()


    pcs = []
    Xs = feat_map(Xs)
    vmin, vmax = np.min(Xs.reshape(-1)), np.max(Xs.reshape(-1))
    print("vmin = %5.3f, vmax = %5.3f" % (vmin, vmax))

    dim2 = 10
    dim1 = math.ceil(Xs.shape[0] / dim2)

    scale = 2
    fig = plt.figure(13123213452, figsize=(scale * dim2, scale * dim1))
    fig.clf()
    self.fig, self.axes = fig, fig.subplots(dim1, dim2)
    cmap = plt.get_cmap("binary")

    for i in range(10):
        #coo = (i // 4, i % 4)
        coo = (i // dim2, i % dim2)
        ax = self.axes.reshape((dim1, dim2))[coo[0], coo[1]]
        pcs.append(
            #ax.contourf(
            ax.imshow(
                np.flip(Xs[i, :].reshape((28, 28)), 0),
                #levels=50,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        )
        ax.set_title("%d" % i)
        ax.set_axis_off()
    idx = np.argmax([np.max(Xs[i, :]) for i in range(10)])
    if save:
        plt.tight_layout()
        plt.savefig("figs/digits.png", dpi=200)
    else:
        self.fig.colorbar(
            pcs[idx],
            ax=self.axes,
            location="right",
            boundaries=np.linspace(vmin, vmax, 50),
        )
    plt.draw_all()
    plt.pause(1e-1)
    return Xs


if __name__ == "__main__":
    Xs = main(save=True)
    #main(Xs)
    pdb.set_trace()
