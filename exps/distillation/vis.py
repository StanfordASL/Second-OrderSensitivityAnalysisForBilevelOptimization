import gzip, pickle, pdb, math, os, sys
import matplotlib.pyplot as plt, numpy as np, torch

from feat_map import feat_map
feat_map_ = feat_map

def main(Xs=None, save=False, fname=None):
    self = main

    if fname is None:
        fname = "data/data_ls_rand.pkl.gz"
    if Xs is None:
        with gzip.open(fname, "rb") as fp:
            Xs = pickle.load(fp)
    elif isinstance(Xs, torch.Tensor):
        Xs = Xs.cpu().detach().numpy()

    feat_map = lambda X: feat_map_(torch.tensor(X)).numpy()

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
                #np.flip(Xs[i, :].reshape((28, 28)), 0),
                Xs[i, :].reshape((28, 28)),
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
        fname = os.path.splitext(os.path.split(fname)[1])[0]
        fname = os.path.splitext(fname)[0]
        plt.savefig("figs/%s.png" % fname, dpi=200)
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
    main(fname="data/data_ls_rand.pkl.gz", save=True)
    main(fname="data/data_ls_mean.pkl.gz", save=True)
    main(fname="data/data_ce_rand.pkl.gz", save=True)
    main(fname="data/data_ce_mean.pkl.gz", save=True)

    #Xs = main(save=True)
    #main(Xs)

    #pdb.set_trace()
