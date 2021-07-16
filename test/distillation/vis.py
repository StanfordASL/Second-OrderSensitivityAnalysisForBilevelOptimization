import gzip, pickle, pdb
import matplotlib.pyplot as plt, numpy as np, torch


def main(Xs=None):
    self = main

    if Xs is None:
        with gzip.open("data.pkl.gz", "rb") as fp:
            Xs = pickle.load(fp)
    elif isinstance(Xs, torch.Tensor):
        Xs = Xs.cpu().detach().numpy()


    pcs = []
    Xs = np.abs(Xs)
    vmin, vmax = np.min(Xs.reshape(-1)), np.max(Xs.reshape(-1))
    print("vmin = %5.3f, vmax = %5.3f" % (vmin, vmax))

    fig = plt.figure(13123213452)
    fig.clf()
    self.fig, self.axes = fig, fig.subplots(3, 4)

    for i in range(10):
        coo = (i // 4, i % 4)
        ax = self.axes[coo[0], coo[1]]
        pcs.append(
            ax.contourf(
                np.flip(np.abs(Xs[i, :]).reshape((28, 28)), 0),
                levels=50,
                vmin=vmin,
                vmax=vmax,
            )
        )
        ax.set_title("%d" % i)
    idx = np.argmax([np.max(Xs[i, :]) for i in range(10)])
    self.fig.colorbar(
        pcs[idx],
        ax=self.axes,
        location="right",
        boundaries=np.linspace(vmin, vmax, 50),
    )
    # self.fig.tight_layout()
    plt.draw_all()
    plt.pause(1e-1)
    return Xs


if __name__ == "__main__":
    Xs = main()
    main(Xs)
    pdb.set_trace()
