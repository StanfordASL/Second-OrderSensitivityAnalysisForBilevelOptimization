import pdb, pickle, os, sys, time
from pprint import pprint
import gzip, shutil

import numpy as np

dirname = os.path.abspath(os.path.dirname(__file__))
fnames = dict(
    trl=os.path.join(dirname, "data/train-labels-idx1-ubyte" + ".gz"),
    tri=os.path.join(dirname, "data/train-images-idx3-ubyte" + ".gz"),
    tsl=os.path.join(dirname, "data/t10k-labels-idx1-ubyte" + ".gz"),
    tsi=os.path.join(dirname, "data/t10k-images-idx3-ubyte" + ".gz"),
)
archive_fname = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "data/mnist_archive.pkl.gz"
)

try:
    with gzip.open(archive_fname, "rb") as fp:
        train, test = pickle.load(fp)
except FileNotFoundError:
    with gzip.open(fnames["trl"], "rb") as fp:
        train_labels = np.frombuffer(fp.read(), dtype=np.uint8)[8:]
    with gzip.open(fnames["tri"], "rb") as fp:
        train_images = np.frombuffer(fp.read(), dtype=np.uint8)[16:].reshape(
            (-1, 28 * 28)
        )
    with gzip.open(fnames["tsl"], "rb") as fp:
        test_labels = np.frombuffer(fp.read(), dtype=np.uint8)[8:]
    with gzip.open(fnames["tsi"], "rb") as fp:
        test_images = np.frombuffer(fp.read(), dtype=np.uint8)[16:].reshape(
            (-1, 28 * 28)
        )
    train = dict(images=train_images, labels=train_labels)
    test = dict(images=test_images, labels=test_labels)

    with gzip.open(archive_fname, "wb") as fp:
        pickle.dump((train, test), fp)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()
    for image in train["images"]:
       plt.clf()
       plt.imshow(image.reshape((28, 28)))
       plt.show()
    pdb.set_trace()
