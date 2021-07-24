import pdb, pickle, os, sys, time
from pprint import pprint
import gzip, shutil

import numpy as np
import zstandard as zstd


def deskew(image, image_shape=None, negated=False):
    # https://github.com/vsvinayak/mnist-helper
    """
    The MIT License (MIT)

    Copyright (c) 2015 Vinayak V S

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    """
    This method deskwes an image using moments
    :param image: a numpy nd array input image
    :param image_shape: a tuple denoting the image`s shape
    :param negated: a boolean flag telling  whether the input image is a negated one
    :returns: a numpy nd array deskewd image
    """
    image_shape = image_shape if image_shape is not None else image.shape

    # negate the image
    if not negated:
        image = 255 - image

    import cv2

    # calculate the moments of the image
    m = cv2.moments(image)
    if abs(m["mu02"]) < 1e-2:
        return image.copy()

    # caclulating the skew
    skew = m["mu11"] / m["mu02"]
    M = np.float32([[1, skew, -0.5 * image_shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(
        image, M, image_shape, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    )

    return img


dirname = os.path.abspath(os.path.dirname(__file__))
fnames = dict(
    trl=os.path.join(dirname, "train-labels-idx1-ubyte" + ".gz"),
    tri=os.path.join(dirname, "train-images-idx3-ubyte" + ".gz"),
    tsl=os.path.join(dirname, "t10k-labels-idx1-ubyte" + ".gz"),
    tsi=os.path.join(dirname, "t10k-images-idx3-ubyte" + ".gz"),
)
archive_fname = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "mnist_archive.pkl.gz"
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
    test["images"] = np.stack(
        [deskew(z.reshape((28, 28))).reshape(-1) for z in test["images"]]
    )
    train["images"] = np.stack(
        [deskew(z.reshape((28, 28))).reshape(-1) for z in train["images"]]
    )

    with gzip.open(archive_fname, "wb") as fp:
        pickle.dump((train, test), fp)

if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # plt.figure()
    # for image in train["images"]:
    #    plt.clf()
    #    plt.imshow(image.reshape((28, 28)))
    #    plt.show()
    pdb.set_trace()
