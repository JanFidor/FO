from typing import NamedTuple
from pathlib import Path
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Mnist(NamedTuple):
    data: npt.NDArray[np.float32]
    target: npt.NDArray[np.int8]

def fetch_mnnist() -> Mnist:
    """
    Download the mnist_64 data from the openml.org by using scikit-learn and
    randomly-extracts neural patterns of 784-dimensional vector with binary units
    from 70 thousand 28*28=784-dimensional vectors with 1 byte units.
    """

    print('start to fetch mnist_64 data from openml.org')
    ds = datasets.load_digits()
    print('finish to fetch mnist_64 data from openml.org')
    return Mnist(data=ds["data"], target=ds["target"])

class MnistForHopfield(NamedTuple):
    original: npt.NDArray[np.int8]
    noised: npt.NDArray[np.int8]

def fetch_minist_for_hopfield(size: int, error_rate: float) -> MnistForHopfield:
    """
    Download the mnist_784 data, and return some of them by renormalizing each unit -1 or 1.
    """
    
    mnist = fetch_mnnist()
    X = mnist.data
    y = mnist.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)

    y_chosen = set()

    X_chosen  = []
    for x, y in zip(X.tolist(), y.tolist()):
        if len(y_chosen) == size: break
        if y in y_chosen: continue

        X_chosen.append(x)
        y_chosen.add(y)


    originals = np.array(X_chosen)
    originals = [np.sign(sample / 255 * 2 - 1) for sample in originals]
    noised = np.copy(originals)
    for i, units in enumerate(noised):
        dim_units = len(units)
        should_flip = np.random.binomial(1, error_rate, dim_units)
        for j in range(dim_units):
            if should_flip[j] != 0:
                units[j] *= -1
    return MnistForHopfield(original=originals, noised=noised)

def show_mnist(data: npt.NDArray[np.int8], title: str):
    num_of_data = len(data)
    fig, ax = plt.subplots(1, num_of_data, figsize=(10, 5))
    for i in range(num_of_data):
        ax[i].matshow(data[i].reshape((8, 8)), cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig.canvas.manager.set_window_title(title)
    plt.show()

if __name__ == '__main__':
    original, noised = fetch_minist_for_hopfield(size=3, error_rate=0.15)
    print('show original data')
    show_mnist(original, 'original data')
    print('show noised data')
    show_mnist(noised, 'noised data')