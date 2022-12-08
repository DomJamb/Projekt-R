import numpy as np
from torchvision import datasets

def load_mnist(dataset_root="./data/"):
    mnist_train = datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = datasets.MNIST(dataset_root, train=False, download=False)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0).reshape(-1, 784), x_test.float().div_(255.0).reshape(-1, 784)

    return x_train, y_train, x_test, y_test

def class_to_onehot(Y):
    Yoh = np.zeros((len(Y), max(Y) + 1))
    Yoh[range(len(Y)), Y] = 1
    return Yoh