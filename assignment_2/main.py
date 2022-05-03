import numpy as np
import pandas as pd

from os.path import exists


def import_data(path: str):
    """
    :param path: filepath
    :return: tuple of X and y in case of a csv file. If a npz file is given, the function returns train X,y and test X,y
    """
    file_extension = path.split(".")[-1]

    if exists(path):
        print(path)
        if file_extension == "csv":
            csv = pd.read_csv(path).to_numpy()
            y_idx = csv.shape[1] - 1

            X, y = csv[:, :y_idx], csv[:, y_idx]

            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            print("-------------------")

            return X, y

        elif file_extension == "npz":
            data = np.load(path)

            X_train, y_train, X_test, y_test = data["train"].T, data["train_labels"].T, data["test"].T, data["test_labels"].T

            print(f"X train shape: {X_train.shape}")
            print(f"y train shape: {y_train.shape}")
            print(f"X test shape: {X_test.shape}")
            print(f"X test shape: {y_test.shape}")

            return X_train, y_train, X_test, y_test
        else:
            raise FileNotFoundError("only the extensions (.csv, .npz) are supported")
    else:
        raise FileNotFoundError("file not found")


def predict(x, w):
    return w.T.dot(x)


class SVM:
    def __init__(self, lambda_):
        # hyperparameter
        self.lambda_ = lambda_

    # regularized hinge loss formulation
    def fit(self, X, y):
        def hinge_loss(x, y, w):
            return np.max(0, 1 - y * predict(x, w))

        def regularizer(w):
            return self.lambda_ * np.linalg.norm(w)

        # TODO: implement multi class hinge loss
        def objective(w):
            return regularizer(w) + sum([hinge_loss(x, y_, w) for x, y_ in zip(X, y)])

    def predict(self, X):
        pass


toydata_small_X, toydata_small_y = import_data("data/toydata_tiny.csv")
toydata_large_X, toydata_large_y = import_data("data/toydata_large.csv")
mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = import_data("data/mnist.npz")


