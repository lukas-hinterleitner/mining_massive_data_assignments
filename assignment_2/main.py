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
        if file_extension == "csv":
            csv = pd.read_csv(path).to_numpy()
            y_idx = csv.shape[1] - 1

            return csv[:, :y_idx], csv[:, y_idx]

        elif file_extension == "npz":
            data = np.load(path)

            return data["train"], data["train_labels"], data["test"], data["test_labels"]
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

        def objective(w):
            return regularizer(w) + sum([hinge_loss(x, y_, w) for x, y_ in zip(X, y)])

        pass

    def predict(self, X):
        pass


data_paths = ["data/toydata_tiny.csv", "data/toydata_large.csv", "data/mnist.npz"]

X, y = import_data(data_paths[0])

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
