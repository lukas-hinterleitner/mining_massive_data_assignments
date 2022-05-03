import numpy as np
import pandas as pd

from os.path import exists
from tqdm import tqdm
from sklearn.metrics import accuracy_score

np.random.seed(42)

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

            X_train, y_train, X_test, y_test = data["train"].T, data["train_labels"].T, data["test"].T, data[
                "test_labels"].T

            print(f"X train shape: {X_train.shape}")
            print(f"y train shape: {y_train.shape}")
            print(f"X test shape: {X_test.shape}")
            print(f"X test shape: {y_test.shape}")

            return X_train, y_train, X_test, y_test
        else:
            raise FileNotFoundError("only the extensions (.csv, .npz) are supported")
    else:
        raise FileNotFoundError("file not found")


class CustomSVM:
    def __init__(self, learning_rate=0.001, epochs=100, C=1):
        self.w_star = None
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.C = C

    # good article: https://towardsdatascience.com/solving-svm-stochastic-gradient-descent-and-hinge-loss-8e8b4dd91f5b
    def fit(self, X, y):
        # TODO: implement multi class hinge loss
        def hinge_loss(x, y, w):  # return the loss and the corresponding gradient
            loss = max(0, 1 - y * np.dot(w, x))

            if loss == 0:
                return loss, w
            else:
                return loss, w - self.C * y * x

        # initialize w0 with zeros
        w = np.zeros(X.shape[1])

        # stochastic gradient descent
        for epoch in tqdm(range(self.epochs)):
            for _ in range(X.shape[0]):
                random_idx = np.random.randint(0, X.shape[0])

                x_random = X[random_idx]
                y_random = y[random_idx]

                loss, gradient = hinge_loss(x_random, y_random, w)

                # perform weight update
                w = w - self.learning_rate * gradient

        self.w_star = w

    def predict(self, X):
        return np.sign(np.dot(X, self.w_star))


toydata_small_X, toydata_small_y = import_data("data/toydata_tiny.csv")
toydata_large_X, toydata_large_y = import_data("data/toydata_large.csv")
mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = import_data("data/mnist.npz")

svm = CustomSVM()
svm.fit(toydata_small_X, toydata_small_y)
predictions = svm.predict(toydata_small_X)
print(accuracy_score(toydata_small_y, predictions))
