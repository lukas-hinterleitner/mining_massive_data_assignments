import os

import threading

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import exists
from os.path import isdir
from tqdm import tqdm

from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

np.random.seed(42)


def import_data(path: str):
    """
    :param path: filepath
    :return: tuple of X and y in case of a csv file. If a npz file is given, the function returns train X,y and test X,y
    """
    file_extension = path.split(".")[-1]

    if exists(path):
        print("========================================")
        print(f"data from: {path}")
        if file_extension == "csv":
            csv = pd.read_csv(path).to_numpy()
            y_idx = csv.shape[1] - 1

            X, y = csv[:, :y_idx], csv[:, y_idx]

            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

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


# TODO: implement multi class hinge loss
def hinge_loss(x, y, w):  # return the loss and the corresponding gradient
    return max(0, 1 - y * np.dot(w, x))


def regularized_hinge_loss_gradient(loss, x, y, w, C):
    if loss == 0:
        return w
    else:
        return w - C * y * x


class CustomSVM:
    def __init__(self, path_of_datafile, epochs, learning_rate, C, cross_validation_is_used=False,
                 parallelize_sgd=False):
        self.w_star = None
        self.epochs = epochs

        self.learning_rate = learning_rate
        self.C = C

        # use_cross_validation only used for a fancier output
        self.cross_validation_is_used = cross_validation_is_used

        # used to save the plot for sgd
        self.path_of_datafile = path_of_datafile
        self.path_to_figure_file = path_of_datafile.replace("data", "plots").replace(".csv", ".png")

        self.parallelize_sgd = parallelize_sgd

    # get_params needed for sklearn's cross validation
    def get_params(self, deep=True):
        return {"epochs": self.epochs, "learning_rate": self.learning_rate, "C": self.C,
                "cross_validation_is_used": self.cross_validation_is_used,
                "path_of_datafile": self.path_of_datafile, "parallelize_sgd": self.parallelize_sgd}

    # set_params needed for sklearn's cross validation
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # good article: https://towardsdatascience.com/solving-svm-stochastic-gradient-descent-and-hinge-loss-8e8b4dd91f5b
    def fit(self, X, y):
        # initialize w0 with zeros
        w = np.zeros(X.shape[1])

        progress_description = "CV Training Progress" if self.cross_validation_is_used else "Training Progress"

        losses_for_each_epoch = []
        save_sgd_figure = not exists(self.path_to_figure_file)  # only save figure if it doesn't already exist

        # stochastic gradient descent
        for t in tqdm(range(self.epochs), desc=progress_description):
            for _ in range(X.shape[0]):
                random_idx = np.random.randint(0, X.shape[0])

                # use random sample since we are using stochastic gradient descent
                x_random = X[random_idx]
                y_random = y[random_idx]

                loss = hinge_loss(x_random, y_random, w)
                gradient = regularized_hinge_loss_gradient(loss, x_random, y_random, w, self.C)

                # perform weight update
                lr = self.learning_rate / (t + 1)
                w = w - lr * gradient

            # calculate the loss again for each epoch to plot it later
            if save_sgd_figure:
                losses_for_each_epoch.append(np.sum(np.maximum(0, 1 - y * np.dot(X, w))))

        # w_p = simu_parallel_sgd(X, y, self.learning_rate, self.C, k_machines=10)

        self.w_star = w

        if save_sgd_figure:
            plt.title(self.path_of_datafile)
            plt.ylabel("training error")
            plt.xlabel("epochs")
            plt.plot(losses_for_each_epoch)
            plt.savefig(self.path_to_figure_file)
            plt.clf()

        return self

    def predict(self, X):
        return np.sign(np.dot(X, self.w_star))


# as stated in the task description, only needed for the toydata
def run_5_fold_cv_SVM(path, epochs=200, learning_rate=0.0001, C=1.0):
    assert path.split(".")[-1] == "csv", "cross validation should only be used for the toydata"

    X, y = import_data(path)

    start_time = time.time()
    svm = CustomSVM(path, epochs=epochs, learning_rate=learning_rate, C=C, cross_validation_is_used=True)
    print(f"used parameters: C={C}, lr={learning_rate}, epochs={epochs}")
    scores = cross_val_score(svm, X, y, scoring="accuracy", cv=5)
    end_time = time.time()
    print(f"accuracy: {sum(scores) / len(scores)}")
    print(f"runtime using CV: {end_time - start_time} seconds")


class MyThread(threading.Thread):
    def __init__(self, threadID, name, counter, lock, X_partition, y_partition, learning_rate, C, T, append_func):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        self.lock = lock

        self.X_partition = X_partition
        self.y_partition = y_partition

        self.learning_rate = learning_rate
        self.C = C

        self.T = T

        self.append_func = append_func

    def run(self):
        X, y = shuffle(self.X_partition, self.y_partition, random_state=42)

        # initialize w0
        w = np.zeros(X.shape[1])

        for _ in range(200):
            for t in range(self.T):
                x_t = X[t]
                y_t = y[t]

                loss = hinge_loss(x_t, y_t, w)
                gradient = regularized_hinge_loss_gradient(loss, x_t, y_t, w, self.C)

                lr = self.learning_rate / (t + 1)

                w = w - lr * gradient

        self.lock.acquire()
        self.append_func(w, self.counter)
        self.lock.release()


def simu_parallel_sgd(X_, y_, learning_rate, C, k_machines):
    T = X_.shape[0] // k_machines

    X_partitions = np.array_split(X_, k_machines)
    y_partitions = np.array_split(y_, k_machines)

    w_array = np.zeros((k_machines, X_.shape[1]))

    def append_func(w, i_):
        w_array[i_] = w

    lock = threading.Lock()

    # TODO: use threadpool instead of MyThread class
    #  https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python#14299004
    for i in range(k_machines):
        thread = MyThread(i, f"thread {i}", i, lock, X_partitions[i], y_partitions[i], learning_rate, C, T, append_func)
        thread.start()
        thread.join()

    return np.sum(w_array, axis=0) / k_machines


if not isdir("./plots"):
    os.mkdir("./plots")

# we used a heuristic search for parameters instead of GridSearch
# GridSearch would be very resource consuming regarding the runtime
run_5_fold_cv_SVM("./data/toydata_tiny.csv", learning_rate=0.1, C=1.5)
# run_5_fold_cv_SVM("./data/toydata_large.csv", learning_rate=0.001, epochs=200, C=1.5)

mnist_X_train, mnist_y_train, mnist_X_test, mnist_y_test = import_data("./data/mnist.npz")
