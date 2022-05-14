import os

import threading

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import exists
from os.path import isdir
from tqdm import tqdm
from multiprocessing import cpu_count

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
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
            print(f"y test shape: {y_test.shape}")

            return X_train, y_train, X_test, y_test
        else:
            raise FileNotFoundError("only the extensions (.csv, .npz) are supported")
    else:
        raise FileNotFoundError("file not found")


def hinge_loss(x, y, w, C=0, gradient=False):  # return the loss and the corresponding gradient
    loss = max(0, 1 - y * np.dot(w, x))

    if not gradient:
        return loss
    else:
        if loss == 0:
            return loss, w
        else:
            return loss, w - C * y * x


def multi_class_hinge_loss(x, y, w, C = 0, gradient=False):  # return the loss and the corresponding gradient
    loss = max(0, 1 + max(np.dot(w[:,np.arange(w.shape[1]) != y].T, x)) - np.dot(w[:,y].T, x)) + C * np.dot(w[:,y].T, w[:,y])
    # use is to avoid abiguity with 0
    if not gradient:
        return loss
    else:
        if loss == 0:
            return loss, 2 * C * w[:,y]
        else:
            grad = np.zeros(w.shape)
            for i in range(w.shape[1]): grad[:,np.array([i])] = (x + 2 * C * w[:,y].T).T
            grad[:, y] = (-x + 2 * C * w[:,y].T).T
            return(loss, grad)


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
        self.path_to_figure_file = path_of_datafile.replace("data", "plots").replace(".csv", ".png").replace(".npz", ".png")

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
        if len(np.unique(y)) == 2:
            w = np.zeros(X.shape[1])
        else:
            w = np.zeros((X.shape[1], len(np.unique(y))))

        progress_description = "CV Training Progress" if self.cross_validation_is_used else "Training Progress"

        # use hinge loss when we have a binary classification problem and the multi class hinge loss otherwise
        loss_gradient_function = multi_class_hinge_loss if np.unique(y).shape[0] > 2 else hinge_loss

        if self.parallelize_sgd:
            self.w_star = simu_parallel_sgd(X, y, self.learning_rate, self.C, self.epochs, k_machines=cpu_count())
        else:
            losses_for_each_epoch = []
            save_sgd_figure = not exists(self.path_to_figure_file)  # only save figure if it doesn't already exist

            # stochastic gradient descent
            for t in tqdm(range(self.epochs), desc=progress_description):
                for _ in range(X.shape[0]):
                    random_idx = np.random.randint(0, X.shape[0])

                    # use random sample since we are using stochastic gradient descent
                    x_random = X[random_idx]
                    y_random = y[random_idx]

                    loss, gradient = loss_gradient_function(x_random, y_random, w, C=self.C, gradient=True)

                    # perform weight update
                    lr = self.learning_rate / (t + 1)
                    w = w - lr * gradient

                # calculate the loss again for each epoch to plot it later
                if save_sgd_figure:
                    losses_for_each_epoch.append(np.sum(loss_gradient_function(X, y, w, C=self.C, gradient=False)))

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
        if len(self.w_star.shape) == 1:
            prediction = np.sign(np.dot(X, self.w_star))
        else:
            prediction = np.argmax(np.dot(X, self.w_star), axis=1)
        return prediction


# as stated in the task description, only needed for the toydata
def evaluate_SVC(path, epochs=200, learning_rate=0.001, C=1.0, parallelize_sgd=False, RFF=None):
    # RFF should be a dict with format : {"sigma": 1, "num_rffs": 100}
    data = import_data(path)

    # initialize variables
    X, y, X_test, y_test = None, None, None, None

    if len(data) == 2:
        X, y = data[0], data[1]
        y = y.astype(int)
    elif len(data) == 4:
        X, y, X_test, y_test = data[0], data[1], data[2], data[3]
        y, y_test = y.astype(int), y_test.astype(int)
    else:
        raise ValueError("got some strange data :/")

    # apply RFF
    if RFF is not None:
        print("training model using RFF")
        print(f"RFF parameters = {RFF}")
        X = rff_transform(X, sigma=RFF["sigma"], num_rffs=RFF["num_rffs"])

        if X_test is not None:
            X_test = rff_transform(X, sigma=RFF["sigma"], num_rffs=RFF["num_rffs"])

    start_time = time.time()

    # do not use cross
    svm = CustomSVM(path, epochs=epochs, learning_rate=learning_rate, C=C, cross_validation_is_used=not parallelize_sgd,
                    parallelize_sgd=parallelize_sgd)
    print(f"used parameters: C={C}, lr={learning_rate}, epochs={epochs}")

    accuracy = 0.0

    if parallelize_sgd:
        print("using a parallelized approach")

    if len(data) == 2:
        print("using cross validation")
        scores = cross_val_score(svm, X, y, scoring="accuracy", cv=5)
        accuracy = sum(scores) / len(scores)
    else:
        svm.fit(X, y)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    print(f"accuracy: {accuracy}")
    end_time = time.time()
    print(f"runtime: {end_time - start_time} seconds")

    return accuracy


# --- RANDOM FOURIER FEATURES ---


def rff_transform(X, sigma, num_rffs):
    # Initialize transformed data
    zx = np.zeros([X.shape[0], num_rffs])

    for i in range(X.shape[0]):
        # Matrix of random vectors w_i10
        W = 1 / sigma * np.random.standard_cauchy(num_rffs * X.shape[1])
        W = W.reshape(num_rffs, X.shape[1])

        # Vector of random values b_i
        b = 2 * np.pi * np.random.rand(num_rffs)

        # Transformation
        zx[i, :] = np.sqrt(2 / num_rffs) * np.cos(W @ X[i, :] + b)

    return zx


# --- PARALLELISM ---

class SimuParallelSGDThread(threading.Thread):
    def __init__(self, counter, lock, X_partition, y_partition, epochs, learning_rate, C, T, append_func):
        threading.Thread.__init__(self)
        self.counter = counter

        self.lock = lock

        self.X_partition = X_partition
        self.y_partition = y_partition

        self.epochs = epochs

        self.learning_rate = learning_rate
        self.C = C

        self.T = T

        self.append_func = append_func

    def run(self):
        X, y = shuffle(self.X_partition, self.y_partition, random_state=42)

        # initialize w0
        w = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            for t in range(self.T):
                x_t = X[t]
                y_t = y[t]

                loss, gradient = hinge_loss(x_t, y_t, w, self.C, gradient=True)

                lr = self.learning_rate / (t + 1)

                w = w - lr * gradient

        self.lock.acquire()
        self.append_func(w, self.counter)
        self.lock.release()


def simu_parallel_sgd(X_, y_, learning_rate, C, epochs, k_machines):
    T = X_.shape[0] // k_machines

    X_partitions = np.array_split(X_, k_machines)
    y_partitions = np.array_split(y_, k_machines)

    w_array = np.zeros((k_machines, X_.shape[1]))

    def append_func(w, i_):
        w_array[i_] = w

    lock = threading.Lock()

    threads = []

    for i in tqdm(range(k_machines)):
        thread = SimuParallelSGDThread(i, lock, X_partitions[i], y_partitions[i], epochs, learning_rate, C, T,
                                       append_func)
        thread.start()

        threads.append(thread)

    for thread in threads:
        thread.join()

    return np.sum(w_array, axis=0) / k_machines


if not isdir("./plots"):
    os.mkdir("./plots")

data_paths = ("./data/toydata_tiny.csv", "./data/toydata_large.csv", "./data/mnist.npz")

# task 1

# we used a heuristic search for parameters instead of GridSearch
# GridSearch would be very resource consuming regarding the runtime, therefore we decided to stick with a manual search

print("Task 1 (Linear SVM Model)\n")

# evaluate_SVC(data_paths[0], learning_rate=0.1, C=1.5, )
# evaluate_SVC(data_paths[1], learning_rate=0.1, C=1.5, )
# evaluate_SVC(data_paths[2], learning_rate=0.1, C=1.5, )

# task 2

print("\n\nTask2 (Random Fourier Features)\n")

for data_path in data_paths:
    test_rffs = [100, 200, 500, 1000]

    for rff in test_rffs:
        evaluate_SVC(data_path, RFF={"sigma": 1, "num_rffs": rff})

    print(f"accuracies using")

# task 3

print("\n\nTask3 (Parallel Implementation)\n")

# use parallelization
evaluate_SVC(data_paths[0], learning_rate=0.1, C=1.5, parallelize_sgd=True)
# evaluate_SVC(data_paths[1], learning_rate=0.1, C=1.5, parallelize_sgd=True)
# evaluate_SVC(data_paths[2], learning_rate=0.1, C=1.5, parallelize_sgd=True)
