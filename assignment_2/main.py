import os
import platform, subprocess

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
from sklearn.svm import SVC
from sklearn.utils import shuffle


# https://stackoverflow.com/a/20161999
def get_processor_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        return subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        return subprocess.check_output(command, shell=True).strip()
    return ""


# use hardware-acceleration for sklearn on Intel processors
if "intel" in str(get_processor_info().lower()):
    from sklearnex import patch_sklearn

    patch_sklearn()

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

            return X, y

        elif file_extension == "npz":
            data = np.load(path)

            X_train, y_train, X_test, y_test = data["train"].T, data["train_labels"].T, data["test"].T, data[
                "test_labels"].T

            return X_train, y_train, X_test, y_test
        else:
            raise FileNotFoundError("only the extensions (.csv, .npz) are supported")
    else:
        raise FileNotFoundError("file not found")


def hinge_loss(x, y, w, C=0, gradient=False):  # return the loss and the corresponding gradient
    loss = np.clip(1 - y * np.dot(x, w), 0, None)

    if not gradient:
        return loss
    else:
        if loss == 0:
            return loss, w
        else:
            return loss, w - C * y * x


def multi_class_hinge_loss(x, y, w, C=0, gradient=False):  # return the loss and the corresponding gradient
    pred = np.dot(x, w)
    loss = np.clip(1 - pred[y.T] + np.max(pred[np.arange(w.shape[1]) != y].reshape(len(y), w.shape[1] - 1)), 0, None)
    if not gradient:
        return loss
    else:
        if loss == 0:
            return loss, 2 * C * w[:, y]
        else:
            grad = np.zeros(w.shape)
            for i in range(w.shape[1]): grad[:, i] = (x + 2 * C * w[:, i].T).T
            grad[:, y] = (-x + 2 * C * w[:, y].T).T
            return loss, grad


class CustomSVM:
    def __init__(self, path_of_datafile, epochs, learning_rate, C, cross_validation_is_used=False,
                 parallelize_sgd=False, RFF=None):
        self.w_star = None
        self.epochs = epochs

        self.learning_rate = learning_rate
        self.C = C

        # use_cross_validation only used for a fancier output
        self.cross_validation_is_used = cross_validation_is_used

        # used to save the plot for sgd
        self.path_of_datafile = path_of_datafile
        self.path_to_figure_file = path_of_datafile.replace("data", "plots").replace(".csv", ".png").replace(".npz",
                                                                                                             ".png")
        if RFF is not None:
            self.path_to_figure_file = self.path_to_figure_file.replace(".png", f"_rff_{RFF['num_rffs']}.png")

        self.RFF = RFF

        self.parallelize_sgd = parallelize_sgd

    # get_params needed for sklearn's cross validation
    def get_params(self, deep=True):
        return {"epochs": self.epochs, "learning_rate": self.learning_rate, "C": self.C,
                "cross_validation_is_used": self.cross_validation_is_used,
                "path_of_datafile": self.path_of_datafile, "parallelize_sgd": self.parallelize_sgd, "RFF": self.RFF}

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
                    losses_for_each_epoch.append(np.mean(loss_gradient_function(X, y, w, C=self.C, gradient=False)))

            self.w_star = w

            if save_sgd_figure:
                plt.title(self.path_of_datafile)
                plt.ylabel("training error")
                plt.xlabel("epochs")
                plt.plot(np.arange(1, self.epochs + 1), losses_for_each_epoch)
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
def evaluate_SVC(path, epochs=200, learning_rate=0.001, C=1.0, parallelize_sgd=False, RFF=None, subset_size=0):
    # RFF should be a dict with format : {"sigma": 1, "num_rffs": 100, subset_size=0}
    data = import_data(path)

    # initialize variables
    X, y, X_test, y_test = None, None, None, None

    if len(data) == 2:
        X, y = shuffle(data[0], data[1], random_state=42)
        y = y.astype(int)

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    elif len(data) == 4:
        X, y, X_test, y_test = data[0], data[1], data[2], data[3]
        X, y = shuffle(X, y, random_state=42)

        if subset_size > 0:
            X = X[:subset_size, :]
            y = y[:subset_size]

        print(f"X train shape: {X.shape}")
        print(f"y train shape: {y.shape}")
        print(f"X test shape: {X_test.shape}")
        print(f"y test shape: {y_test.shape}")

        y, y_test = y.astype(int), y_test.astype(int)

    else:
        raise ValueError("got some strange data :/")

    # apply RFF
    if RFF is not None:
        print("training model using RFF")
        print(f"RFF parameters = {RFF}")
        W, b = generate_rff_transformation(X, RFF["sigma"], RFF["num_rffs"])
        X = rff_transform(X, W, b)

        if X_test is not None:
            X_test = rff_transform(X_test, W, b)

    start_time = time.time()

    svm = CustomSVM(path, epochs=epochs, learning_rate=learning_rate, C=C, cross_validation_is_used=not parallelize_sgd and len(data) == 2,
                    parallelize_sgd=parallelize_sgd, RFF=RFF)
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
    runtime = end_time - start_time
    print(f"runtime: {runtime} seconds")

    return accuracy, runtime, data if subset_size == 0 else (X, y, X_test, y_test)


# --- RANDOM FOURIER FEATURES ---
def generate_rff_transformation(X, sigma, num_rffs):
    # Matrix of random vectors w_i10
    W = 1 / sigma * np.random.standard_cauchy(num_rffs * X.shape[1])
    W = W.reshape(X.shape[1], num_rffs)

    # Vector of random values b_i
    b = 2 * np.pi * np.random.rand(num_rffs)
    return (W, b)


def rff_transform(X, W, b):
    zx = np.sqrt(2 / b.shape[0]) * np.cos(np.dot(X, W) + b)
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
        # initialize w0
        w = np.zeros(self.X_partition.shape[1])

        # use hinge loss when we have a binary classification problem and the multi class hinge loss otherwise
        loss_gradient_function = multi_class_hinge_loss if np.unique(self.y_partition).shape[0] > 2 else hinge_loss

        for _ in range(self.epochs):
            for t in range(self.T):
                x_t = self.X_partition[t]
                y_t = self.y_partition[t]

                loss, gradient = loss_gradient_function(x_t, y_t, w, self.C, gradient=True)

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

for data_path in ["./data/mnist.npz"]:  # TODO change to original
    test_rffs = [100, 200, 500, 1000]

    for rff in test_rffs:
        accuracy, _, _ = evaluate_SVC(data_path, epochs=200, RFF={"sigma": 1, "num_rffs": rff})
        print(f"accuracy using {rff} random fourier features: {accuracy}")

# plot mnist subset performance and runtime
subset_sizes = [1000, 2000, 3000]

accuracies = {"custom": [], "sklearn": []}
runtimes = {"custom": [], "sklearn": []}

accuracy_filepath = "./plots/mnist_accuracy.png"
runtime_filepath = "./plots/mnist_performance.png"

for subset_size in subset_sizes:
    accuracy_custom, runtime_custom, data = evaluate_SVC(data_paths[2], epochs=200, RFF={"sigma": 1, "num_rffs": 100},
                                                         subset_size=subset_size)

    accuracies["custom"].append(accuracy_custom)
    runtimes["custom"].append(runtime_custom)

    X, y, X_test, y_test = data[0][:subset_size], data[1][:subset_size], data[2], data[3]

    start_time = time.time()
    sklearn_svc = SVC(C=1.5, max_iter=200)
    sklearn_svc.fit(X, y)
    y_pred = sklearn_svc.predict(X_test)
    accuracies["sklearn"].append(accuracy_score(y_test, y_pred))
    runtimes["sklearn"].append(time.time() - start_time)

plt.title("mnist accuracy")
plt.ylabel("accuracy")
plt.xlabel("sample size")
plt.plot(subset_sizes, accuracies["custom"], label="custom")
plt.plot(subset_sizes, accuracies["sklearn"], label="sklearn")
plt.legend()
plt.savefig(accuracy_filepath)
plt.clf()

plt.title("mnist runtime")
plt.ylabel("runtime in seconds")
plt.xlabel("sample size")
plt.plot(subset_sizes, runtimes["custom"], label="custom")
plt.plot(subset_sizes, runtimes["sklearn"], label="sklearn")
plt.legend()
plt.savefig(runtime_filepath)
plt.clf()

# task 3

print("\n\nTask3 (Parallel Implementation)\n")

# use parallelization
evaluate_SVC(data_paths[0], learning_rate=0.1, C=1.5, parallelize_sgd=True)
evaluate_SVC(data_paths[1], learning_rate=0.1, C=1.5, parallelize_sgd=True)
evaluate_SVC(data_paths[2], learning_rate=0.1, C=1.5, parallelize_sgd=True)
