import subprocess
import platform

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from os.path import exists

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import pairwise_distances


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


# install sklearn-intelex: https://intel.github.io/scikit-learn-intelex/installation.html

# use hardware-acceleration for sklearn on Intel processors
if "intel" in str(get_processor_info().lower()):
    from sklearnex import patch_sklearn

    patch_sklearn()

np.random.seed(42)


# lloyd's algorithm for the kdd data set
class LloydKDD:
    # starting lloyd's heuristic with max_iter=float("inf") will run till it converges
    def __init__(self, max_iter=float("inf")):
        self.k = k
        self.max_iter = max_iter

    def assign_clusters(self, X, cluster_centers):
        # use sklearn pairwise_distances instead of cdist from scipy.spatial.distance because if performs faster
        distances = pairwise_distances(X, cluster_centers, "euclidean")

        # get the min distance to each cluster as index array
        arg_distances = distances.argmin(axis=1)

        clusters = {}

        for i, x in enumerate(X):
            cluster_idx = arg_distances[i]

            # associate data point to corresponding cluster
            try:
                clusters[cluster_idx].append(x)
            except KeyError:
                clusters[cluster_idx] = [x]

        return clusters, arg_distances

    def calculate_cluster_centers(self, clusters, cluster_center_shape):
        new_cluster_centers = np.zeros(cluster_center_shape)

        for cluster_idx in clusters.keys():
            new_cluster_centers[cluster_idx] = np.mean(clusters[cluster_idx], axis=0)

        return new_cluster_centers

    def has_converged(self, clustering, old_clustering):
        return np.array_equiv(clustering, old_clustering)

    def preprocess_kdd_data(self, X):
        block_ids = X[:, 0]

        X = X[:, 1:]
        # X = StandardScaler().fit_transform(X)

        return X, block_ids

    def fit(self, X):
        # block_ids contain the true clustering
        X, block_ids = self.preprocess_kdd_data(X)

        start_time = time.time()

        # initialize cluster centers
        random_indices = np.random.choice(X.shape[0], size=self.k)
        cluster_centers = X[random_indices]
        new_cluster_centers = cluster_centers + 1e-06

        # iteration counter
        i = 0

        clustering = np.zeros(X.shape[0])
        clustering_old = np.empty(X.shape[0])

        nmi_scores_ = []

        while (not self.has_converged(clustering, clustering_old)) and i < self.max_iter:
            cluster_centers = new_cluster_centers
            clustering_old = clustering
            clusters, clustering = self.assign_clusters(X, cluster_centers)
            new_cluster_centers = self.calculate_cluster_centers(clusters, cluster_centers.shape)

            nmi_scores_.append(normalized_mutual_info_score(block_ids, clustering))

            i += 1

        runtime_ = time.time() - start_time

        print(f"\truntime in seconds: {runtime_}")
        print(f"\titerations till convergence: {i}")
        print(f"\tNMI-score: {np.mean(nmi_scores_)}")

        return nmi_scores_, runtime_, i


def save_fig(x, y, x_label, y_label, title, path):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)
    plt.savefig(path)
    plt.clf()


raw_data = pd.read_csv("./data/bio_train.csv").to_numpy()

# shuffle data
raw_data = shuffle(raw_data, random_state=42)

# use 153 cluster
k = 153

# execute Task 1: Lloyd’s algorithm for k-Means Clustering (34%)
print("Task 1: Lloyd’s algorithm for k-Means Clustering")

overall_nmi_scores, overall_runtime = [], []

for l in range(5):
    print(f"Iteration {l + 1}:")

    k_means = LloydKDD(k)
    nmi_scores, runtime, iterations = k_means.fit(raw_data)

    # save figure of k-means convergence once
    if l == 0:
        save_fig(range(len(nmi_scores)), nmi_scores, "Iterations", "NMI-Score", "NMI-Scores", "./plots/k_means.png")

    # use final nmi score and append it overall scores
    overall_nmi_scores.append(nmi_scores[-1])
    overall_runtime.append(runtime)

print()
print(f"averaged nmi scores: {np.mean(overall_nmi_scores)}")
print(f"averaged runtime in seconds: {np.mean(overall_runtime)}")
print("========================================================\n")
