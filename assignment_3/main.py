import subprocess
import platform

import time

import numpy as np
import pandas as pd

from copy import deepcopy

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


# install sklearnex: https://intel.github.io/scikit-learn-intelex/installation.html

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

        return deepcopy(clusters), arg_distances

    def calculate_cluster_centers(self, clusters, cluster_center_shape):
        new_cluster_centers = np.zeros(cluster_center_shape)

        for cluster_idx in clusters.keys():
            new_cluster_centers[cluster_idx] = np.mean(clusters[cluster_idx], axis=0)

        return new_cluster_centers

    def has_converged(self, cluster_centers, new_cluster_centers):
        centers_almost_the_same = np.allclose(cluster_centers, new_cluster_centers)
        return centers_almost_the_same

    def preprocess_kdd_data(self, X):
        block_ids = X[:, 0].copy()
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

        while (not self.has_converged(cluster_centers, new_cluster_centers)) and i < self.max_iter:
            cluster_centers = new_cluster_centers
            clusters, clustering = self.assign_clusters(X, cluster_centers)
            new_cluster_centers = self.calculate_cluster_centers(clusters, cluster_centers.shape)

            print(normalized_mutual_info_score(block_ids, clustering))

            i += 1

        runtime = time.time() - start_time

        print(f"runtime in seconds: {runtime}")
        print(f"iterations till convergence: {i}")


raw_data = pd.read_csv("data/bio_train.csv").to_numpy()

# shuffle data
raw_data = shuffle(raw_data, random_state=42)

# since we have 153 different values we will use
k = 153

k_means = LloydKDD(k)
k_means.fit(raw_data)
