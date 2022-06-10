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

distance_computation_counter = 0
distance_computation_counter_lsh = 0

# lloyd's algorithm for the kdd data set
class LloydKDD:
    # starting lloyd's heuristic with max_iter=float("inf") will run till it converges
    def __init__(self, k, max_iter=float("inf"), method=None, num_AND=1, num_OR=1, width=1):
        
        """
        k...number of clusters
        
        max_iter...maximum number of iterations 
        
        method...method to be used. "LSH" is locality sensitive hashing
        
        num_AND, num_OR...number of AND combinations / OR combinations for LSH 
            Example: num_AND=3, num_OR=2: (h1 and h2 and h3) or (h4 and h5 and h6)
            uses 6 hash functions in total
        
        width...width of intervals used for LSH
        
        
        """
        self.k = k
        self.max_iter = max_iter
        self.method = method
        
        if self.method == "LSH":
            
            self.num_AND = num_AND
            self.num_OR = num_OR
            self.width = width
            
            self.num_hashes = self.num_AND * self.num_OR
            

    def assign_clusters(self, X, cluster_centers):
        
        
        if self.method == "LSH":
            global distance_computation_counter_lsh

            
            hash_centers = np.floor((cluster_centers@self.a + self.b)/self.width).astype(int)
            
            assigned_to_cluster = np.full([X.shape[0], cluster_centers.shape[0]], False)


            for i in range(cluster_centers.shape[0]):

                same_bucket = self.hash_matrix == hash_centers[i,:]
                
                same_AND = np.full([X.shape[0], self.num_OR], False)
                
                for j in range(self.num_OR):
                    
                    same_AND[:,j] = np.all(same_bucket[:,(self.num_AND*j):(self.num_AND*(j+1))], axis=1)
            
                same_OR = np.any(same_AND, axis=1)
                
                assigned_to_cluster[:,i] = same_OR
            
        
            sum_assigned = assigned_to_cluster.sum(axis=1)
            none_assigned = sum_assigned == 0
            
            
            distances = pairwise_distances(X[none_assigned,:], cluster_centers, "euclidean")
            distance_computation_counter_lsh += np.sum(none_assigned) * cluster_centers.shape[0]
            
            arg_distances = distances.argmin(axis=1)

            labels_pred = np.empty(X.shape[0])
            clusters = {}
            
            count = 0
            for i, x in enumerate(X):
                
                if sum_assigned[i] > 0:
                    
                    potential_idx = assigned_to_cluster[i,:].nonzero()
                    
                    cluster_idx = np.random.choice(potential_idx[0], size=1)[0]
                
                else: 
                    cluster_idx = arg_distances[count]
                    count += 1

                # associate data point to corresponding cluster
                try:
                    clusters[cluster_idx].append(x)
                except KeyError:
                    clusters[cluster_idx] = [x]
                
                labels_pred[i] = cluster_idx
    
    
        
        
        else:
            
            global distance_computation_counter
            
            # use sklearn pairwise_distances instead of cdist from scipy.spatial.distance because if performs faster
            distances = pairwise_distances(X, cluster_centers, "euclidean")
            distance_computation_counter += X.shape[0] * cluster_centers.shape[0]
    
            # get the min distance to each cluster as index array
            labels_pred = distances.argmin(axis=1)

            clusters = {}
    
            for i, x in enumerate(X):
                cluster_idx = labels_pred[i]
    
                # associate data point to corresponding cluster
                try:
                    clusters[cluster_idx].append(x)
                except KeyError:
                    clusters[cluster_idx] = [x]
    
        
        return clusters, labels_pred

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
        X = StandardScaler().fit_transform(X)

        return X, block_ids
    
    def compute_hash_matrix(self, X):
        
        a = np.random.normal(size = X.shape[1]*self.num_hashes)
        a = a.reshape(X.shape[1], self.num_hashes)

        b = np.random.random(self.num_hashes) * self.width
            
        hash_matrix = np.floor((X@a + b)/self.width)
        
        self.a = a
        self.b = b
        self.hash_matrix = hash_matrix.astype(int)
        

    def fit(self, X):
        # block_ids contain the true clustering
        X, block_ids = self.preprocess_kdd_data(X)

        start_time = time.time()

        # If method LSH is specified, compute hash matrix
        if self.method == "LSH": self.compute_hash_matrix(X)
            

        # initialize cluster centers
        random_indices = np.random.choice(X.shape[0], size=self.k)
        cluster_centers = X[random_indices]
        new_cluster_centers = cluster_centers + 1e-06

        # iteration counter
        i = 0

        clustering = np.zeros(X.shape[0])
        clustering_old = np.ones(X.shape[0])

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


def task_1(raw_data, k):
    global distance_computation_counter
    # execute Task 1: Lloyd’s algorithm for k-Means Clustering (34%)
    print("Task 1: Lloyd’s algorithm for k-Means Clustering")

    overall_nmi_scores, overall_runtime = [], []

    for l in range(5):
        print(f"Iteration {l + 1}:")

        k_means = LloydKDD(k = k, max_iter=10)
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
    print(f"distance computations: {distance_computation_counter}")
    print("========================================================\n")


def task_2(raw_data, k):
    global distance_computation_counter_lsh
    # execute Task 2: LSH for k-Means Clustering (33%)
    print("Task 2: LSH for  k-Means Clustering")

    overall_nmi_scores, overall_runtime = [], []

    for l in range(5):
        print(f"Iteration {l + 1}:")

        k_means = LloydKDD(k = k, max_iter=10, method="LSH", num_AND=3, num_OR=2, width=10)
        nmi_scores, runtime, iterations = k_means.fit(raw_data)

        # save figure of k-means convergence once
        if l == 0:
            save_fig(range(len(nmi_scores)), nmi_scores, "Iterations", "NMI-Score", "NMI-Scores", "./plots/lsh_k_means.png")

        # use final nmi score and append it overall scores
        overall_nmi_scores.append(nmi_scores[-1])
        overall_runtime.append(runtime)

    print()
    print(f"averaged nmi scores: {np.mean(overall_nmi_scores)}")
    print(f"averaged runtime in seconds: {np.mean(overall_runtime)}")
    print(f"distance computations: {distance_computation_counter_lsh}")
    print("========================================================\n")



def main():
    raw_data = pd.read_csv("./data/bio_train.csv").to_numpy()

    # shuffle data
    raw_data = shuffle(raw_data, random_state=42)

    # use 153 cluster
    k = 153
    
    raw_data = raw_data[:20000,:]
    
    task_1(raw_data, k)
    task_2(raw_data, k)

if __name__ == "__main__":
    main()



#--------------------------------------------------


raw_data = pd.read_csv("./data/bio_train.csv").to_numpy()

# shuffle data
raw_data = shuffle(raw_data, random_state=42)


X = raw_data[:10,:]

num_AND=3
num_OR =2

width=20000

num_hashes = num_AND*num_OR



a = np.random.normal(size=X.shape[1]*num_hashes).reshape(X.shape[1],num_hashes)

b = np.random.random(num_hashes) * width

temp = X@a

temp2 = (X@a+b)/width
temp2.astype(int)

res = np.floor((X@a+b)/width)

hash_matrix = res.astype(int)





cluster_centers = raw_data[100:105,:]
cluster_centers.shape

assigned_to_cluster = np.full([X.shape[0], cluster_centers.shape[0]], False)

for i in range(cluster_centers.shape[0]):
    
    center = cluster_centers[i,:]
    
    hash_center = np.floor((center@a+b)/width).astype(int)
    
    same_bucket = hash_matrix == hash_center
    
    #print(same_bucket)
    same_AND = np.full([X.shape[0], num_OR], False)
    
    for j in range(num_OR):
        
         
        same_AND[:,j] = np.all(same_bucket[:,(num_AND*j):(num_AND*(j+1))], axis=1)

    same_OR = np.any(same_AND, axis=1)
    assigned_to_cluster[:,i] = same_OR
    
print(assigned_to_cluster)

np.where(assigned_to_cluster,axis=1)
assigned_to_cluster[6,:].nonzero()

any_assigned = np.any(assigned_to_cluster, axis=1)
sum_assigned = assigned_to_cluster.sum(axis=1)

for k in range(len(sum_assigned)):
    #print(k)
    if sum_assigned[k] > 0:
        
        temp = assigned_to_cluster[k,:].nonzero()
        print(temp[0])
        print(assigned_to_cluster[k,:])
        print(np.random.choice(temp[0]))
        
np.random.choice([4],size=1)[0]

X[np.invert(any_assigned),:].shape

np.sum(assigned_to_cluster,axis=1)

assigned_to_cluster.sum(axis=1)

assigned_to_cluster.argmin(axis=1)


np.full([num_hashes, num_OR], True)

np.array(False)

np.floor(1.5)
a.shape
X.shape

dic1 = {}
dic2 = {}


dic1[1] = [11]
dic1[1] = [12]

dic2[1]=[13,14]

dic3 = dic1.update(dic2)
dic1
dic2
dic3
print(dic1)



def task_2(raw_data, k):
    global distance_computation_counter
    # execute Task 2: LSH for k-Means Clustering (33%)
    print("Task 2: LSH for  k-Means Clustering")

    overall_nmi_scores, overall_runtime = [], []

    for l in range(5):
        print(f"Iteration {l + 1}:")

        k_means = LloydKDD(k = k, max_iter=10, method="LSH", num_AND=3, num_OR=2, width=10)
        nmi_scores, runtime, iterations = k_means.fit(raw_data)

        # save figure of k-means convergence once
        if l == 0:
            save_fig(range(len(nmi_scores)), nmi_scores, "Iterations", "NMI-Score", "NMI-Scores", "./plots/lsh_k_means.png")

        # use final nmi score and append it overall scores
        overall_nmi_scores.append(nmi_scores[-1])
        overall_runtime.append(runtime)

    print()
    print(f"averaged nmi scores: {np.mean(overall_nmi_scores)}")
    print(f"averaged runtime in seconds: {np.mean(overall_runtime)}")
    print(f"distance computations: {distance_computation_counter}")
    print("========================================================\n")

raw_data = pd.read_csv("./data/bio_train.csv").to_numpy()

    # shuffle data
raw_data = shuffle(raw_data, random_state=42)

    # use 153 cluster
k = 153

distance_computation_counter = 0
task_2(raw_data, k)
distance_computation_counter = 0
task_1(raw_data,k)






