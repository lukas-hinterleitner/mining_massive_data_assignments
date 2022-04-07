from enum import Enum
from math import sqrt
from tqdm import tqdm
from scipy.spatial.distance import cosine

import numpy as np
import pandas as pd


# creates a random matrix with shape n x d
def create_random_matrix(n, d):
    # random matrix construction using the method introduced by Achlioptas
    return np.random.default_rng().choice([+1, 0, -1], size=(n, d), p=[1 / 6, 2 / 3, 1 / 6]) * sqrt(3)


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.__hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = create_random_matrix(self.__hash_size, inp_dimensions)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, list()) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.__num_tables = num_tables
        self.__hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.__num_tables):
            self.hash_tables.append(HashTable(self.__hash_size, self.inp_dimensions))

    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label

    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))


# enum just used to define the distance measures for our k nearest neighbour function
class DistanceMeasure(Enum):
    euclidian = 1
    cosine = 2


# import and split tracks dataset
tracks = pd.read_csv("fma_metadata/tracks.csv", index_col=0, header=[0, 1])
tracks = tracks[tracks["set"]["subset"] == "small"]

# split tracks data to later extract features by track id
# track id's are embedded in the index column because we imported the data with parameter index_col=0

tracks_training_data = tracks[tracks["set"]["split"] == "training"]
tracks_validation_data = tracks[tracks["set"]["split"] == "validation"]
tracks_test_data = tracks[tracks["set"]["split"] == "test"]

print(f"tracks training data shape: {tracks_training_data.shape}")
print(f"tracks validation data shape: {tracks_training_data.shape}")
print(f"tracks test data shape: {tracks_training_data.shape}\n")


# approximate nearest neighbour classifier using LSH
class ApproximateNearestNeighbour:
    """
    l = length of the hash value
    n = amount of hash tables
    k = use k nearest neighbours
    m = similarity measure (cosine or euclidian distance)
    """

    def __init__(self, l: int, n: int, k: int, m: DistanceMeasure):
        self.__lsh = None
        self.__training_set = None
        self.__hash_size = l
        self.__num_tables = n
        self.__k_nearest_neighbours = k
        self.__distance_measure = m

    def train(self, training_set):
        self.__lsh = LSH(self.__num_tables, self.__hash_size, training_set.shape[1])
        self.__training_set = training_set

        for (index, track) in tqdm(zip(training_set.index, training_set.values), desc="Training"):
            self.__lsh[track] = index

    def predict(self, validation_set):
        # distance function can be euclidian or cosine distance
        distance_function = lambda x, y: np.linalg.norm(x - y)

        if self.__distance_measure == DistanceMeasure.cosine:
            distance_function = cosine

        """
        this list will be filled with 0 and 1
        a zero will be appended if the predicted genre is incorrect
        a one will be appended if the predicted genre is correct"""
        genres_identified = []

        # iterate over validation tracks and calculate nearest neighbours based on LSH
        for validation_track_id in tqdm(validation_set.index, desc="Prediction"):
            validation_track = validation_set.loc[validation_track_id]

            # neighbours of track with given track_index
            neighbour_ids = self.__lsh[validation_track]
            neighbours = self.__training_set.filter(neighbour_ids, axis=0)

            # if no nearest neighbours could be found with LSH, skip this track and continue the loop
            if len(neighbour_ids) == 0:
                continue

            # calculate distances between neighbours and validation track with specific track_id
            distances = []
            for i in range(len(neighbours)):
                # append tuple of neighbour id and actual distance to validation track
                distances.append((neighbour_ids[i], distance_function(neighbours.iloc[i], validation_track)))

            # sort distances ascending
            distances.sort(key=lambda x: x[1])

            # use k nearest neighbours for classification
            # if only less than k nearest neighbours can be found, use only these neighbours
            # TODO: decide if we want this behaviour or just ignore tracks with less than k neighbours
            distances = distances[:min(self.__k_nearest_neighbours, len(distances))]

            # extract ids of k nearest neighbours to get the corresponding genres from the tracks dataset later
            nearest_neighbour_ids = np.array(distances)[:, 0].astype(int)

            # dataframe.loc gets the element based on the index label (in our case this is the track_id)
            validation_track_genre = tracks.loc[validation_track_id]['track']['genre_top']

            # get genres for the k nearest neighbours
            k_nearest_neighbours_genres = tracks.filter(items=nearest_neighbour_ids, axis=0)['track']['genre_top']

            # get most frequent genre from k nearest neighbours
            most_frequent_genre = k_nearest_neighbours_genres.mode()[0]

            # append 0 or 1
            genres_identified.append(int(validation_track_genre == most_frequent_genre))

        # return accuracy
        return sum(genres_identified) / len(genres_identified)


# import and split features dataset
features = pd.read_csv("fma_metadata/features.csv", index_col=0, header=[0, 1, 2])

features_training_data = features.filter(items=tracks_training_data.index, axis=0)
features_validation_data = features.filter(items=tracks_validation_data.index, axis=0)
features_test_data = features.filter(items=tracks_test_data.index, axis=0)

print(f"features training data shape: {features_training_data.shape}")
print(f"features validation data shape: {features_validation_data.shape}")
print(f"features test data shape: {features_test_data.shape}", "\n")

# try different parameters
l_values = [50, 100, 150]
n_values = [10, 15, 20]
k_values = [10, 15, 20]

best_parameters = {}
best_accuracy = 0

for l, n, k in zip(l_values, n_values, k_values):
    print(f"\nParameters: l={l}, n={n}, k={k}")
    classifier_euclidian = ApproximateNearestNeighbour(l, n, k, DistanceMeasure.euclidian)
    classifier_euclidian.train(features_training_data)

    accuracy_euclidian = classifier_euclidian.predict(features_validation_data)
    print(f"accuracy using euclidian distance: {accuracy_euclidian}\n")

    classifier_cosine = ApproximateNearestNeighbour(l, n, k, DistanceMeasure.cosine)
    classifier_cosine.train(features_training_data)

    accuracy_cosine = classifier_cosine.predict(features_validation_data)
    print(f"accuracy using cosine distance: {accuracy_cosine}\n")

    if accuracy_euclidian > best_accuracy:
        best_parameters = {"l": l, "n": n, "k": k, "m": DistanceMeasure.euclidian}
        best_accuracy = accuracy_euclidian

    if accuracy_cosine > best_accuracy:
        best_parameters = {"l": l, "n": n, "k": k, "m": DistanceMeasure.cosine}
        best_accuracy = accuracy_cosine

    print("--------------------------------------------------------------")

# combine training and validation data
features_training_new = pd.concat([features_training_data, features_validation_data], axis=0)
tracks_training_new = pd.concat([tracks_training_data, tracks_validation_data], axis=0)

classifier = ApproximateNearestNeighbour(best_parameters["l"], best_parameters["n"],
                                         best_parameters["k"], best_parameters["m"])

classifier.train(features_training_new)

test_accuracy = classifier.predict(features_test_data)
print(f"accuracy using on test set: {test_accuracy}\n")
print(f"parameters used: {best_parameters}")


# %%
