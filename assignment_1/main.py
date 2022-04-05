import numpy as np
import pandas as pd

from math import sqrt


# creates a random matrix with shape n x d
def create_random_matrix(n, d):
    # random matrix construction using the method introduced by Achlioptas
    return np.random.default_rng().choice([+1, 0, -1], size=(n, d), p=[1 / 6, 2 / 3, 1 / 6]) * sqrt(3)


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = create_random_matrix(self.hash_size, inp_dimensions)

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
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))

    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label

    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))


# import and filter data
tracks = pd.read_csv("fma_metadata/tracks.csv", index_col=0, header=[0, 1])
tracks = tracks[tracks["set"]["subset"] == "small"]

# split tracks data to later extract features by track id
# track id's are embedded in the index column because we imported the data with parameter index_col=0

tracks_training_data = tracks[tracks["set"]["split"] == "training"]
tracks_validation_data = tracks[tracks["set"]["split"] == "validation"]
tracks_test_data = tracks[tracks["set"]["split"] == "test"]

print(f"tracks training data shape: {tracks_training_data.shape}")
print(f"tracks validation data shape: {tracks_training_data.shape}")
print(f"tracks test data shape: {tracks_training_data.shape}")
print(f"available genres:\n", set(tracks_training_data['track']['genre_top']), "\n")

features = pd.read_csv("fma_metadata/features.csv", index_col=0, header=[0, 1, 2])

features_training_data = features.filter(items=tracks_training_data.index, axis=0)
features_validation_data = features.filter(items=tracks_validation_data.index, axis=0)
features_test_data = features.filter(items=tracks_test_data.index, axis=0)

print(f"features training data shape: {features_training_data.shape}")
print(f"features validation data shape: {features_validation_data.shape}")
print(f"features test data shape: {features_test_data.shape}", "\n")

lsh = LSH(10, 100, features.shape[1])

for index, track in zip(features_training_data.index, features_training_data.values):
    lsh[track] = index

# needs to be put into a for loop for each validation track --- start

track_index = 4

validation_track_id, validation_track = features_validation_data.index[track_index], features_validation_data.iloc[
    track_index]

# neighbours of track with given track_index
neighbour_ids = lsh[validation_track]
neighbours = features_training_data.filter(neighbour_ids, axis=0)

# distance function can be euclidian distance or cosine distance
distance_function = np.linalg.norm

# calculate distances between neighbours and validation track with specific track_index
distances = []
for i in range(len(neighbours)):
    distances.append((neighbour_ids[i], (distance_function(neighbours.iloc[i] - validation_track))))

distances.sort(key=lambda x: x[1])
nearest_neighbour_ids = np.array(distances)[:, 0]

print(f"the ten nearest neighbours of track {validation_track_id} are: {nearest_neighbour_ids[:11]}")

# needs to be put into a for loop for each validation track --- end


# %%
