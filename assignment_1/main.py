import numpy as np
import pandas as pd

from math import sqrt


# creates a random matrix with shape n x d
def create_random_matrix(n, d):
    # random matrix construction using the method introduced by Achlioptas
    return np.random.default_rng().choice([+1, 0, -1], size=(n, d), p=[1 / 6, 2 / 3, 1 / 6]) * sqrt(3)


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