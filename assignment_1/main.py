import numpy
import pandas as pd

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


#%%
