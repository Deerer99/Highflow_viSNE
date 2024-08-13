import os
import random
import flowio
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set the random seed
seed = 100
random.seed(seed)

# Define the folder path and list of selected files
folder_path = "C:/Users/bruno/OneDrive/Desktop/Programmer/Programmer/Professional/fcm_project/fcm_data/fcm_field_data"
file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("A") and file.endswith(".fcs")]


# Define the number of subsamples
subsampling_number = 300

# Function for random subsampling
def random_subsampling(flow_frame, num_samples=subsampling_number):
    num_events = len(flow_frame)
    if num_samples >= num_events:
        return flow_frame
    else:
        sampled_indices = random.sample(range(num_events), num_samples)
        return flow_frame.iloc[sampled_indices, :]

# Define the transformation parameter (b)
b = 150

# Create a list to store the subsampled data
fcs_data_list = {}

for file in file_list:
    with flowio.FlowData(file) as f:
        fcs_data = pd.DataFrame(f.events)
        subsampled_data = random_subsampling(fcs_data, num_samples=subsampling_number)
        fcs_data_list[os.path.basename(file)] = subsampled_data

# Combine data into a matrix
combined_matrix = pd.concat(fcs_data_list.values(), ignore_index=True)

# Perform t-SNE with defined perplexity
perplexity_value = 500

tsne_result = TSNE(n_components=2, perplexity=perplexity_value, verbose=True).fit_transform(combined_matrix)

# Create a DataFrame from the t-SNE result
tsne_df = pd.DataFrame(tsne_result, columns=["Dimension 1", "Dimension 2"])

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(tsne_df["Dimension 1"], tsne_df["Dimension 2"], s=0.25)
plt.title(f"viSNE Plot (Perplexity = {perplexity_value}, seed = {seed}, transformation = {b})")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
