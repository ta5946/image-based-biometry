import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Read similarity matrix and file names
distance_matrix = pd.read_csv("lbp_distance_matrix_6.csv", header=None).to_numpy()
file_names = os.listdir("./IBB_A2_Data/")

# Fill the bottom left triangle
lower_triangle_indices = np.tril_indices_from(distance_matrix, k=-1)
distance_matrix[lower_triangle_indices] = distance_matrix.T[lower_triangle_indices]
n_samples = distance_matrix.shape[0]

# Plot heatmap
ticks = np.arange(0, 8, n_samples)
plt.figure(figsize=(12, 10))
sns.heatmap(distance_matrix, cmap="viridis", cbar_kws={"label": "HDBIF distance"}, square=True)
plt.title(f"Similarity matrix for N={n_samples} samples")
plt.xlabel("Iris index")
plt.ylabel("Iris index")
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.show()

# Collect genuine and impostor scores
genuine_scores = []
impostor_scores = []
for i1, f1 in enumerate(file_names):
    for i2, f2 in enumerate(file_names):
        if i2 < i1:
            continue
        score = distance_matrix[i1, i2]
        if f1.split('_')[0:2] == f2.split('_')[0:2]:  # matches to the second underscore
            genuine_scores.append(score)
        else:
            impostor_scores.append(score)
genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

# Calculate EER score
labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
distance_scores = np.concatenate([genuine_scores, impostor_scores])
similarity_scores = 1 - distance_scores  # convert distance to similarity
fpr, tpr, thresholds = roc_curve(labels, similarity_scores)
fnr = 1 - tpr
eer_threshold = thresholds[np.argmin(np.absolute(fnr - fpr))]
eer = fpr[np.argmin(np.absolute(fnr - fpr))]
print(f"Equal error rate {eer * 100:.2f}% at similarity threshold {eer_threshold:.2f}")
