import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load high-dimensional dataset
# The Breast Cancer dataset has 30 features
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names

print(f"Original Dataset Dimensions: {X.shape[0]} samples, {X.shape[1]} features.")

# 2. Standardize the data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Report Variance Retained
explained_variance = pca.explained_variance_ratio_
total_variance = np.sum(explained_variance)

print(f"\nVariance explained by Component 1: {explained_variance[0]*100:.2f}%")
print(f"Variance explained by Component 2: {explained_variance[1]*100:.2f}%")
print(f"Total Variance Retained: {total_variance*100:.2f}%")

# 5. Visualize the reduced data
plt.figure(figsize=(10, 7))
colors = ['red', 'blue']
for i, color, target_name in zip([0, 1], colors, target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.6, label=target_name, s=40)

plt.title('2D PCA Visualization of 30-Dimensional Breast Cancer Dataset')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.2f}%)')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('pca_visualization.png')
print("\nSaved pca_visualization.png")
