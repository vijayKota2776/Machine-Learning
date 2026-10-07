import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Generate Synthetic Customer Dataset manually to ensure realistic values
np.random.seed(42)
n_samples = 300

# Create 4 distinct customer personas
# Persona 1: High Income, High Spending (VIPs)
inc_1 = np.random.normal(90, 15, 75)
spd_1 = np.random.normal(85, 10, 75)

# Persona 2: High Income, Low Spending (Careful Savers)
inc_2 = np.random.normal(90, 15, 75)
spd_2 = np.random.normal(25, 10, 75)

# Persona 3: Low Income, High Spending (Impulsive Buyers)
inc_3 = np.random.normal(30, 10, 75)
spd_3 = np.random.normal(80, 10, 75)

# Persona 4: Low Income, Low Spending (Budget Conscious)
inc_4 = np.random.normal(30, 10, 75)
spd_4 = np.random.normal(25, 10, 75)

income = np.concatenate([inc_1, inc_2, inc_3, inc_4])
spending = np.concatenate([spd_1, spd_2, spd_3, spd_4])

# Clip bounds just in case
income = np.clip(income, 10, 150)
spending = np.clip(spending, 1, 100)

X = np.column_stack((income, spending))
df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

# 2. Determine optimal clusters using the Elbow Method
wcss = []
k_range = range(1, 11)
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.savefig('elbow_method.png')
print("Saved elbow_method.png")

# 3. Apply K-Means with optimal k (k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)
df['Cluster'] = y_kmeans

# 4. Visualize the Clusters
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'purple']
labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

for i in range(optimal_k):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=50, c=colors[i], label=labels[i], alpha=0.6)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', marker='*', edgecolor='black', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('clusters.png')
print("Saved clusters.png")

# Output means
cluster_means = df.groupby('Cluster').mean()
print("\nCluster Means:")
print(cluster_means)
