# Segmenting users based on their behavior using Hierarchical clustering
# Evaluating the optimal number of clusters using Silhouette Score

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Load your data
df = pd.read_csv('C:\\Users\\akjee\\Documents\\AI\\ML\\Unsupervised Learning\\userbehaviour.csv')
print(df.head())
print(df.describe())
df.dropna(inplace=True)

features = df.drop(['userid', 'Status'], axis=1).columns  # Adjust based on your dataset
X = df[features]  # Drop non-numeric or identifier columns

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot dendrogram using original scaled data
plt.figure(figsize=(10, 7))
linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.show()

# Find optimal number of clusters using Silhouette Score
sil_scores = []
cluster_range = range(2, 11)
for n_clusters in cluster_range:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    sil_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, sil_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Choose optimal number of clusters
optimal_clusters = cluster_range[np.argmax(sil_scores)]
print(f'Optimal number of clusters: {optimal_clusters}')

# Fit final clustering model
final_clusterer = AgglomerativeClustering(n_clusters=optimal_clusters)
df['Cluster'] = final_clusterer.fit_predict(X_scaled)

# Visualize clusters in PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set2')
plt.title('User Segments (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# Display cluster-wise feature means
cluster_summary = df.groupby('Cluster')[features].mean()
print("\nCluster-wise Feature Averages:")
print(cluster_summary)

# Display cluster counts
print("\nCluster Counts:")
print(df['Cluster'].value_counts())
