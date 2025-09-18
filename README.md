# 📊 User Segmentation via Hierarchical Clustering

This project demonstrates how to segment users into meaningful groups based on behavioral attributes using Hierarchical Clustering. It leverages Python libraries such as SciPy, scikit-learn, and matplotlib to analyze and visualize clustering performance.

## 📁 Files Included

User Segmentation.py: Main script for clustering and analysis

userbehaviour.csv: Dataset containing user attributes

README.md: Project overview

.gitignore, LICENSE: Standard repo files

## 🧠 Methodology

1. Data Preprocessing
   
Load user data from userbehaviour.csv

Normalize features for clustering using StandardScaler from sklearn.preprocessing

2. Hierarchical Clustering
   
Apply Agglomerative Clustering using scipy.cluster.hierarchy and sklearn.cluster.AgglomerativeClustering

Generate dendrograms to visualize cluster formation

Choose optimal number of clusters based on dendrogram cut-off and silhouette score

3. Silhouette Analysis

Evaluate clustering quality using silhouette_score from sklearn.metrics

Helps determine the optimal number of clusters by measuring intra-cluster cohesion vs inter-cluster separation

4. Visualization

Plot dendrograms using scipy.cluster.hierarchy.dendrogram

Display cluster assignments and silhouette scores

## 🛠 Libraries Used
Library	Purpose
scipy	Dendrogram generation and linkage computation

sklearn	Clustering, scaling, and evaluation metrics

matplotlib	Visualizations

pandas	Data manipulation

numpy	Numerical operations

## 🚀 How to Run
pip install pandas numpy matplotlib scipy scikit-learn

python "User Segmentation.py"

## 📌 License
This project is licensed under the MIT License.
