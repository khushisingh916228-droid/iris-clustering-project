"""
Mini Project 3: Iris Flower Clustering Project
Week 3 - Unsupervised Learning (K-Means Clustering)
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ---------- STEP 1: Load Dataset ----------
iris = load_iris()
X = iris.data          # features: sepal length, sepal width, petal length, petal width
y_true = iris.target   # true species labels (0, 1, 2) - only used for comparison

print("Dataset shape:", X.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# ---------- STEP 2: Apply K-Means (k=3) ----------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

print("\nCluster centers:\n", kmeans.cluster_centers_)

# ---------- STEP 3: Reduce to 2D with PCA (for visualization) ----------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance captured by 2 components: {sum(pca.explained_variance_ratio_):.2%}")

# ---------- STEP 4: Visualize Clusters ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="viridis", edgecolor="k")
axes[0].set_title("K-Means Predicted Clusters")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="viridis", edgecolor="k")
axes[1].set_title("True Species Labels")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")

plt.tight_layout()
plt.savefig("iris_clusters.png")
print("\nSaved chart: iris_clusters.png")

# ---------- STEP 5: Compare Predicted Clusters vs True Labels ----------
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (rows = true species, columns = predicted cluster):\n", cm)