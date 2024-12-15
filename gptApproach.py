import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Normalize data
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("Training data shape:", X_train_full.shape)
print("Training labels shape:", y_train_full.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Let's quickly visualize a few samples:
fig, axes = plt.subplots(1, 5, figsize=(10,2))
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
for i, ax in enumerate(axes):
    ax.imshow(X_train_full[i], cmap='gray')
    ax.set_title(class_names[y_train_full[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()


# Flatten the training set (use a subset for faster PCA and t-SNE)
X_train_flat = X_train_full.reshape((X_train_full.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# For computational efficiency in demonstration:
subset_size = 10000
X_subset = X_train_flat[:subset_size]
y_subset = y_train_full[:subset_size]

# PCA to 50 components
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_subset)

print("Explained variance ratio by first 50 components:", np.sum(pca.explained_variance_ratio_))


tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plot t-SNE
plt.figure(figsize=(8,8))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_subset, cmap="tab10", alpha=0.6)
plt.title("t-SNE visualization of a 10k subset of Fashion-MNIST")
plt.colorbar(scatter, ticks=range(10))
plt.show()


kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)

# Evaluate clustering result with known labels
ari = adjusted_rand_score(y_subset, cluster_labels)
ami = adjusted_mutual_info_score(y_subset, cluster_labels)
print("Adjusted Rand Index (ARI):", ari)
print("Adjusted Mutual Info (AMI):", ami)


distances = np.linalg.norm(X_pca - kmeans.cluster_centers_[cluster_labels], axis=1)
outlier_threshold = np.percentile(distances, 99)  # top 1% distance
outliers = np.where(distances > outlier_threshold)[0]
print("Number of potential outliers detected:", len(outliers))


X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat = X_val.reshape((X_val.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

print("Training set:", X_train.shape, "Validation set:", X_val.shape, "Test set:", X_test.shape)


clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_flat.reshape(len(X_train_flat), -1), y_train)
y_pred_val = clf.predict(X_val_flat)
val_acc = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", val_acc)

y_pred_test = clf.predict(X_test_flat)
test_acc = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", test_acc)
