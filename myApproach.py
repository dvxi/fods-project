import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import cv2
import time

# 1. Load the Fashion-MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("Original Train:", X_train_full.shape, y_train_full.shape)
print("Original Test:", X_test.shape, y_test.shape)

# === Resample/Subset the Dataset ===
# For example, use only 20,000 training images instead of 60,000.
subset_train_size = 20000
X_train_full = X_train_full[:subset_train_size]
y_train_full = y_train_full[:subset_train_size]
print("Resampled Train:", X_train_full.shape, y_train_full.shape)

# Normalize pixel values
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Split train into train/val
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, 
                                                  test_size=0.1, random_state=42)
print("Training set:", X_train.shape, "Validation set:", X_val.shape, "Test set:", X_test.shape)

# 2. Create Embeddings Using a More Efficient Model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def preprocess_for_model(img, size=(224,224)):
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img_3ch = np.stack([img_resized]*3, axis=-1)
    return img_3ch

def get_embeddings(model, X, batch_size=256):
    embeddings = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        batch_prepped = np.array([preprocess_for_model(img) for img in batch])
        batch_preprocessed = preprocess_input(batch_prepped)
        batch_emb = model.predict(batch_preprocessed, verbose=0)
        embeddings.append(batch_emb)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

# For demonstration, we can use an even smaller subset (e.g., 5,000) for t-SNE and clustering.
subset_size = 5000
X_subset = X_train[:subset_size]
y_subset = y_train[:subset_size]

start_time = time.time()
X_subset_emb = get_embeddings(base_model, X_subset)
print("Subset embeddings shape:", X_subset_emb.shape)
print("Embedding extraction time (5k subset):", time.time() - start_time, "seconds")

# 3. Dimensionality Reduction before t-SNE
pca_dim = 50
pca = PCA(n_components=pca_dim, random_state=42)
X_subset_pca = pca.fit_transform(X_subset_emb)
print(f"Explained variance by {pca_dim} components: {np.sum(pca.explained_variance_ratio_):.4f}")

start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_subset_pca)
print("t-SNE time:", time.time() - start_time, "seconds")

plt.figure(figsize=(8,8))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_subset, cmap="tab10", alpha=0.6)
plt.title("t-SNE visualization (MobileNetV2 embeddings, 5k subset)")
plt.colorbar(scatter, ticks=range(10))
plt.show()

# 4. Clustering on the PCA-Reduced Embeddings
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_subset_pca)

ari = adjusted_rand_score(y_subset, cluster_labels)
ami = adjusted_mutual_info_score(y_subset, cluster_labels)
print("Adjusted Rand Index (ARI):", ari)
print("Adjusted Mutual Info (AMI):", ami)

# Detecting Outliers
distances = np.linalg.norm(X_subset_pca - kmeans.cluster_centers_[cluster_labels], axis=1)
outlier_threshold = np.percentile(distances, 99)
outliers = np.where(distances > outlier_threshold)[0]
print("Number of potential outliers detected:", len(outliers))

# 5. Extract embeddings for full (resampled) training and test sets
X_train_emb = get_embeddings(base_model, X_train)
X_val_emb = get_embeddings(base_model, X_val)
X_test_emb = get_embeddings(base_model, X_test)

X_train_emb_pca = pca.transform(X_train_emb)
X_val_emb_pca = pca.transform(X_val_emb)
X_test_emb_pca = pca.transform(X_test_emb)

# 6. Classification using Random Forest on embeddings
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_emb_pca, y_train)
y_val_pred = clf.predict(X_val_emb_pca)
val_acc = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_acc)

y_test_pred = clf.predict(X_test_emb_pca)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_acc)
