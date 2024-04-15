import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from keras.datasets import fashion_mnist
import cv2

# Step 1: Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Step 2: Feature Extraction (ORB)
def extract_orb_features(images):
    orb = cv2.ORB_create()
    keypoints = []
    descriptors = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None and not np.isnan(des).any():
            descriptors.append(des)
            keypoints.append(kp)
    descriptors = np.vstack(descriptors) if descriptors else None
    return keypoints, descriptors


train_keypoints, train_descriptors = extract_orb_features(X_train)
test_keypoints, test_descriptors = extract_orb_features(X_test)

# Step 3: Quantization (K-means clustering)
def quantize_descriptors(descriptors, n_clusters=100):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(descriptors)
    return kmeans

kmeans = quantize_descriptors(train_descriptors)

# Step 4: Formulate fixed-length feature vectors
def create_feature_vector(keypoints, descriptors, kmeans):
    features = np.zeros((len(keypoints), kmeans.n_clusters))
    for i, descriptor in enumerate(descriptors):
        labels = kmeans.predict(descriptor.reshape(1, -1))
        features[i, :] = np.bincount(labels, minlength=kmeans.n_clusters)
    return features

X_train_features = create_feature_vector(train_keypoints, train_descriptors, kmeans)
X_test_features = create_feature_vector(test_keypoints, test_descriptors, kmeans)

# Step 5: Train traditional ML models
classifiers = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

for name, clf in classifiers.items():
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
