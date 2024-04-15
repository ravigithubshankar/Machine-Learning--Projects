import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import cv2
import tensorflow as tf

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Split dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Feature extraction functions
def extract_hog(image):
    gray = cv2.cvtColor(image.reshape(32, 32, 3), cv2.COLOR_RGB2GRAY)
    # Compute gradients in x and y directions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    # Compute magnitude and angle of gradients
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    # Quantize angles into 9 bins
    bins = np.int32((angle / 20) % 9)
    # Calculate histogram for each cell (4x4 cells)
    hist = np.zeros((4, 4, 9), dtype=np.float32)
    cell_size = 8
    for i in range(4):
        for j in range(4):
            cell_magnitude = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_angle = bins[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            hist[i, j] = np.bincount(cell_angle.ravel(), cell_magnitude.ravel(), minlength=9)
    # Flatten histogram
    return hist.flatten()

def extract_lbp(image):
    gray = cv2.cvtColor(image.reshape(32, 32, 3), cv2.COLOR_RGB2GRAY)
    lbp = np.zeros((32, 32), dtype=np.uint8)
    for i in range(1, 31):
        for j in range(1, 31):
            center = gray[i, j]
            code = 0
            code |= (gray[i - 1, j - 1] > center) << 7
            code |= (gray[i - 1, j] > center) << 6
            code |= (gray[i - 1, j + 1] > center) << 5
            code |= (gray[i, j + 1] > center) << 4
            code |= (gray[i + 1, j + 1] > center) << 3
            code |= (gray[i + 1, j] > center) << 2
            code |= (gray[i + 1, j - 1] > center) << 1
            code |= (gray[i, j - 1] > center) << 0
            lbp[i, j] = code
    # Calculate histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 256), range=(0, 256))
    return hist.astype(np.float32)

# Extract features from images
X_train_hog = np.array([extract_hog(image) for image in X_train])
X_val_hog = np.array([extract_hog(image) for image in X_val])
X_test_hog = np.array([extract_hog(image) for image in X_test])

X_train_lbp = np.array([extract_lbp(image) for image in X_train])
X_val_lbp = np.array([extract_lbp(image) for image in X_val])
X_test_lbp = np.array([extract_lbp(image) for image in X_test])

# Define models
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate models for each feature extraction method
for feature_name, X_train_features, X_val_features, X_test_features in [
    ("HoG", X_train_hog, X_val_hog, X_test_hog),
    ("LBP", X_train_lbp, X_val_lbp, X_test_lbp)
]:
    print(f"Feature extraction method: {feature_name}")
    for model_name, model in models.items():
        pipeline = make_pipeline(StandardScaler(), model)
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train_features, y_train.ravel(), cv=5)
        print(f"Model: {model_name}")
        print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} (± {np.std(cv_scores):.4f})")
        # Fit model
        pipeline.fit(X_train_features, y_train.ravel())
        # Predictions
        y_pred = pipeline.predict(X_val_features)
        # Evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\n")

# Once you have selected the best performing model and feature extraction method on validation set, 
# you can evaluate its performance on the test set.
