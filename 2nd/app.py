import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Define the Streamlit app
def main():
    st.title("KNN Decision Boundary Visualization")

    # File uploader for synthetic dataset
    st.sidebar.title("Upload Synthetic Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the dataset
        dataset = np.loadtxt(uploaded_file, delimiter=',')

        # Split dataset into features and labels
        X = dataset[:, :2]
        y = dataset[:, 2]

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Implement KNN algorithm
        k_values = [1, 5, 20, len(X_train)]  # different values of K

        # Plot decision boundaries for different K values
        plot_decision_boundaries(X, y, X_train, y_train, k_values)


    
# Function to plot decision boundaries
def plot_decision_boundaries(X, y, X_train, y_train, k_values):
    plt.figure(figsize=(12, 8))

    for i, k in enumerate(k_values, 1):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        
        h = 0.1  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.subplot(2, 2, i)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(['red', 'blue', 'green', 'yellow', 'purple', 'orange']), marker='o', label='Train')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue', 'green', 'yellow', 'purple', 'orange']), marker='x', label='Test')
        plt.title(f'KNN Decision Boundary (K={k})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar()
        plt.legend()

    plt.tight_layout()
    st.pyplot()


# Run the Streamlit app
if __name__ == "__main__":
    main()
