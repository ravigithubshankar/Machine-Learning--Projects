import numpy as np
import matplotlib.pyplot as plt

n=200 # samples per center
centers= [ [10,5], [-2,4], [13,-25], [11,20], [15,-30], [3,-2], ] # centers
#centers= [ [10,5], [-2,4], [11,20], [15,-34], ]
dataset=np.zeros((0,3))
sigma=2

for i in range(len(centers)):
    correlation=np.random.rand()
    center=centers[i]
    cluster=np.random.multivariate_normal(center, [[sigma, correlation],[correlation, sigma]], n)
    label=np.zeros((n,1))+i
    cluster=np.hstack([cluster,label])
    dataset=np.vstack([dataset,cluster])
print(dataset.shape)


plt.scatter(dataset[:,0],dataset[:,1])
filename="2d_complex.csv"
np.savetxt(filename,dataset,delimiter=",",fmt='%f')



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Generate synthetic dataset
n = 200  # samples per center
centers = [[10,5], [-2,4], [13,-25], [11,20], [15,-30], [3,-2]]  # centers
sigma = 2
dataset = np.zeros((0,3))

for i in range(len(centers)):
    correlation = np.random.rand()
    center = centers[i]
    cluster = np.random.multivariate_normal(center, [[sigma, correlation],[correlation, sigma]], n)
    label = np.zeros((n,1)) + i
    cluster = np.hstack([cluster, label])
    dataset = np.vstack([dataset, cluster])

# Split dataset into features and labels
X = dataset[:, :2]
y = dataset[:, 2]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement KNN algorithm
k_values = [1, 5, 20, len(X_train)]  # different values of K
plt.figure(figsize=(12, 8))

for i, k in enumerate(k_values, 1):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    h = 0.1  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.subplot(2, 2, i)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['red', 'blue', 'green', 'yellow', 'purple', 'orange']))
    plt.title(f'KNN Decision Boundary (K={k}) - Accuracy: {accuracy:.2f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()

plt.tight_layout()
plt.show()
