import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Generate synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Train and evaluate decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Train and evaluate support vector machine (SVM) model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Train and evaluate Radial Basis Function Network (RBFN) model
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_
distances = euclidean_distances(X_train, centers)
sigma = np.mean(np.max(distances, axis=1))

rbfn_model = MLPClassifier(hidden_layer_sizes=(len(centers),), activation='logistic', solver='lbfgs', max_iter=1000)
rbfn_model.fit(distances, y_train)
rbfn_pred = rbfn_model.predict(euclidean_distances(X_test, centers))
rbfn_accuracy = accuracy_score(y_test, rbfn_pred)

# Display results
print("Logistic Regression Accuracy:", lr_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("RBFN Accuracy:", rbfn_accuracy)

from sklearn.metrics import classification_report

# Evaluate logistic regression model
lr_report = classification_report(y_test, lr_pred)
print("Logistic Regression Classification Report:")
print(lr_report)

# Evaluate decision tree model
dt_report = classification_report(y_test, dt_pred)
print("Decision Tree Classification Report:")
print(dt_report)

# Evaluate support vector machine (SVM) model
svm_report = classification_report(y_test, svm_pred)
print("SVM Classification Report:")
print(svm_report)

# Evaluate Radial Basis Function Network (RBFN) model
rbfn_report = classification_report(y_test, rbfn_pred)
print("RBFN Classification Report:")
print(rbfn_report)



print(f"------------For the regression task:---------------------")
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import cdist

# Generate synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# Train and evaluate decision tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

# Train and evaluate random forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Train and evaluate Radial Basis Function Network (RBFN) model
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_
distances = cdist(X_train, centers)
sigma = np.mean(np.max(distances, axis=1))

rbfn_model = MLPRegressor(hidden_layer_sizes=(len(centers),), activation='relu', solver='adam', max_iter=1000)
rbfn_model.fit(distances, y_train)
rbfn_pred = rbfn_model.predict(cdist(X_test, centers))
rbfn_mse = mean_squared_error(y_test, rbfn_pred)
rbfn_r2 = r2_score(y_test, rbfn_pred)

# Display results
print("Linear Regression MSE:", lr_mse, "R2 Score:", lr_r2)
print("Decision Tree MSE:", dt_mse, "R2 Score:", dt_r2)
print("Random Forest MSE:", rf_mse, "R2 Score:", rf_r2)
print("RBFN MSE:", rbfn_mse, "R2 Score:", rbfn_r2)

