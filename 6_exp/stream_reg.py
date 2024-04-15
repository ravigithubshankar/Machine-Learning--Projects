import streamlit as st
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Streamlit app
st.title("Regression Task with Synthetic Dataset")

# Generate synthetic regression dataset
n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000)
n_features = st.sidebar.slider("Number of features", 1, 20, 10)
noise = st.sidebar.slider("Noise level", 0.0, 1.0, 0.1)
random_state = 42

X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)

# Split dataset into training and testing sets
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# Train decision tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)

# Train random forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Train Radial Basis Function Network (RBFN) model
kmeans = KMeans(n_clusters=10, random_state=random_state)
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
st.subheader("Model Performance:")
st.write("Linear Regression MSE:", lr_mse, "R2 Score:", lr_r2)
st.write("Decision Tree MSE:", dt_mse, "R2 Score:", dt_r2,font="60px")
st.write("Random Forest MSE:", rf_mse, "R2 Score:", rf_r2)
st.write("RBFN MSE:", rbfn_mse, "R2 Score:", rbfn_r2)
