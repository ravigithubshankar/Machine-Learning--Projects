import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Streamlit app
st.title("Classification Task with Synthetic Dataset")

# Generate synthetic classification dataset
n_samples = st.sidebar.slider("Number of samples", 100, 2000, 1000)
n_features = st.sidebar.slider("Number of features", 1, 20, 10)
n_classes = 2  # Binary classification
random_state = 42

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=random_state)

# Split dataset into training and testing sets
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model Selection
model_type = st.sidebar.radio("Select Model Type", ("Linear", "Non-linear"))

if model_type == "Linear":
    # Train and evaluate logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred, output_dict=True)
    lr_df = pd.DataFrame(lr_report).transpose()
    
    st.subheader("Logistic Regression Model Performance:")
    st.write("Accuracy:", lr_accuracy)
    st.write("Classification Report:")
    st.write(lr_df)
    
elif model_type == "Non-linear":
    # Train and evaluate MLP classifier (RBFN) model
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam', max_iter=1000, random_state=random_state)
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    mlp_report = classification_report(y_test, mlp_pred, output_dict=True)
    mlp_df = pd.DataFrame(mlp_report).transpose()
    
    st.subheader("MLP Classifier (RBFN) Model Performance:")
    st.write("Accuracy:", mlp_accuracy)
    st.write("Classification Report:")
    st.write(mlp_df)
