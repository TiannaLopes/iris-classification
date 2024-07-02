# ui.py

import streamlit as st
import numpy as np
from iris_classification import train_model
from sklearn.datasets import load_iris

# Train the model and get the scaler
model, scaler, accuracy, report = train_model()

# Load the Iris dataset
iris = load_iris()

# Title and description
st.title("Iris Flower Classification")
st.write("This is a simple machine learning project to classify iris flowers into three species based on their features.")

# Show accuracy and classification report
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.text(report)

# User inputs for the features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), float(iris.data[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), float(iris.data[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), float(iris.data[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), float(iris.data[:, 3].mean()))

# Predict button
if st.sidebar.button("Classify"):
    # Scale the input features
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_features_scaled = scaler.transform(input_features)
    
    # Make a prediction
    prediction = model.predict(input_features_scaled)
    predicted_species = iris.target_names[prediction][0]
    
    st.write(f"The predicted species is: **{predicted_species}**")
