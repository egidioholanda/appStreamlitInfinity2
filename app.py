import streamlit as st

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Iris Classification")
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
# Load the iris dataset
iris = datasets.load_iris()

# Create a form to collect the user's input
with st.form("iris_form"):
    sepal_length = st.number_input("Sepal length (cm)")
    sepal_width = st.number_input("Sepal width (cm)")
    petal_length = st.number_input("Petal length (cm)")
    petal_width = st.number_input("Petal width (cm)")
    submit = st.form_submit_button("Predict")

# If the user clicks the submit button, predict the class of the iris flower
if submit:
    # Create a data frame with the user's input
    user_input = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width]
    })

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict the class of the user's input
    prediction = model.predict(user_input)

    # Display the prediction and an image of the corresponding flower
    st.write("Predicted class:", prediction)
    st.image(iris.data[prediction][0], caption=iris.target_names[prediction])
