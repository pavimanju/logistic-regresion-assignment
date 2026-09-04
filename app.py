
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
sex_male = st.selectbox("Sex", ["Female", "Male"])
embarked_q = st.selectbox("Embarked Q", [0, 1])
embarked_s = st.selectbox("Embarked S", [0, 1])

# Convert sex to numerical value
sex_value = 1 if sex_male == "Male" else 0

# Create input dataframe
input_data = pd.DataFrame(0, index=[0], columns=feature_names)

# Fill available features
values = {
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Sex_male": sex_value,
    "Embarked_Q": embarked_q,
    "Embarked_S": embarked_s
}

for feature, value in values.items():
    if feature in input_data.columns:
        input_data[feature] = value

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("The passenger is predicted to SURVIVE.")
    else:
        st.error("The passenger is predicted NOT TO SURVIVE.")

    st.write("Survival Probability:", round(probability * 100, 2), "%")
