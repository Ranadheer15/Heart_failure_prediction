import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open("models/heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("❤️ Heart Failure Prediction App")

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.5, 1.0)
slope = st.selectbox("Slope of Peak Exercise", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, 1 if sex=="Male" else 0, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ High chance of Heart Failure")
    else:
        st.success("✅ Low chance of Heart Failure")
