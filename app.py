import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("models/random_forest_model.pkl")  # use your best model

st.set_page_config(page_title="Heart Failure Prediction App", layout="centered")

# Title
st.title("🩺 Heart Failure Prediction App")
st.write("This app predicts the **risk of heart disease** based on medical parameters.")

# User Inputs
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0,1])
restecg = st.selectbox("Resting ECG (restecg)", [0,1,2])
thalach = st.slider("Maximum Heart Rate (thalach)", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0,1])
oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST segment (slope)", [0,1,2])
ca = st.selectbox("Number of major vessels (ca)", [0,1,2,3,4])
thal = st.selectbox("Thalassemia (thal)", [0,1,2,3])

# Convert categorical inputs
sex = 1 if sex == "Male" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {prob:.2f})")
