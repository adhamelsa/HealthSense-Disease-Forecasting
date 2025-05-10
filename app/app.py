import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config("HealthSense", layout="centered")
st.title("🩺 HealthSense Disease Forecasting")

# Select disease
disease = st.selectbox("Select a disease", ["-", "Diabetes", "Heart Disease", "Lung Cancer"])

if disease != "-":
    params = {}

    if disease == "Diabetes":
        for col in ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]:
            params[col] = st.number_input(f"Enter {col}", min_value=0.0)
        model = joblib.load("models/rf_diabetes.pkl")
        data = pd.DataFrame([params])

    # Add more disease input conditions here (e.g., Heart Disease, Lung Cancer)

    if st.button("Predict"):
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0]
        st.success(f"Prediction: {'Positive' if pred else 'Negative'}")
        fig, ax = plt.subplots()
        ax.barh(model.classes_, proba)
        ax.set_xlabel("Probability")
        st.pyplot(fig)
