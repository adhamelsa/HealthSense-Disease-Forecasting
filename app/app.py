import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config("HealthSense", layout="centered")
st.title("🩺 HealthSense Disease Forecasting")

# اختيار المرض
disease = st.selectbox("اختر المرض", ["-", "السكري", "مرض القلب", "سرطان الرئة"])
if disease != "-":
    params = {}
    if disease == "السكري":
        for col in ["Pregnancies","Glucose","BloodPressure","BMI","Age"]:
            params[col] = st.number_input(col)
        model = joblib.load("models/rf_diabetes.pkl")
        data = pd.DataFrame([params])
    # إضافات لأمراض أخرى مشابهة...

    if st.button("تنبأ"):
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0]
        st.success(f"التنبؤ: {'إيجابي' if pred else 'سلبي'}")
        fig, ax = plt.subplots()
        ax.barh(model.classes_, proba)
        st.pyplot(fig)

