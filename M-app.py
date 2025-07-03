import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model", "mental_health_model.pkl")
encoder_path = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
features_path = os.path.join(BASE_DIR, "model", "model_features.pkl")

model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)
model_features = joblib.load(features_path)

st.set_page_config(page_title="BrainWave", page_icon="🧠")
st.title("🧠 BrainWave: Mental Health Risk Prediction")
st.markdown("Provide your lifestyle and psychological details to assess your risk level.")

age = st.slider("Age", 18, 65, 20, step=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
employment_status = st.selectbox("Employment Status", ["Employed", "Student", "Self-employed", "Unemployed"])
work_environment = st.selectbox("Work Environment", ["On-site", "Remote", "Hybrid"])
mental_health_history = st.selectbox("Mental Health History", ["Yes", "No"])
seeks_treatment = st.selectbox("Currently Seeking Treatment?", ["Yes", "No"])
stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
sleep_hours = st.slider("Average Sleep Hours", 0.0, 12.0, 7.0, step=0.5)
physical_activity_days = st.slider("Physical Activity Days per Week", 0, 7, 3)
depression_score = st.slider("Depression Score (0-30)", 0, 30, 5)
anxiety_score = st.slider("Anxiety Score (0-20)", 0, 20, 5, step=1)
social_support_score = st.slider("Social Support Score (0-100)", 0, 100, 50, step=1)
productivity_score = st.slider("Productivity Score (0-100)", 0.0, 100.0, 50.0, step=0.1)

input_data = {
    'age': age,
    'stress_level': stress_level,
    'sleep_hours': sleep_hours,
    'physical_activity_days': physical_activity_days,
    'depression_score': depression_score,
    'anxiety_score': anxiety_score,
    'social_support_score': social_support_score,
    'productivity_score': productivity_score,
    'gender_Male': 1 if gender == "Male" else 0,
    'gender_Other': 1 if gender == "Other" else 0,
    'employment_status_Self-employed': 1 if employment_status == "Self-employed" else 0,
    'employment_status_Student': 1 if employment_status == "Student" else 0,
    'employment_status_Unemployed': 1 if employment_status == "Unemployed" else 0,
    'work_environment_Hybrid': 1 if work_environment == "Hybrid" else 0,
    'work_environment_Remote': 1 if work_environment == "Remote" else 0,
    'mental_health_history_Yes': 1 if mental_health_history == "Yes" else 0,
    'seeks_treatment_Yes': 1 if seeks_treatment == "Yes" else 0,
}

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=model_features, fill_value=0)

if st.button("Predict Risk Level"):
    prediction = model.predict(input_df)[0]
    result = label_encoder.inverse_transform([prediction])[0]

    st.success(f"🧠 Predicted Mental Health Risk Level: **{result}**")

    if result == "High":
        st.markdown(f"""
        <div style='background-color:#ffe6e6; padding:20px; border-radius:10px; border-left:6px solid red;'>
            <h3 style='color:red;'>☣️ HIGH RISK DETECTED</h3>
            <p style='color:#333; font-size:16px;'>
                Your responses indicate a <strong>high risk of mental health challenges</strong>.<br>
                Please <strong>consult a certified mental health professional</strong> as soon as possible.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif result == "Medium":
        st.warning("🧩 Medium risk detected. Monitor and take preventive actions.")

    else:
        st.markdown(f"""
        <div style='background-color:#e6ffed; padding:20px; border-radius:10px; border-left:6px solid green;'>
            <h3 style='color:green;'>🧘 LOW RISK DETECTED</h3>
            <p style='color:#333; font-size:16px;'>
                You're showing signs of strong mental well-being. Keep practicing good habits and stay mindful.
            </p>
        </div>
        """, unsafe_allow_html=True)