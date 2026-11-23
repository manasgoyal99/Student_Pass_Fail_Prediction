
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project Path
BASE_DIR = Path(__file__).resolve().parent

# Load Trained Model
model_path = BASE_DIR / "models" / "student_pass_fail_model.pkl"
model = joblib.load(model_path)

# Page Configuration
st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    layout="centered"
)

# App Title
st.title(" Student Pass/Fail Prediction")
st.write("Enter student details to predict Pass or Fail.")

# Input Fields
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=6.0)
attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=80.0)
assignment_score = st.number_input("Assignment Score", min_value=0.0, max_value=100.0, value=70.0)
midterm_score = st.number_input("Midterm Score", min_value=0.0, max_value=100.0, value=60.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=8.0)
internet_hours = st.number_input("Internet Hours", min_value=0.0, max_value=24.0, value=3.0)

# Prediction Button
if st.button("Predict Result"):

    student_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "assignment_score": assignment_score,
        "midterm_score": midterm_score,
        "sleep_hours": sleep_hours,
        "internet_hours": internet_hours
    }])

    prediction = model.predict(student_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("PASS ")
    else:
        st.error("FAIL ")
