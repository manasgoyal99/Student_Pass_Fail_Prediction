
import joblib
from pathlib import Path

# Project Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Saved Model
model_path = BASE_DIR / "models" / "student_pass_fail_model.pkl"

model = joblib.load(model_path)

print("Model Loaded Successfully!")


# Take Student Input
study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))
assignment_score = float(input("Enter Assignment Score: "))
midterm_score = float(input("Enter Midterm Score: "))
sleep_hours = float(input("Enter Sleep Hours: "))
internet_hours = float(input("Enter Internet Hours: "))


import pandas as pd

student_data = pd.DataFrame([{
    "study_hours": study_hours,
    "attendance": attendance,
    "assignment_score": assignment_score,
    "midterm_score": midterm_score,
    "sleep_hours": sleep_hours,
    "internet_hours": internet_hours
}])

prediction = model.predict(student_data)

# Display Result
print("\nStudent Prediction Result:")

if prediction[0] == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")
