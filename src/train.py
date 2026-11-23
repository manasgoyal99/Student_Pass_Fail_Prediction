
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Project Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Dataset
df = pd.read_csv(
    BASE_DIR / "dataset" / "student_performance_dataset.csv"
)


# Feature Selection
X = df[
    [
        "study_hours",
        "attendance",
        "assignment_score",
        "midterm_score",
        "sleep_hours",
        "internet_hours"
    ]
]


# Target Creation
y = df["grade"].map({
    "A+": 1,
    "A": 1,
    "B": 1,
    "C": 1,
    "D": 0,
    "F": 0
})


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create ML Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])


# Train Model
model.fit(X_train, y_train)

print("Model Training Completed!")

# Save Trained Model
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "student_pass_fail_model.pkl"

joblib.dump(model, model_path)

print("\nModel Saved Successfully!")
print("Model Path:", model_path)


# Prediction
y_pred = model.predict(X_test)


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
