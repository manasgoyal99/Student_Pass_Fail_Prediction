
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Load Dataset
df = pd.read_csv(
    BASE_DIR / "dataset" / "student_performance_dataset.csv"
)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


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


# Check Target
print("\nTarget Distribution:")
print(y.value_counts())

print("\nMissing Target Values:")
print(y.isnull().sum())


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Display Training and Testing Shapes
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)


print("\nPreprocessing Completed Successfully!")
