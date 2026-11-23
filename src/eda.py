import pandas as pd
from pathlib import Path


Base_dir = Path(__file__).resolve().parent.parent
file_path = Base_dir / 'dataset' / 'student_performance_dataset.csv'
df = pd.read_csv(file_path)  

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Dataset Information
print("\nDataset Information:")
df.info()

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Column Names
print("\nColumns:")
print(df.columns.tolist())


# 1. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Check Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 3. Check Data Types
print("\nData Types:")
print(df.dtypes)

# 4. Check Dataset Information
print("\nDataset Info:")
df.info()
