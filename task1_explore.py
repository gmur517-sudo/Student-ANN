# ============================================================
# TASK 1: Understand the Dataset
# ============================================================

import pandas as pd

# Load the dataset
df = pd.read_excel("dataset.xlsx")

print("=" * 55)
print("       STUDENT PERFORMANCE DATASET - OVERVIEW")
print("=" * 55)

# First 5 rows
print("\n📋 First 5 Rows:")
print(df.head())

# Column names
print("\n📌 Column Names:")
print(df.columns.tolist())

# Shape
print(f"\n📐 Shape of Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Input features and target variable
X = df[['attendance', 'assignment', 'quiz', 'mid', 'study_hours']]
y = df['result']

print("\n✅ Input Features (X):")
print("  attendance, assignment, quiz, mid, study_hours")

print("\n🎯 Target Variable (y):")
print("  result (0 = Fail, 1 = Pass)")

print("\n📊 Target Distribution:")
print(y.value_counts().rename({0: "Fail (0)", 1: "Pass (1)"}))

print("\n📖 What Each Column Represents:")
print("  attendance   → % of classes attended (0–100)")
print("  assignment   → marks scored in assignments (0–100)")
print("  quiz         → marks scored in quizzes (0–100)")
print("  mid          → marks scored in mid-term exam (0–100)")
print("  study_hours  → hours studied per week")
print("  result       → final outcome: 0 = Fail, 1 = Pass")

print("\n🧠 Problem Type: CLASSIFICATION")
print("""  Justification:
  The target variable 'result' has only 2 discrete values (0 or 1).
  We are predicting a category (Pass/Fail), NOT a continuous number.
  Therefore this is a BINARY CLASSIFICATION problem.""")

print("\n📈 Basic Statistics:")
print(df.describe().round(2))
