# ============================================================
# train_ann.py  —  Tasks 3, 4, 5, 6, 7, 8
# Student Performance ANN Trainer
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────────────────────
# TASK 3 — Data Preprocessing
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  TASK 3: Data Preprocessing")
print("=" * 55)

df = pd.read_excel("dataset.xlsx")

# Separate features and target
X = df[['attendance', 'assignment', 'quiz', 'mid', 'study_hours']]
y = df['result']

print(f"\nTotal samples   : {len(df)}")
print(f"Features (X)    : {X.shape[1]} columns")
print(f"Target (y)      : result (0=Fail, 1=Pass)")

# Train/test split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing  samples: {len(X_test)}")

# Feature scaling with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n✅ Scaling applied using StandardScaler")
print("""
  WHY SCALING IS REQUIRED IN ANN:
  ANN uses gradient-based optimisation. If features have very
  different ranges (e.g. attendance 0–100 vs study_hours 0–20),
  large-valued features dominate the weight updates and learning
  slows or diverges. StandardScaler brings every feature to
  mean=0, std=1, so all inputs contribute equally to training.
""")

# ─────────────────────────────────────────────────────────────
# TASK 4 — Build ANN Model
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  TASK 4: Build ANN Model (MLPClassifier)")
print("=" * 55)

# ANN architecture:
#   Input layer  → 5 neurons (one per feature)
#   Hidden layer 1 → 32 neurons, ReLU activation
#   Hidden layer 2 → 16 neurons, ReLU activation
#   Output layer → 1 neuron  (sigmoid → binary class probability)

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),   # 2 hidden layers
    activation='relu',             # ReLU for hidden layers
    solver='adam',                 # adaptive moment estimation
    max_iter=500,
    random_state=42,
    verbose=True,
)

print("""
  ANN CONCEPTS:
  ┌─────────────────────────────────────────────────────┐
  │  NEURONS       — Each neuron receives inputs,        │
  │                  applies weights+bias, passes result │
  │                  through an activation function.     │
  │                                                      │
  │  ACTIVATION    — A non-linear function applied to    │
  │  FUNCTION        each neuron's output:               │
  │                  • ReLU  : max(0, x) — fast, common  │
  │                  • Sigmoid: squashes to (0,1)        │
  │                  Without activations the network is  │
  │                  just linear regression.             │
  │                                                      │
  │  HIDDEN LAYERS — Layers between input and output.    │
  │                  They learn intermediate patterns.   │
  │                  More layers → more complex patterns │
  │                  (but risk over-fitting).            │
  └─────────────────────────────────────────────────────┘

  Our Architecture:
    Input  →  [32 ReLU]  →  [16 ReLU]  →  Output (softmax)
    (5 features)  Hidden 1     Hidden 2    (Pass / Fail)
""")

# ─────────────────────────────────────────────────────────────
# TASK 5 — Train the Model
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  TASK 5: Training the ANN")
print("=" * 55)

model.fit(X_train_scaled, y_train)

print(f"\n✅ Training complete!")
print(f"   Iterations run   : {model.n_iter_}")
print(f"   Final loss value : {model.loss_:.4f}")
print(f"   Converged        : {model.n_iter_ < model.max_iter}")

# ─────────────────────────────────────────────────────────────
# TASK 6 — Evaluate the Model
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 6: Model Evaluation")
print("=" * 55)

y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])

print(f"\n🎯 Accuracy: {acc * 100:.2f}%")
print("""
  WHAT ACCURACY MEANS:
  Accuracy = correct predictions / total predictions.
  If accuracy is 85%, the model correctly classified 85 out
  of every 100 students.  It does NOT tell us WHERE it fails.
""")

print("📊 Confusion Matrix:")
print(cm)
print("""
         Predicted: Fail   Predicted: Pass
  Actual Fail   [ TN ]         [ FP ]
  Actual Pass   [ FN ]         [ TP ]

  TN = Correctly predicted Fail
  TP = Correctly predicted Pass
  FP = Predicted Pass but actually Fail  (Type I error)
  FN = Predicted Fail but actually Pass  (Type II error)
""")

print("📋 Classification Report:")
print(report)

print("""
  MISTAKES THE MODEL MAY MAKE:
  • FP (False Positive) — student is predicted to Pass but Fails.
    This is dangerous in academic planning; student may not get
    the support they need.
  • FN (False Negative) — student is predicted to Fail but Passes.
    This is less harmful but wastes intervention resources.
""")

# Plot confusion matrix heatmap (Bonus)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Fail", "Pass"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — ANN Student Evaluator")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("✅ Confusion matrix saved → confusion_matrix.png")

# ─────────────────────────────────────────────────────────────
# TASK 7 — Evaluation Function
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 7: evaluate_student() Function")
print("=" * 55)

def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict whether a student will Pass or Fail.

    Parameters
    ----------
    attendance   : int/float  (0–100 %)
    assignment   : int/float  (0–100 marks)
    quiz         : int/float  (0–100 marks)
    mid          : int/float  (0–100 marks)
    study_hours  : int/float  (hours per week)

    Returns
    -------
    dict with keys:
        result      — 0 (Fail) or 1 (Pass)
        label       — "Pass" or "Fail"
        probability — confidence (0–1)
    """
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    features_scaled = scaler.transform(features)
    prediction  = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][prediction]

    return {
        "result"     : int(prediction),
        "label"      : "Pass" if prediction == 1 else "Fail",
        "probability": round(float(probability) * 100, 2),
    }

# Quick test
sample = evaluate_student(
    attendance=85, assignment=80, quiz=75, mid=70, study_hours=10
)
print(f"\nSample prediction → {sample}")

# ─────────────────────────────────────────────────────────────
# TASK 8 — Save Model & Scaler
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 8: Saving Model & Scaler")
print("=" * 55)

joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("""
✅ model.joblib  → saved
✅ scaler.joblib → saved

  WHY BOTH MUST BE SAVED:
  • model.joblib  — contains the trained ANN weights and biases.
    Without it you cannot make predictions.
  • scaler.joblib — stores the mean and std values computed on the
    TRAINING data. Every new input MUST be scaled with the SAME
    scaler used during training. If you scale differently (or skip
    scaling), the model receives out-of-distribution inputs and
    produces garbage predictions.
""")

print("=" * 55)
print("  All Tasks Completed Successfully!")
print("=" * 55)
