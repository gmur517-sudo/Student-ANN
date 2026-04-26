# ============================================================
# predict.py  —  Reusable Student Evaluation Function
# ============================================================
# This file loads the pre-trained ANN and scaler from disk
# and exposes evaluate_student() for use by any other module.
# ============================================================

import numpy as np
import joblib

# Load saved artefacts (must exist — run train_ann.py first)
model  = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")


def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict student outcome using the trained ANN.

    Parameters
    ----------
    attendance   : float  — attendance percentage (0–100)
    assignment   : float  — assignment marks     (0–100)
    quiz         : float  — quiz marks           (0–100)
    mid          : float  — mid-term marks       (0–100)
    study_hours  : float  — weekly study hours

    Returns
    -------
    dict
        result      : int   — 0 = Fail, 1 = Pass
        label       : str   — "Pass" or "Fail"
        probability : float — model's confidence in %
        performance : str   — "Low / Medium / High" band (bonus)
    """
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled   = scaler.transform(features)

    prediction  = model.predict(scaled)[0]
    proba_pass  = model.predict_proba(scaled)[0][1]   # probability of Pass

    # Bonus: performance band
    if proba_pass < 0.35:
        band = "Low"
    elif proba_pass < 0.65:
        band = "Medium"
    else:
        band = "High"

    return {
        "result"     : int(prediction),
        "label"      : "Pass" if prediction == 1 else "Fail",
        "probability": round(float(proba_pass) * 100, 2),
        "performance": band,
    }


# ─── Command-line interface (Option A — Beginner UI) ──────────
if __name__ == "__main__":
    print("=" * 50)
    print("   Student Performance Evaluator — CLI")
    print("=" * 50)

    try:
        att   = float(input("  Attendance    (0–100 %) : "))
        asgn  = float(input("  Assignment    (0–100)   : "))
        quiz  = float(input("  Quiz marks    (0–100)   : "))
        mid   = float(input("  Mid-term marks(0–100)   : "))
        hrs   = float(input("  Study hours/week        : "))
    except ValueError:
        print("  ❌ Please enter valid numbers.")
        raise SystemExit(1)

    result = evaluate_student(att, asgn, quiz, mid, hrs)

    print("\n" + "=" * 50)
    print(f"  🎓 Predicted Result  : {result['label']}")
    print(f"  📊 Confidence        : {result['probability']}%")
    print(f"  📈 Performance Band  : {result['performance']}")
    print("=" * 50)

    if result['label'] == "Pass":
        print("  ✅ The student is likely to PASS.")
    else:
        print("  ⚠️  The student is at risk of FAILING.")
        print("  💡 Suggestion: Increase study hours and attendance.")
