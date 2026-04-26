# ============================================================
# app.py  —  Task 9 : Streamlit Web UI
# Run with:  streamlit run app.py
# ============================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ── Resolve paths relative to THIS file (works locally & on Streamlit Cloud)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
DATA_PATH   = os.path.join(BASE_DIR, "dataset.xlsx")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Evaluator",
    page_icon="🎓",
    layout="centered",
)

# ── Train model on-the-fly if .joblib files are missing ──────
# (needed on Streamlit Cloud where only source files are deployed)
def train_and_save():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_excel(DATA_PATH)
    X  = df[['attendance', 'assignment', 'quiz', 'mid', 'study_hours']]
    y  = df['result']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

# ── Load model & scaler ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return train_and_save()
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_artifacts()

# ── Evaluation function ───────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled   = scaler.transform(features)
    prediction   = model.predict(scaled)[0]
    proba_pass   = model.predict_proba(scaled)[0][1]

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

# ── Title & description ───────────────────────────────────────
st.title("🎓 Student Performance Evaluator")
st.markdown(
    """
    Enter a student's academic details below.  
    The **Artificial Neural Network** will predict whether the student
    will **Pass** or **Fail** and show a confidence score.
    """
)
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    attendance  = st.slider("Attendance (%)",        0, 100, 75)
    assignment  = st.slider("Assignment Marks",       0, 100, 70)
    quiz        = st.slider("Quiz Marks",             0, 100, 65)

with col2:
    mid         = st.slider("Mid-Term Marks",         0, 100, 60)
    study_hours = st.slider("Study Hours / Week",     0,  20,  7)

st.divider()

# ── Predict button ────────────────────────────────────────────
if st.button("🔍 Evaluate Student", use_container_width=True):
    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    # ── Result display ─────────────────────────────────────────
    st.subheader("📊 Prediction Result")

    col_r, col_p = st.columns(2)

    with col_r:
        if result["label"] == "Pass":
            st.success(f"✅ **{result['label']}**")
        else:
            st.error(f"❌ **{result['label']}**")

    with col_p:
        st.metric(
            label="Pass Probability",
            value=f"{result['probability']}%",
        )

    # Performance band
    band_colour = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
    st.info(
        f"{band_colour[result['performance']]}  "
        f"**Performance Band: {result['performance']}**"
    )

    # ── Gauge chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 2.5))
    prob = result["probability"] / 100.0

    # Background bar
    ax.barh(0, 1, color="#e0e0e0", height=0.4)
    # Value bar
    bar_color = "#4CAF50" if result["label"] == "Pass" else "#f44336"
    ax.barh(0, prob, color=bar_color, height=0.4)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(f"Pass Probability: {result['probability']}%", pad=10)
    ax.axvline(0.5, color="gray", linewidth=1, linestyle="--")
    ax.text(0.5, 0.55, "Decision\nThreshold", ha="center",
            va="bottom", fontsize=8, color="gray", transform=ax.get_xaxis_transform())
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Feature summary table ──────────────────────────────────
    st.subheader("📌 Input Summary")
    st.table({
        "Feature"     : ["Attendance (%)", "Assignment", "Quiz",
                         "Mid-Term", "Study Hours/Week"],
        "Value Entered": [attendance, assignment, quiz, mid, study_hours],
    })

    # ── Interpretation ─────────────────────────────────────────
    st.subheader("💡 Interpretation")
    if result["label"] == "Pass":
        st.markdown(
            f"""
            The model predicts this student will **Pass** with **{result['probability']}%** 
            confidence. The student shows strong academic indicators.  
            Keep up the good work! 🎉
            """
        )
    else:
        tips = []
        if attendance < 75:
            tips.append("📅 Improve attendance (currently below 75%)")
        if study_hours < 6:
            tips.append("📚 Increase weekly study hours")
        if mid < 50:
            tips.append("📝 Focus more on mid-term preparation")
        if quiz < 50:
            tips.append("❓ Practice more quizzes")

        tips_text = "\n".join(f"- {t}" for t in tips) if tips else "- Review all subjects thoroughly."
        st.markdown(
            f"""
            The model predicts this student is **at risk of Failing**.  
            Confidence in this prediction: **{result['probability']}%**.

            **Suggested Improvements:**  
            {tips_text}
            """
        )

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Powered by an ANN (MLPClassifier) trained on 600 student records • "
    "Built with Streamlit & scikit-learn"
)
