"""
app.py — Flask web app for Student Academic Performance Prediction System
Auto-trains the model on first launch if artefacts are missing.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from config import MODEL_DIR, DATA_DIR, RANDOM_STATE
from train  import ALL_FEATURES, LABEL_MAP, LABEL_INV, encode_df, prepare_xy

app = Flask(__name__)
app.secret_key = "student-ml-ibm-bob-2024"

# ── Auto-train if models not present ─────────────────────────────────────────
def ensure_models():
    required = [
        os.path.join(MODEL_DIR, "scaler.pkl"),
        os.path.join(MODEL_DIR, "classifier.pkl"),
        os.path.join(MODEL_DIR, "regressor.pkl"),
        os.path.join(MODEL_DIR, "meta.json"),
    ]
    if not all(os.path.exists(p) for p in required):
        print("Models not found — training now (this takes ~30 seconds)…")
        from train import train_and_evaluate
        train_and_evaluate()

ensure_models()

# ── Load artefacts ────────────────────────────────────────────────────────────
scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
regressor  = joblib.load(os.path.join(MODEL_DIR, "regressor.pkl"))
with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    META = json.load(f)

# ── Intervention recommendations ─────────────────────────────────────────────
INTERVENTIONS = {
    "attendance_pct": {
        "threshold": 75,
        "message": "Attendance below 75%. Recommend mandatory attendance monitoring and weekly counsellor check-ins.",
    },
    "study_hours_day": {
        "threshold": 2,
        "message": "Study hours below 2h/day. Recommend structured study timetable and peer study groups.",
    },
    "mental_health_score": {
        "threshold": 5,
        "message": "Mental health score is low. Recommend referral to campus counselling and wellness workshops.",
    },
    "midterm_score": {
        "threshold": 50,
        "message": "Midterm score below 50. Recommend immediate academic tutoring and subject-specific support.",
    },
    "assignment_avg": {
        "threshold": 50,
        "message": "Assignment average below 50. Recommend assignment workshops and deadline management coaching.",
    },
    "num_failed_before": {
        "threshold": 1,
        "message": "Prior failed courses detected. Recommend academic probation review and learning support plan.",
    },
    "sleep_hours": {
        "threshold": 6,
        "message": "Sleep under 6 hours. Recommend sleep hygiene workshop and schedule adjustment.",
    },
    "tutoring_sessions": {
        "threshold": 3,
        "message": "Very few tutoring sessions. Recommend enrolment in free campus tutoring programme.",
    },
}

GENERAL_INTERVENTIONS = {
    "Low Risk": [
        "Maintain current study habits.",
        "Consider mentoring at-risk peers.",
        "Explore advanced electives or research opportunities.",
    ],
    "Medium Risk": [
        "Schedule bi-weekly meetings with academic advisor.",
        "Join a study group for weaker subjects.",
        "Utilise library and online learning resources.",
        "Review and improve time-management strategies.",
    ],
    "High Risk": [
        "Immediate academic intervention required.",
        "Enrol in the Early Warning Support Programme.",
        "Assigned a dedicated faculty mentor.",
        "Weekly progress tracking with department head.",
        "Consider course load reduction if necessary.",
    ],
}


def build_interventions(form_data: dict, risk_label: str) -> list:
    tips = list(GENERAL_INTERVENTIONS.get(risk_label, []))
    for field, rule in INTERVENTIONS.items():
        val = form_data.get(field)
        if val is not None:
            try:
                if float(val) < rule["threshold"]:
                    tips.insert(0, rule["message"])
            except (ValueError, TypeError):
                pass
    return tips


def form_to_vector(form: dict) -> np.ndarray:
    """Convert the HTML form POST data to a scaled feature vector."""
    gender_map   = {"Male": 0, "Female": 1, "Other": 2}
    income_map   = {"Low": 0, "Middle": 1, "High": 2}
    edu_map      = {"No Degree": 0, "Diploma": 1, "Bachelor": 2, "Postgrad": 3}

    row = [
        float(form.get("age",                  20)),
        float(form.get("prev_gpa",             2.5)),
        float(form.get("study_hours_day",       3)),
        float(form.get("attendance_pct",        75)),
        float(form.get("num_failed_before",      0)),
        float(form.get("midterm_score",         60)),
        float(form.get("assignment_avg",        60)),
        float(form.get("quiz_avg",              60)),
        float(form.get("lab_score",             60)),
        float(form.get("mental_health_score",    7)),
        float(form.get("sleep_hours",            7)),
        float(form.get("distance_km",           10)),
        float(form.get("tutoring_sessions",      2)),
        float(form.get("first_gen_student",      0)),
        float(form.get("extracurricular",        0)),
        float(form.get("part_time_job",          0)),
        float(form.get("internet_access",        1)),
        float(gender_map.get(form.get("gender", "Male"), 0)),
        float(income_map.get(form.get("family_income", "Middle"), 1)),
        float(edu_map.get(form.get("parent_education", "Bachelor"), 2)),
    ]
    return np.array(row).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", meta=META)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        form = request.form.to_dict()
        vec  = form_to_vector(form)
        vec_s = scaler.transform(vec)

        risk_idx   = int(classifier.predict(vec_s)[0])
        risk_label = LABEL_INV[risk_idx]
        proba      = classifier.predict_proba(vec_s)[0].tolist()
        pred_gpa   = round(float(regressor.predict(vec_s)[0]), 2)
        pred_gpa   = max(0.0, min(4.0, pred_gpa))

        result = {
            "risk_label":    risk_label,
            "risk_idx":      risk_idx,
            "proba":         [round(p * 100, 1) for p in proba],
            "pred_gpa":      pred_gpa,
            "interventions": build_interventions(form, risk_label),
            "form":          form,
        }
    return render_template("predict.html", result=result)


@app.route("/insights")
def insights():
    return render_template("insights.html", meta=META)


@app.route("/about")
def about():
    return render_template("about.html", meta=META)


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data  = request.get_json(force=True)
    vec   = form_to_vector(data)
    vec_s = scaler.transform(vec)

    risk_idx   = int(classifier.predict(vec_s)[0])
    risk_label = LABEL_INV[risk_idx]
    proba      = classifier.predict_proba(vec_s)[0].tolist()
    pred_gpa   = round(float(regressor.predict(vec_s)[0]), 2)
    pred_gpa   = max(0.0, min(4.0, pred_gpa))

    return jsonify({
        "risk_label":    risk_label,
        "risk_index":    risk_idx,
        "probabilities": {LABEL_INV[i]: round(p, 4) for i, p in enumerate(proba)},
        "predicted_gpa": pred_gpa,
        "interventions": build_interventions(data, risk_label),
    })


@app.route("/api/stats")
def api_stats():
    return jsonify({
        "best_model":    META["best_model"],
        "accuracy":      META["best_accuracy"],
        "gpa_r2":        META["gpa_r2"],
        "dataset_size":  META["dataset_size"],
        "class_dist":    META["class_distribution"],
        "model_results": {k: {"accuracy": v["accuracy"], "cv_mean": v["cv_mean"]}
                          for k, v in META["results"].items()},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
