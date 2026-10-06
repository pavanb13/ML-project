"""
generate_data.py
Generates a realistic synthetic dataset of 2000 students with 20 features.
Run standalone:  python generate_data.py
"""

import os
import numpy as np
import pandas as pd
from config import RANDOM_STATE, N_STUDENTS, DATA_DIR

rng = np.random.default_rng(RANDOM_STATE)


def generate_dataset(n: int = N_STUDENTS) -> pd.DataFrame:
    # ── Demographic & background ─────────────────────────────────────────────
    gender            = rng.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    age               = rng.integers(17, 24, n)
    family_income     = rng.choice(["Low", "Middle", "High"], n, p=[0.30, 0.50, 0.20])
    parent_education  = rng.choice(["No Degree", "Diploma", "Bachelor", "Postgrad"], n,
                                   p=[0.15, 0.25, 0.40, 0.20])
    first_gen_student = rng.choice([0, 1], n, p=[0.65, 0.35])

    # ── Academic history ─────────────────────────────────────────────────────
    prev_gpa          = rng.uniform(1.5, 4.0, n).round(2)
    study_hours_day   = rng.uniform(0.5, 8.0, n).round(1)
    attendance_pct    = rng.uniform(40, 100, n).round(1)
    num_failed_before = rng.integers(0, 4, n)

    # ── Current semester metrics ─────────────────────────────────────────────
    midterm_score     = rng.uniform(20, 100, n).round(1)
    assignment_avg    = rng.uniform(30, 100, n).round(1)
    quiz_avg          = rng.uniform(25, 100, n).round(1)
    lab_score         = rng.uniform(30, 100, n).round(1)

    # ── Extracurricular & wellbeing ──────────────────────────────────────────
    extracurricular   = rng.choice([0, 1], n, p=[0.45, 0.55])
    part_time_job     = rng.choice([0, 1], n, p=[0.60, 0.40])
    internet_access   = rng.choice([0, 1], n, p=[0.10, 0.90])
    mental_health_score = rng.integers(1, 11, n)   # 1=poor, 10=excellent
    sleep_hours       = rng.uniform(4, 10, n).round(1)

    # ── Distance / support ───────────────────────────────────────────────────
    distance_km       = rng.uniform(0.5, 80, n).round(1)
    tutoring_sessions = rng.integers(0, 20, n)

    # ── Derive GPA (target) with realistic noise ─────────────────────────────
    income_map = {"Low": 0.0, "Middle": 0.15, "High": 0.30}
    edu_map    = {"No Degree": 0.0, "Diploma": 0.08, "Bachelor": 0.15, "Postgrad": 0.22}

    gpa = (
        0.50 * prev_gpa
        + 0.06 * (study_hours_day / 8.0) * 4
        + 0.08 * (attendance_pct  / 100) * 4
        + 0.10 * (midterm_score   / 100) * 4
        + 0.07 * (assignment_avg  / 100) * 4
        + 0.06 * (quiz_avg        / 100) * 4
        + 0.05 * (lab_score       / 100) * 4
        + 0.04 * (mental_health_score / 10) * 4
        + 0.02 * (sleep_hours     / 10) * 4
        + 0.02 * (tutoring_sessions / 20) * 4
        + np.array([income_map[v] for v in family_income])
        + np.array([edu_map[v]    for v in parent_education])
        - 0.12 * num_failed_before
        - 0.08 * part_time_job
        + rng.normal(0, 0.20, n)
    )
    # Normalise to 0–4 range
    gpa = (gpa - gpa.min()) / (gpa.max() - gpa.min() + 1e-9) * 4.0
    gpa = np.clip(gpa, 0.0, 4.0).round(2)

    # ── Risk label (percentile-based for balanced classes) ───────────────────
    gpa_p33 = float(np.percentile(gpa, 33))
    gpa_p66 = float(np.percentile(gpa, 66))

    def risk_label(g, att, mh):
        # Hard rule overrides for clearly at-risk signals
        if att < 55 or mh <= 2:
            return "High Risk"
        # GPA tertile-based bucketing
        if g <= gpa_p33:
            return "High Risk"
        if g <= gpa_p66:
            return "Medium Risk"
        return "Low Risk"

    risk = np.array([
        risk_label(gpa[i], attendance_pct[i], mental_health_score[i])
        for i in range(n)
    ])

    df = pd.DataFrame({
        "student_id":         [f"STU{str(i+1).zfill(4)}" for i in range(n)],
        "gender":             gender,
        "age":                age,
        "family_income":      family_income,
        "parent_education":   parent_education,
        "first_gen_student":  first_gen_student,
        "prev_gpa":           prev_gpa,
        "study_hours_day":    study_hours_day,
        "attendance_pct":     attendance_pct,
        "num_failed_before":  num_failed_before,
        "midterm_score":      midterm_score,
        "assignment_avg":     assignment_avg,
        "quiz_avg":           quiz_avg,
        "lab_score":          lab_score,
        "extracurricular":    extracurricular,
        "part_time_job":      part_time_job,
        "internet_access":    internet_access,
        "mental_health_score":mental_health_score,
        "sleep_hours":        sleep_hours,
        "distance_km":        distance_km,
        "tutoring_sessions":  tutoring_sessions,
        "final_gpa":          gpa,
        "risk_category":      risk,
    })
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_dataset()
    path = os.path.join(DATA_DIR, "students.csv")
    df.to_csv(path, index=False)
    print(f"Dataset saved → {path}  ({len(df)} rows)")
    print(df["risk_category"].value_counts())
