"""
train.py
Trains multiple classifiers and a GPA regressor.
Saves the best classifier, the regressor, the scaler, and evaluation metrics.

Run standalone:  python train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier

from generate_data import generate_dataset
from config        import RANDOM_STATE, DATA_DIR, MODEL_DIR

# ── Feature lists ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "prev_gpa", "study_hours_day", "attendance_pct", "num_failed_before",
    "midterm_score", "assignment_avg", "quiz_avg", "lab_score",
    "mental_health_score", "sleep_hours", "distance_km", "tutoring_sessions",
    "first_gen_student", "extracurricular", "part_time_job", "internet_access",
]
CATEGORICAL_FEATURES = ["gender", "family_income", "parent_education"]
ALL_FEATURES         = NUMERIC_FEATURES + ["gender_enc", "family_income_enc", "parent_education_enc"]

LABEL_MAP = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns and return a copy ready for ML."""
    d = df.copy()
    for col, mapping in [
        ("gender",           {"Male": 0, "Female": 1, "Other": 2}),
        ("family_income",    {"Low": 0, "Middle": 1, "High": 2}),
        ("parent_education", {"No Degree": 0, "Diploma": 1, "Bachelor": 2, "Postgrad": 3}),
    ]:
        d[col + "_enc"] = d[col].map(mapping)
    return d


def prepare_xy(df: pd.DataFrame):
    d  = encode_df(df)
    X  = d[ALL_FEATURES].values.astype(float)
    y  = d["risk_category"].map(LABEL_MAP).values
    gpa = d["final_gpa"].values
    return X, y, gpa


def train_and_evaluate():
    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, "students.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_dataset()
        df.to_csv(csv_path, index=False)
    print(f"Dataset: {len(df)} rows | Features: {len(ALL_FEATURES)}")

    X, y, gpa = prepare_xy(df)

    X_train, X_test, y_train, y_test, gpa_train, gpa_test = train_test_split(
        X, y, gpa, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    scaler   = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Classifiers ──────────────────────────────────────────────────────────
    classifiers = {
        "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":          RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting":      GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM":                    SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbours":   KNeighborsClassifier(n_neighbors=7),
    }

    results   = {}
    best_name = None
    best_acc  = 0.0
    best_model = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in classifiers.items():
        clf.fit(X_train_s, y_train)
        y_pred   = clf.predict(X_test_s)
        acc      = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="accuracy")
        report   = classification_report(y_test, y_pred,
                                         target_names=list(LABEL_MAP.keys()),
                                         output_dict=True)
        results[name] = {
            "accuracy":    round(acc * 100, 2),
            "cv_mean":     round(cv_scores.mean() * 100, 2),
            "cv_std":      round(cv_scores.std()  * 100, 2),
            "report":      report,
            "cm":          confusion_matrix(y_test, y_pred).tolist(),
        }
        print(f"  {name:<28}  Acc={acc:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_model = clf

    print(f"\nBest classifier: {best_name} ({best_acc*100:.2f}%)")

    # ── GPA Regressor ─────────────────────────────────────────────────────────
    regressor = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    regressor.fit(X_train_s, gpa_train)
    gpa_pred = regressor.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(gpa_test, gpa_pred))
    r2   = r2_score(gpa_test, gpa_pred)
    print(f"GPA Regressor: RMSE={rmse:.4f}  R²={r2:.4f}")

    # ── Feature importance (from best tree-based or RF fallback) ─────────────
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = regressor.feature_importances_

    feat_imp = sorted(
        zip(ALL_FEATURES, importances.tolist()),
        key=lambda x: x[1], reverse=True
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(scaler,      os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(best_model,  os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(regressor,   os.path.join(MODEL_DIR, "regressor.pkl"))

    meta = {
        "best_model":        best_name,
        "best_accuracy":     round(best_acc * 100, 2),
        "gpa_rmse":          round(rmse, 4),
        "gpa_r2":            round(r2, 4),
        "feature_names":     ALL_FEATURES,
        "label_map":         LABEL_MAP,
        "results":           results,
        "feature_importance": feat_imp,
        "class_distribution": df["risk_category"].value_counts().to_dict(),
        "dataset_size":       len(df),
    }
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Plots (saved to static/img) ───────────────────────────────────────────
    os.makedirs("static/img", exist_ok=True)
    _plot_confusion_matrix(confusion_matrix(y_test, best_model.predict(X_test_s)), best_name)
    _plot_feature_importance(feat_imp[:12])
    _plot_model_comparison(results)
    _plot_risk_distribution(df)

    print("Training complete. Artefacts saved to ./models/")
    return meta


def _plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(LABEL_MAP.keys()),
                yticklabels=list(LABEL_MAP.keys()), ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, pad=12)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("static/img/confusion_matrix.png", dpi=120)
    plt.close()


def _plot_feature_importance(feat_imp):
    names, vals = zip(*feat_imp)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(names)))
    ax.barh(list(reversed(names)), list(reversed(vals)), color=list(reversed(colors)))
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Feature Importances", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig("static/img/feature_importance.png", dpi=120)
    plt.close()


def _plot_model_comparison(results):
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    cvs   = [results[n]["cv_mean"]  for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - 0.2, accs, 0.35, label="Test Accuracy (%)",  color="#3b82d4")
    b2 = ax.bar(x + 0.2, cvs,  0.35, label="CV Accuracy (%)",    color="#7c5cd8")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(50, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison", fontsize=12, pad=12)
    ax.legend()
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    plt.savefig("static/img/model_comparison.png", dpi=120)
    plt.close()


def _plot_risk_distribution(df):
    counts = df["risk_category"].value_counts()
    colors = {"Low Risk": "#22c55e", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(c, "#94a3b8") for c in counts.index])
    ax.set_ylabel("Number of Students")
    ax.set_title("Risk Category Distribution", fontsize=12, pad=12)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig("static/img/risk_distribution.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    train_and_evaluate()
