"""
Hospital Readmission Risk Intelligence System
Step 03 — ML Model Training, Evaluation & Logging

Models trained:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost

Outputs:
  - outputs/models/   — saved model files (.pkl)
  - outputs/metrics/  — CSV of metrics per model
  - outputs/plots/    — ROC curves, feature importance, confusion matrices
  - MySQL model_predictions table — logged for Tableau Q7 query
"""

import os
import pickle
import warnings
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score,
    precision_recall_curve
)
import xgboost as xgb
from sqlalchemy import create_engine
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_USER    = os.getenv("DB_USER", "root")
DB_PASS    = os.getenv("DB_PASS", "")
DB_HOST    = os.getenv("DB_HOST", "localhost")
DB_PORT    = os.getenv("DB_PORT", "3306")
DB_NAME    = os.getenv("DB_NAME", "hospital_readmission")

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
METRIC_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")
PLOT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
for d in [MODEL_DIR, METRIC_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

FEATURE_COLS = [
    "age_mid", "gender_enc", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "total_visits", "number_diagnoses",
    "on_insulin", "change_meds", "on_diabetes_med",
    "has_a1c_result", "a1c_high",
    "diag_diabetes", "diag_circulatory", "diag_respiratory",
    "specialty_internal_med",
]
TARGET = "readmitted_30"

PALETTE = {
    "lr":  "#2C5F8A",
    "rf":  "#E05C2A",
    "xgb": "#2A7A45",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    df = pd.read_sql("SELECT * FROM patient_features", engine)
    return df


# ── Model training ────────────────────────────────────────────────────────────

def prepare_splits(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return (X_train, X_train_sc, X_test, X_test_sc,
            y_train, y_test, idx_test, scaler)


def train_models(X_train_sc, y_train, X_train, class_weight):
    models = {}

    print("[MODEL] Training Logistic Regression...")
    lr = LogisticRegression(
        C=0.1, max_iter=1000, solver="lbfgs",
        class_weight=class_weight, random_state=42
    )
    lr.fit(X_train_sc, y_train)
    models["Logistic Regression"] = ("lr", lr, True)   # True = needs scaling

    print("[MODEL] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight=class_weight, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = ("rf", rf, False)

    print("[MODEL] Training XGBoost...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False, eval_metric="auc",
        tree_method="hist", random_state=42, n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    models["XGBoost"] = ("xgb", xgb_clf, False)

    return models


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(models, X_test, X_test_sc, y_test):
    results = {}
    for name, (short, clf, scaled) in models.items():
        X_ev = X_test_sc if scaled else X_test
        proba  = clf.predict_proba(X_ev)[:, 1]
        label  = clf.predict(X_ev)
        results[name] = {
            "short":  short,
            "clf":    clf,
            "scaled": scaled,
            "proba":  proba,
            "label":  label,
            "roc_auc": roc_auc_score(y_test, proba),
            "avg_pr":  average_precision_score(y_test, proba),
            "f1":      f1_score(y_test, label),
            "report":  classification_report(y_test, label, output_dict=True),
            "cm":      confusion_matrix(y_test, label),
        }
        print(f"  {name:22s}  ROC-AUC={results[name]['roc_auc']:.4f}  "
              f"AUCPR={results[name]['avg_pr']:.4f}  F1={results[name]['f1']:.4f}")
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        short = res["short"]
        ax.plot(fpr, tpr, linewidth=2.2, label=f"{name}  (AUC={res['roc_auc']:.3f})",
                color=PALETTE.get(short, "#555"))
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1.2, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 30-day Readmission Prediction",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curves.png"))
    plt.close()
    print("[PLOT] Saved: roc_curves.png")


def plot_precision_recall(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["proba"])
        short = res["short"]
        ax.plot(rec, prec, linewidth=2.2,
                label=f"{name}  (AUCPR={res['avg_pr']:.3f})",
                color=PALETTE.get(short, "#555"))
    baseline = y_test.mean()
    ax.axhline(baseline, linestyle="--", color="#999", linewidth=1.2,
               label=f"Random Baseline (prevalence={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "precision_recall_curves.png"))
    plt.close()
    print("[PLOT] Saved: precision_recall_curves.png")


def plot_feature_importance(results):
    """Plot feature importance for tree-based models."""
    for name in ["Random Forest", "XGBoost"]:
        clf = results[name]["clf"]
        importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
        importances = importances.sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        color = PALETTE["rf"] if name == "Random Forest" else PALETTE["xgb"]
        ax.barh(importances.index, importances.values,
                color=color, edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Feature Importance (Gini / Gain)")
        ax.set_title(f"{name} — Top Feature Importances",
                     fontsize=13, fontweight="bold", pad=10)
        plt.tight_layout()
        fname = name.lower().replace(" ", "_") + "_feature_importance.png"
        plt.savefig(os.path.join(PLOT_DIR, fname))
        plt.close()
        print(f"[PLOT] Saved: {fname}")


def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["cm"]
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Readmitted", "Readmitted"])
        ax.set_yticklabels(["Not Readmitted", "Readmitted"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=13, color="white" if cm[i,j] > cm.max()/2 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrices.png"))
    plt.close()
    print("[PLOT] Saved: confusion_matrices.png")


# ── Save models & metrics ─────────────────────────────────────────────────────

def save_models(models, scaler):
    for name, (short, clf, _) in models.items():
        path = os.path.join(MODEL_DIR, f"{short}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(clf, f)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Models saved to {MODEL_DIR}")


def save_metrics(results, y_test):
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":   name,
            "ROC_AUC": round(res["roc_auc"], 4),
            "AUCPR":   round(res["avg_pr"], 4),
            "F1":      round(res["f1"], 4),
            "Precision_class1": round(res["report"]["1"]["precision"], 4),
            "Recall_class1":    round(res["report"]["1"]["recall"], 4),
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(METRIC_DIR, "model_metrics.csv"), index=False)
    print(f"[INFO] Metrics saved to {METRIC_DIR}/model_metrics.csv")
    print(metrics_df.to_string(index=False))
    return metrics_df


def log_predictions_to_db(results, df, idx_test, y_test):
    """Write best model predictions to MySQL for Tableau Q7."""
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    res       = results[best_name]
    enc_ids   = df.loc[idx_test, "encounter_id"].values
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pred_df = pd.DataFrame({
        "encounter_id":    enc_ids,
        "model_name":      best_name,
        "predicted_proba": res["proba"],
        "predicted_label": res["label"],
        "actual_label":    y_test,
        "run_timestamp":   ts,
    })
    pred_df.to_sql("model_predictions", con=engine, if_exists="append",
                   index=False, chunksize=5000, method="multi")
    print(f"[INFO] Logged {len(pred_df):,} predictions for '{best_name}' to MySQL.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading data...")
    df = load_data()
    print(f"[INFO] {len(df):,} rows | class balance: "
          f"{df[TARGET].value_counts().to_dict()}")

    (X_train, X_train_sc, X_test, X_test_sc,
     y_train, y_test, idx_test, scaler) = prepare_splits(df)

    class_weight = "balanced"
    print("\n[INFO] Training models...")
    models = train_models(X_train_sc, y_train, X_train, class_weight)

    print("\n[INFO] Evaluating models...")
    results = evaluate(models, X_test, X_test_sc, y_test)

    print("\n[INFO] Generating plots...")
    plot_roc_curves(results, y_test)
    plot_precision_recall(results, y_test)
    plot_feature_importance(results)
    plot_confusion_matrices(results, y_test)

    save_models(models, scaler)
    save_metrics(results, y_test)
    log_predictions_to_db(results, df, idx_test, y_test)

    print("\n[DONE] Step 03 complete.")


if __name__ == "__main__":
    main()
