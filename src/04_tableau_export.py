"""
Hospital Readmission Risk Intelligence System
Step 04 — Tableau Data Exports

Generates clean CSVs from MySQL that Tableau connects to directly.
Run after Step 03.

Exported files (saved to data/tableau/):
  - tableau_patient_features.csv   — row-level data for scatter/detail views
  - tableau_kpis.csv               — aggregated KPI segments
  - tableau_model_metrics.csv      — model comparison for performance dashboard
  - tableau_predictions.csv        — predicted probabilities per patient
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_USER   = os.getenv("DB_USER", "root")
DB_PASS   = os.getenv("DB_PASS", "")
DB_HOST   = os.getenv("DB_HOST", "localhost")
DB_PORT   = os.getenv("DB_PORT", "3306")
DB_NAME   = os.getenv("DB_NAME", "hospital_readmission")

OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "tableau")
METRIC_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")
os.makedirs(OUT_DIR, exist_ok=True)


def get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def export_patient_features(engine):
    """
    Full patient feature table with human-readable labels added.
    Tableau uses this for scatter plots, filters, and detail tooltips.
    """
    df = pd.read_sql("SELECT * FROM patient_features", engine)

    # Add human-readable columns for Tableau
    bins   = [0, 30, 50, 70, 120]
    labels = ["0-30 yrs", "31-50 yrs", "51-70 yrs", "71+ yrs"]
    df["age_group"]     = pd.cut(df["age_mid"], bins=bins, labels=labels, right=True)
    df["gender_label"]  = df["gender_enc"].map({0: "Female", 1: "Male"})
    df["insulin_label"] = df["on_insulin"].map({0: "No Insulin", 1: "On Insulin"})
    df["readmit_label"] = df["readmitted_30"].map({0: "Not Readmitted", 1: "Readmitted <30d"})

    # Risk tier based on total_visits + num_medications
    def risk_tier(row):
        score = 0
        if row["total_visits"]   >= 5:  score += 2
        if row["total_visits"]   >= 3:  score += 1
        if row["num_medications"] >= 20: score += 2
        if row["num_medications"] >= 15: score += 1
        if row["a1c_high"] == 1:         score += 1
        if row["time_in_hospital"] >= 7: score += 1
        if score >= 5: return "High Risk"
        if score >= 3: return "Medium Risk"
        return "Low Risk"

    df["risk_tier"] = df.apply(risk_tier, axis=1)

    out = os.path.join(OUT_DIR, "tableau_patient_features.csv")
    df.to_csv(out, index=False)
    print(f"[EXPORT] {out}  ({len(df):,} rows)")


def export_kpis(engine):
    """
    Run the KPI population query and export.
    """
    # Trigger MySQL KPI population via the SQL script logic (replicated here for portability)
    df = pd.read_sql("SELECT * FROM patient_features", engine)

    rows = []

    # Overall
    rows.append({
        "segment_type": "overall",
        "segment_value": "All Patients",
        "total_encounters": len(df),
        "readmitted_30_count": df["readmitted_30"].sum(),
        "readmission_rate_30": round(df["readmitted_30"].mean(), 4),
        "avg_time_in_hospital": round(df["time_in_hospital"].mean(), 2),
        "avg_num_medications": round(df["num_medications"].mean(), 2),
        "avg_total_visits": round(df["total_visits"].mean(), 2),
    })

    # By age group
    bins   = [0, 30, 50, 70, 120]
    labels = ["0-30", "31-50", "51-70", "71+"]
    df["_age_group"] = pd.cut(df["age_mid"], bins=bins, labels=labels, right=True)
    for grp, sub in df.groupby("_age_group", observed=True):
        rows.append({
            "segment_type": "age_group", "segment_value": str(grp),
            "total_encounters": len(sub),
            "readmitted_30_count": sub["readmitted_30"].sum(),
            "readmission_rate_30": round(sub["readmitted_30"].mean(), 4),
            "avg_time_in_hospital": round(sub["time_in_hospital"].mean(), 2),
            "avg_num_medications": round(sub["num_medications"].mean(), 2),
            "avg_total_visits": round(sub["total_visits"].mean(), 2),
        })

    # By insulin
    for val, label in [(0, "No Insulin"), (1, "On Insulin")]:
        sub = df[df["on_insulin"] == val]
        rows.append({
            "segment_type": "insulin", "segment_value": label,
            "total_encounters": len(sub),
            "readmitted_30_count": sub["readmitted_30"].sum(),
            "readmission_rate_30": round(sub["readmitted_30"].mean(), 4),
            "avg_time_in_hospital": round(sub["time_in_hospital"].mean(), 2),
            "avg_num_medications": round(sub["num_medications"].mean(), 2),
            "avg_total_visits": round(sub["total_visits"].mean(), 2),
        })

    # By diagnosis category
    for col, label in [("diag_diabetes","Primary Dx: Diabetes"),
                       ("diag_circulatory","Primary Dx: Circulatory"),
                       ("diag_respiratory","Primary Dx: Respiratory")]:
        sub = df[df[col] == 1]
        rows.append({
            "segment_type": "diagnosis", "segment_value": label,
            "total_encounters": len(sub),
            "readmitted_30_count": sub["readmitted_30"].sum(),
            "readmission_rate_30": round(sub["readmitted_30"].mean(), 4),
            "avg_time_in_hospital": round(sub["time_in_hospital"].mean(), 2),
            "avg_num_medications": round(sub["num_medications"].mean(), 2),
            "avg_total_visits": round(sub["total_visits"].mean(), 2),
        })

    kpis = pd.DataFrame(rows)
    out  = os.path.join(OUT_DIR, "tableau_kpis.csv")
    kpis.to_csv(out, index=False)
    print(f"[EXPORT] {out}  ({len(kpis)} rows)")


def export_model_metrics():
    metrics_path = os.path.join(METRIC_DIR, "model_metrics.csv")
    if not os.path.exists(metrics_path):
        print("[WARN] model_metrics.csv not found — run Step 03 first.")
        return
    df  = pd.read_csv(metrics_path)
    out = os.path.join(OUT_DIR, "tableau_model_metrics.csv")
    df.to_csv(out, index=False)
    print(f"[EXPORT] {out}  ({len(df)} rows)")


def export_predictions(engine):
    try:
        df  = pd.read_sql(
            "SELECT p.*, f.age_mid, f.gender_enc, f.time_in_hospital, "
            "f.num_medications, f.total_visits "
            "FROM model_predictions p "
            "JOIN patient_features f ON p.encounter_id = f.encounter_id",
            engine
        )
        df["gender_label"] = df["gender_enc"].map({0: "Female", 1: "Male"})
        out = os.path.join(OUT_DIR, "tableau_predictions.csv")
        df.to_csv(out, index=False)
        print(f"[EXPORT] {out}  ({len(df):,} rows)")
    except Exception as e:
        print(f"[WARN] Could not export predictions: {e}")


def main():
    engine = get_engine()
    export_patient_features(engine)
    export_kpis(engine)
    export_model_metrics()
    export_predictions(engine)
    print("[DONE] All Tableau exports complete.")


if __name__ == "__main__":
    main()
