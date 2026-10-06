"""
Hospital Readmission Risk Intelligence System
Step 05 — Inference Script

Given a new patient's data (as a dict or CSV row),
load the best saved model and output a readmission risk score.

Usage:
    python src/05_predict_new_patient.py
    python src/05_predict_new_patient.py --input data/new_patients.csv
"""

import os
import pickle
import argparse
import pandas as pd
import numpy as np

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
FEATURE_COLS = [
    "age_mid", "gender_enc", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "num_medications",
    "total_visits", "number_diagnoses",
    "on_insulin", "change_meds", "on_diabetes_med",
    "has_a1c_result", "a1c_high",
    "diag_diabetes", "diag_circulatory", "diag_respiratory",
    "specialty_internal_med",
]

# Sample patient for demonstration
SAMPLE_PATIENT = {
    "age_mid":               65,
    "gender_enc":            1,    # Male
    "time_in_hospital":      7,
    "num_lab_procedures":    55,
    "num_procedures":        2,
    "num_medications":       18,
    "total_visits":          6,
    "number_diagnoses":      9,
    "on_insulin":            1,
    "change_meds":           1,
    "on_diabetes_med":       1,
    "has_a1c_result":        1,
    "a1c_high":              1,
    "diag_diabetes":         1,
    "diag_circulatory":      0,
    "diag_respiratory":      0,
    "specialty_internal_med":1,
}


def load_model(model_name: str = "xgb"):
    model_path  = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run Step 03 first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler


def predict(patient_dict: dict, model_name: str = "xgb") -> dict:
    model, scaler = load_model(model_name)

    df = pd.DataFrame([patient_dict])[FEATURE_COLS]

    # XGBoost / RF don't need scaling
    if model_name == "lr" and scaler is not None:
        X = scaler.transform(df.values)
    else:
        X = df.values

    proba = model.predict_proba(X)[0, 1]
    label = int(proba >= 0.5)

    risk = "HIGH" if proba >= 0.5 else ("MEDIUM" if proba >= 0.3 else "LOW")

    return {
        "readmission_probability": round(float(proba), 4),
        "predicted_readmission_<30d": bool(label),
        "risk_tier": risk,
        "model_used": model_name.upper(),
    }


def predict_from_csv(csv_path: str, model_name: str = "xgb") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        patient = row[FEATURE_COLS].to_dict()
        res     = predict(patient, model_name)
        results.append({**row.to_dict(), **res})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Predict readmission risk for new patients.")
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to CSV file with patient rows")
    parser.add_argument("--model",  type=str, default="xgb",
                        choices=["lr", "rf", "xgb"],
                        help="Which model to use (default: xgb)")
    args = parser.parse_args()

    if args.input:
        print(f"[INFO] Running batch inference on {args.input}")
        results = predict_from_csv(args.input, args.model)
        out_path = args.input.replace(".csv", "_predictions.csv")
        results.to_csv(out_path, index=False)
        print(f"[INFO] Predictions saved to {out_path}")
        print(results[["readmission_probability", "predicted_readmission_<30d", "risk_tier"]].head(10))
    else:
        print("[INFO] Running inference on sample patient:")
        for k, v in SAMPLE_PATIENT.items():
            print(f"  {k}: {v}")
        print()
        result = predict(SAMPLE_PATIENT, args.model)
        print("=" * 45)
        print("  READMISSION RISK PREDICTION")
        print("=" * 45)
        print(f"  Probability of 30-day Readmission : {result['readmission_probability']:.1%}")
        print(f"  Predicted Readmission             : {'YES' if result['predicted_readmission_<30d'] else 'NO'}")
        print(f"  Risk Tier                         : {result['risk_tier']}")
        print(f"  Model Used                        : {result['model_used']}")
        print("=" * 45)


if __name__ == "__main__":
    main()
