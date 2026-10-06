"""
Hospital Readmission Risk Intelligence System
Step 01 — Data Download, Cleaning & MySQL Load

Dataset: UCI Diabetes 130-US hospitals (1999-2008)
Source:  https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Run this first.  It will:
  1. Download the dataset zip from UCI
  2. Parse and clean the CSV
  3. Engineer features
  4. Write two tables to MySQL: patient_encounters + patient_features
  5. Export a clean CSV to data/ for Tableau use
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_URL  = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "patient_features_clean.csv")

DB_USER   = os.getenv("DB_USER", "root")
DB_PASS   = os.getenv("DB_PASS", "")
DB_HOST   = os.getenv("DB_HOST", "localhost")
DB_PORT   = os.getenv("DB_PORT", "3306")
DB_NAME   = os.getenv("DB_NAME", "hospital_readmission")

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_data(url: str, dest_dir: str) -> str:
    """Download the UCI zip and return path to extracted CSV."""
    os.makedirs(dest_dir, exist_ok=True)
    csv_path = os.path.join(dest_dir, "diabetic_data.csv")

    if os.path.exists(csv_path):
        print(f"[INFO] Dataset already present at {csv_path}. Skipping download.")
        return csv_path

    print(f"[INFO] Downloading dataset from UCI...")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        # The CSV inside is named diabetic_data.csv
        for name in z.namelist():
            if name.endswith("diabetic_data.csv"):
                z.extract(name, dest_dir)
                # Move to top-level data/ if extracted into a subfolder
                extracted = os.path.join(dest_dir, name)
                if extracted != csv_path:
                    os.replace(extracted, csv_path)
                break

    print(f"[INFO] Dataset saved to {csv_path}.")
    return csv_path


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning of the raw UCI DataFrame."""
    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Drop rows where gender is 'Unknown/Invalid'
    df = df[df["gender"].isin(["Male", "Female"])].copy()

    # Keep only first encounter per patient to avoid data leakage
    df.sort_values("encounter_id", inplace=True)
    df.drop_duplicates(subset="patient_nbr", keep="first", inplace=True)

    # Remove expired or hospice discharge (disposition 11,13,14,19,20,21)
    expired_codes = {11, 13, 14, 19, 20, 21}
    df = df[~df["discharge_disposition_id"].isin(expired_codes)].copy()

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from cleaned raw data."""
    feat = pd.DataFrame()

    feat["encounter_id"]   = df["encounter_id"].astype(int)
    feat["patient_nbr"]    = df["patient_nbr"].astype(int)

    # Age midpoint from bracket strings like '[50-60)'
    def parse_age_mid(bracket: str) -> int:
        try:
            nums = bracket.replace("[", "").replace("(", "").replace("]", "").replace(")", "")
            lo, hi = nums.split("-")
            return (int(lo) + int(hi)) // 2
        except Exception:
            return 55   # fallback median
    feat["age_mid"] = df["age"].apply(parse_age_mid)

    feat["gender_enc"]         = (df["gender"] == "Male").astype(int)
    feat["time_in_hospital"]   = df["time_in_hospital"].astype(int)
    feat["num_lab_procedures"] = df["num_lab_procedures"].astype(int)
    feat["num_procedures"]     = df["num_procedures"].astype(int)
    feat["num_medications"]    = df["num_medications"].astype(int)
    feat["number_outpatient"]  = df["number_outpatient"].astype(int)
    feat["number_emergency"]   = df["number_emergency"].astype(int)
    feat["number_inpatient"]   = df["number_inpatient"].astype(int)
    feat["number_diagnoses"]   = df["number_diagnoses"].astype(int)

    feat["total_visits"] = (
        df["number_outpatient"].astype(int)
        + df["number_emergency"].astype(int)
        + df["number_inpatient"].astype(int)
    )

    feat["on_insulin"]       = (df["insulin"].isin(["Up", "Down", "Steady"])).astype(int)
    feat["change_meds"]      = (df["change"] == "Ch").astype(int)
    feat["on_diabetes_med"]  = (df["diabetesMed"] == "Yes").astype(int)

    feat["has_a1c_result"]   = (df["A1Cresult"] != "None").astype(int)
    feat["a1c_high"]         = (df["A1Cresult"].isin([">8", ">7"])).astype(int)

    # Primary diagnosis ICD-9 grouping
    def diag_flag(code: str, group: str) -> int:
        try:
            c = str(code).strip()
            # Remove leading 'E' or 'V' codes (external cause / supplementary)
            if c.startswith(("E", "V")):
                return 0
            num = float(c)
            if group == "diabetes":
                return int(250 <= num < 251)
            elif group == "circulatory":
                return int(390 <= num < 460 or 785 <= num < 786)
            elif group == "respiratory":
                return int(460 <= num < 520 or 786 <= num < 787)
        except Exception:
            pass
        return 0

    feat["diag_diabetes"]     = df["diag_1"].apply(lambda x: diag_flag(x, "diabetes"))
    feat["diag_circulatory"]  = df["diag_1"].apply(lambda x: diag_flag(x, "circulatory"))
    feat["diag_respiratory"]  = df["diag_1"].apply(lambda x: diag_flag(x, "respiratory"))

    feat["specialty_internal_med"] = (
        df["medical_specialty"].str.lower().str.contains("internal", na=False)
    ).astype(int)

    # Target variables
    feat["readmitted_30"]  = (df["readmitted"] == "<30").astype(int)
    feat["readmitted_any"] = (df["readmitted"] != "NO").astype(int)

    return feat.reset_index(drop=True)


def get_engine() -> sqlalchemy.engine.Engine:
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, echo=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Download
    csv_path = download_data(DATA_URL, DATA_DIR)

    # 2. Load raw CSV
    print("[INFO] Reading CSV...")
    raw = pd.read_csv(csv_path, low_memory=False)
    print(f"[INFO] Raw shape: {raw.shape}")

    # 3. Clean
    print("[INFO] Cleaning data...")
    cleaned = clean_raw(raw)
    print(f"[INFO] After cleaning: {cleaned.shape}")

    # 4. Feature engineering
    print("[INFO] Engineering features...")
    features = engineer_features(cleaned)
    print(f"[INFO] Feature table shape: {features.shape}")

    # 5. Write to MySQL
    print("[INFO] Writing to MySQL...")
    engine = get_engine()

    # Raw encounters (use pandas to_sql — fast for initial load)
    cleaned.to_sql(
        "patient_encounters",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi",
    )
    print("[INFO] patient_encounters written.")

    features.to_sql(
        "patient_features",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi",
    )
    print("[INFO] patient_features written.")

    # Add primary key constraints (can't do via to_sql easily)
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE patient_features ADD PRIMARY KEY (encounter_id);"
            ))
            conn.commit()
        except Exception:
            pass   # PK already exists on re-run

    # 6. Export clean CSV for Tableau
    features.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Clean CSV exported to {OUTPUT_CSV}")
    print("[DONE] Step 01 complete.")


if __name__ == "__main__":
    main()
