# Hospital Readmission Risk Intelligence System

A full end-to-end data project that predicts **30-day hospital readmissions** for diabetic patients using SQL, Python, and Tableau — built on real clinical data from 130 US hospitals.

---

## Why This Project

Hospital readmissions within 30 days cost the US healthcare system over **$26 billion annually**. CMS (Centers for Medicare & Medicaid) penalises hospitals financially for excess readmissions under the Hospital Readmissions Reduction Program (HRRP). Early identification of high-risk patients allows clinical teams to intervene — closer follow-ups, medication review, discharge planning.

This project builds that identification system end to end.

---

## What It Does

| Layer | Tool | What Happens |
|---|---|---|
| Data Storage & Analysis | **MySQL + SQL** | Schema design, analytical queries, KPI aggregation |
| Data Wrangling & ML | **Python** | Cleaning, feature engineering, model training (LR / RF / XGBoost), evaluation |
| Visualisation & Reporting | **Tableau** | Interactive dashboard — readmission rates, risk segments, model performance |

---

## Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)**  
→ https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

- 101,766 encounters from 130 hospitals across the US
- 50 features: demographics, diagnoses, medications, lab results
- Target: `readmitted` — `<30` days / `>30` days / `NO`

The dataset is downloaded automatically when you run Step 01.

---

## Project Structure

```
hospital_readmission/
│
├── sql/
│   ├── 01_create_schema.sql       # All table definitions
│   └── 02_analysis_queries.sql    # 8 analytical queries + KPI population
│
├── src/
│   ├── 01_load_and_clean.py       # Download data → clean → MySQL load
│   ├── 02_eda.py                  # 6 EDA plots saved to outputs/eda/
│   ├── 03_model.py                # Train LR, RF, XGBoost; ROC/PR/FI plots
│   ├── 04_tableau_export.py       # Export CSVs for Tableau connection
│   └── 05_predict_new_patient.py  # Inference: score a new patient
│
├── data/
│   └── tableau/                   # Tableau-ready CSVs (auto-generated)
│
├── outputs/
│   ├── eda/                       # EDA plots (PNG)
│   ├── models/                    # Saved .pkl model files
│   ├── metrics/                   # model_metrics.csv
│   └── plots/                     # ROC, PR, confusion matrix, feature importance
│
├── tableau_exports/               # Tableau workbook (.twbx) — see below
├── .env.example                   # DB config template
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Prerequisites

- Python 3.10+
- MySQL 8.0+ running locally
- Tableau Desktop or Tableau Public (free)

### 2. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/hospital-readmission-risk.git
cd hospital-readmission-risk
```

### 3. Create Python virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Set up MySQL database

Open MySQL Workbench (or any MySQL client) and run:

```sql
-- In MySQL Workbench / terminal:
source sql/01_create_schema.sql
```

### 5. Configure environment

```bash
cp .env.example .env
# Open .env and fill in your MySQL username and password
```

### 6. Run the pipeline (in order)

```bash
# Step 1 — Download UCI data, clean it, load into MySQL
python src/01_load_and_clean.py

# Step 2 — Generate EDA plots
python src/02_eda.py

# Step 3 — Train models, evaluate, log predictions to MySQL
python src/03_model.py

# Step 4 — Export CSVs for Tableau
python src/04_tableau_export.py

# Step 5 — Optional: score a sample patient
python src/05_predict_new_patient.py
```

Total runtime: ~5–10 minutes end to end (XGBoost training is the slowest step).

---

## SQL Queries Covered

| Query | What It Answers |
|---|---|
| Q1 | Overall 30-day and any-readmission rates |
| Q2 | Readmission rate by age group with avg LOS |
| Q3 | High-risk patient profiles (high visits + many medications) |
| Q4 | Insulin prescription vs readmission correlation |
| Q5 | A1C testing — does measuring it reduce readmission? |
| Q6 | Primary diagnosis category and readmission risk |
| Q7 | Model prediction accuracy pulled from logs |
| Q8 | Aggregated KPI table for Tableau dashboards |

Run these from MySQL Workbench after Step 01:
```
source sql/02_analysis_queries.sql
```

---

## ML Models & Results

Three models trained on an 80/20 stratified split:

| Model | ROC-AUC | AUC-PR | F1 Score |
|---|---|---|---|
| Logistic Regression | ~0.64 | ~0.21 | ~0.23 |
| Random Forest | ~0.67 | ~0.24 | ~0.26 |
| **XGBoost** | **~0.69** | **~0.26** | **~0.28** |

> Note: 30-day readmission prediction is genuinely hard (class imbalance ~11%). These numbers are consistent with published literature on this dataset. The goal here is the system, not state-of-the-art accuracy.

Key findings from feature importance:
- **Number of inpatient visits** (prior hospitalisations) is the strongest predictor
- **Number of medications** and **time in hospital** are highly informative
- **A1C testing** correlates with lower readmission — patients tested are managed better

---

## Tableau Dashboard

The Tableau workbook (`tableau_exports/readmission_dashboard.twbx`) includes 4 sheets:

1. **Overview KPIs** — total encounters, readmission rate, avg LOS
2. **Segment Analysis** — readmission rate by age group, insulin use, diagnosis
3. **Risk Tier Distribution** — High / Medium / Low risk patient breakdown
4. **Model Performance** — ROC-AUC comparison bar chart

To open: double-click the `.twbx` file in Tableau Desktop / Tableau Public.  
To reconnect data: Data → Edit Data Source → point to `data/tableau/` CSVs.

---

## Key Insights

1. Patients aged 71+ have a **~12.5% 30-day readmission rate** — the highest of any age group
2. Prior inpatient visits are a stronger predictor than the current admission's length of stay
3. Patients on insulin who had their medication **changed** during the visit show lower readmission — suggesting active management works
4. Only ~56% of encounters included an A1C test, yet those patients had lower readmission — testing gap is a clinical intervention opportunity
5. XGBoost flags ~68% of actual readmissions in the top-30% risk score bucket (lift = 2.1×)

---

## Resume Bullet Points

Feel free to copy these:

> - Built an end-to-end hospital readmission prediction system on 100K+ real patient encounters using Python (scikit-learn, XGBoost), MySQL, and Tableau
> - Designed a normalised relational schema; wrote 8 analytical SQL queries covering readmission KPIs, segment breakdowns, and A1C testing gaps
> - Trained and compared Logistic Regression, Random Forest, and XGBoost classifiers; best model achieved ROC-AUC of 0.69 on a class-imbalanced clinical dataset
> - Built an interactive Tableau dashboard with 4 views covering risk segmentation, demographic breakdown, and model performance comparison

---

## References

- Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International*. DOI: 10.1155/2014/781670
- UCI Machine Learning Repository — Diabetes 130-US Hospitals Dataset
- CMS Hospital Readmissions Reduction Program (HRRP): https://www.cms.gov/medicare/quality/initiatives/hospital-quality-initiative/hospital-readmissions-reduction-program

---

## Author

**Shashank**  
M.Tech — Biomedical Devices Engineering, IIT Indore
