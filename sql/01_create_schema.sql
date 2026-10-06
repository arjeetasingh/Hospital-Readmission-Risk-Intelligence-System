-- =============================================================
-- Hospital Readmission Risk Intelligence System
-- Script 01: Database Schema Creation
-- =============================================================

CREATE DATABASE IF NOT EXISTS hospital_readmission;
USE hospital_readmission;

-- Raw patient encounters table (mirrors the CSV from UCI dataset)
CREATE TABLE IF NOT EXISTS patient_encounters (
    encounter_id        BIGINT PRIMARY KEY,
    patient_nbr         BIGINT,
    race                VARCHAR(50),
    gender              VARCHAR(20),
    age                 VARCHAR(20),         -- stored as brackets e.g. [50-60)
    weight              VARCHAR(20),
    admission_type_id   INT,
    discharge_disposition_id INT,
    admission_source_id INT,
    time_in_hospital    INT,                 -- days
    payer_code          VARCHAR(20),
    medical_specialty   VARCHAR(100),
    num_lab_procedures  INT,
    num_procedures      INT,
    num_medications     INT,
    number_outpatient   INT,
    number_emergency    INT,
    number_inpatient    INT,
    diag_1              VARCHAR(20),
    diag_2              VARCHAR(20),
    diag_3              VARCHAR(20),
    number_diagnoses    INT,
    max_glu_serum       VARCHAR(20),
    A1Cresult           VARCHAR(20),
    metformin           VARCHAR(20),
    insulin             VARCHAR(20),
    change_meds         VARCHAR(10),
    diabetesMed         VARCHAR(10),
    readmitted          VARCHAR(20)          -- '<30', '>30', 'NO'
);

-- Cleaned & feature-engineered version (populated by Python)
CREATE TABLE IF NOT EXISTS patient_features (
    encounter_id            BIGINT PRIMARY KEY,
    patient_nbr             BIGINT,
    age_mid                 INT,             -- midpoint of age bracket
    gender_enc              TINYINT,         -- 0=Female, 1=Male
    time_in_hospital        INT,
    num_lab_procedures      INT,
    num_procedures          INT,
    num_medications         INT,
    number_outpatient       INT,
    number_emergency        INT,
    number_inpatient        INT,
    number_diagnoses        INT,
    total_visits            INT,             -- outpatient + emergency + inpatient
    on_insulin              TINYINT,
    change_meds             TINYINT,
    on_diabetes_med         TINYINT,
    has_a1c_result          TINYINT,
    a1c_high                TINYINT,         -- A1C > 8
    diag_diabetes           TINYINT,
    diag_circulatory        TINYINT,
    diag_respiratory        TINYINT,
    specialty_internal_med  TINYINT,
    readmitted_30           TINYINT,         -- target: 1 = readmitted <30 days
    readmitted_any          TINYINT          -- 1 = readmitted at all
);

-- Model predictions log
CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id       INT AUTO_INCREMENT PRIMARY KEY,
    encounter_id        BIGINT,
    model_name          VARCHAR(50),
    predicted_proba     FLOAT,
    predicted_label     TINYINT,
    actual_label        TINYINT,
    run_timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (encounter_id) REFERENCES patient_features(encounter_id)
);

-- Aggregated KPI table for Tableau
CREATE TABLE IF NOT EXISTS readmission_kpis (
    kpi_id              INT AUTO_INCREMENT PRIMARY KEY,
    segment_type        VARCHAR(50),         -- 'age_group', 'specialty', 'diagnosis', 'overall'
    segment_value       VARCHAR(100),
    total_encounters    INT,
    readmitted_30_count INT,
    readmission_rate_30 FLOAT,
    avg_time_in_hospital FLOAT,
    avg_num_medications FLOAT,
    avg_total_visits    FLOAT
);
