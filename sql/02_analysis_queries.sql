-- =============================================================
-- Hospital Readmission Risk Intelligence System
-- Script 02: Core Analytical Queries
-- (Run these after Python has loaded data into patient_features)
-- =============================================================

USE hospital_readmission;

-- ---------------------------------------------------------------
-- Q1: Overall readmission rate
-- ---------------------------------------------------------------
SELECT
    COUNT(*)                                              AS total_encounters,
    SUM(readmitted_30)                                    AS readmitted_under_30,
    ROUND(100.0 * SUM(readmitted_30) / COUNT(*), 2)      AS readmission_rate_30_pct,
    SUM(readmitted_any)                                   AS readmitted_any,
    ROUND(100.0 * SUM(readmitted_any) / COUNT(*), 2)     AS readmission_rate_any_pct
FROM patient_features;


-- ---------------------------------------------------------------
-- Q2: Readmission rate by age group
-- ---------------------------------------------------------------
SELECT
    CASE
        WHEN age_mid BETWEEN 0  AND 30  THEN '0-30'
        WHEN age_mid BETWEEN 31 AND 50  THEN '31-50'
        WHEN age_mid BETWEEN 51 AND 70  THEN '51-70'
        ELSE '71+'
    END                                                   AS age_group,
    COUNT(*)                                              AS total,
    SUM(readmitted_30)                                    AS readmitted_30,
    ROUND(100.0 * SUM(readmitted_30) / COUNT(*), 2)      AS rate_pct,
    ROUND(AVG(time_in_hospital), 2)                       AS avg_los,
    ROUND(AVG(num_medications), 2)                        AS avg_medications
FROM patient_features
GROUP BY age_group
ORDER BY age_group;


-- ---------------------------------------------------------------
-- Q3: High-risk patient profile
--     (patients with high total visits AND many medications)
-- ---------------------------------------------------------------
SELECT
    encounter_id,
    patient_nbr,
    age_mid,
    time_in_hospital,
    num_medications,
    total_visits,
    on_insulin,
    a1c_high,
    readmitted_30
FROM patient_features
WHERE total_visits >= 5
  AND num_medications >= 15
ORDER BY total_visits DESC, num_medications DESC
LIMIT 100;


-- ---------------------------------------------------------------
-- Q4: Does insulin change correlate with lower readmission?
-- ---------------------------------------------------------------
SELECT
    on_insulin,
    change_meds,
    COUNT(*)                                              AS encounters,
    SUM(readmitted_30)                                    AS readmitted_30,
    ROUND(100.0 * SUM(readmitted_30) / COUNT(*), 2)      AS readmission_rate_pct,
    ROUND(AVG(time_in_hospital), 2)                       AS avg_los
FROM patient_features
GROUP BY on_insulin, change_meds
ORDER BY on_insulin, change_meds;


-- ---------------------------------------------------------------
-- Q5: A1C testing and readmission — does testing matter?
-- ---------------------------------------------------------------
SELECT
    has_a1c_result,
    a1c_high,
    COUNT(*)                                              AS encounters,
    ROUND(100.0 * SUM(readmitted_30) / COUNT(*), 2)      AS readmission_rate_30_pct,
    ROUND(AVG(num_medications), 2)                        AS avg_medications
FROM patient_features
GROUP BY has_a1c_result, a1c_high
ORDER BY has_a1c_result, a1c_high;


-- ---------------------------------------------------------------
-- Q6: Diagnosis category impact on readmission
-- ---------------------------------------------------------------
SELECT
    diag_diabetes,
    diag_circulatory,
    diag_respiratory,
    COUNT(*)                                              AS encounters,
    ROUND(100.0 * SUM(readmitted_30) / COUNT(*), 2)      AS readmission_rate_30_pct,
    ROUND(AVG(time_in_hospital), 2)                       AS avg_los
FROM patient_features
GROUP BY diag_diabetes, diag_circulatory, diag_respiratory
ORDER BY readmission_rate_30_pct DESC;


-- ---------------------------------------------------------------
-- Q7: Model performance summary from predictions log
-- ---------------------------------------------------------------
SELECT
    model_name,
    COUNT(*)                                              AS predictions_made,
    SUM(CASE WHEN predicted_label = actual_label THEN 1 ELSE 0 END) AS correct,
    ROUND(100.0 * SUM(CASE WHEN predicted_label = actual_label
                           THEN 1 ELSE 0 END) / COUNT(*), 2)        AS accuracy_pct,
    ROUND(AVG(predicted_proba), 4)                        AS avg_predicted_proba,
    run_timestamp
FROM model_predictions
GROUP BY model_name, run_timestamp
ORDER BY run_timestamp DESC;


-- ---------------------------------------------------------------
-- Q8: Populate KPI summary table for Tableau
-- ---------------------------------------------------------------
TRUNCATE TABLE readmission_kpis;

-- Overall
INSERT INTO readmission_kpis
    (segment_type, segment_value, total_encounters,
     readmitted_30_count, readmission_rate_30,
     avg_time_in_hospital, avg_num_medications, avg_total_visits)
SELECT
    'overall', 'All Patients',
    COUNT(*),
    SUM(readmitted_30),
    ROUND(SUM(readmitted_30) / COUNT(*), 4),
    ROUND(AVG(time_in_hospital), 2),
    ROUND(AVG(num_medications), 2),
    ROUND(AVG(total_visits), 2)
FROM patient_features;

-- By age group
INSERT INTO readmission_kpis
    (segment_type, segment_value, total_encounters,
     readmitted_30_count, readmission_rate_30,
     avg_time_in_hospital, avg_num_medications, avg_total_visits)
SELECT
    'age_group',
    CASE
        WHEN age_mid BETWEEN 0  AND 30  THEN '0-30'
        WHEN age_mid BETWEEN 31 AND 50  THEN '31-50'
        WHEN age_mid BETWEEN 51 AND 70  THEN '51-70'
        ELSE '71+'
    END,
    COUNT(*),
    SUM(readmitted_30),
    ROUND(SUM(readmitted_30) / COUNT(*), 4),
    ROUND(AVG(time_in_hospital), 2),
    ROUND(AVG(num_medications), 2),
    ROUND(AVG(total_visits), 2)
FROM patient_features
GROUP BY segment_value;

-- By insulin use
INSERT INTO readmission_kpis
    (segment_type, segment_value, total_encounters,
     readmitted_30_count, readmission_rate_30,
     avg_time_in_hospital, avg_num_medications, avg_total_visits)
SELECT
    'insulin',
    CASE WHEN on_insulin = 1 THEN 'On Insulin' ELSE 'Not on Insulin' END,
    COUNT(*),
    SUM(readmitted_30),
    ROUND(SUM(readmitted_30) / COUNT(*), 4),
    ROUND(AVG(time_in_hospital), 2),
    ROUND(AVG(num_medications), 2),
    ROUND(AVG(total_visits), 2)
FROM patient_features
GROUP BY on_insulin;
