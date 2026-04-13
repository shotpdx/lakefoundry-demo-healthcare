# Backlog Item

**Title:** Diabetic Patient Treatment Outcome Comparison Pipeline

**Description:**
## Problem Statement
Analyze diabetic patients and compare outcomes by treatment.

## Goals
- Identify diabetic patients from OMOP data
- Define treatment groups for diabetic patients based on specific medications (Metformin, Insulin glargine, Glipizide)
- Compare mortality outcomes across different treatment groups
- Generate pre-calculated survival statistics using Kaplan-Meier estimator

## Data Source
- `sandbox_us_west_2.hls_demo_omop` - OMOP CDM tables (person, condition_occurrence, drug_exposure, death, observation_period)

## Output Tables
- `sandbox_us_west_2.hls_demo_omop_analytics.diabetic_cohort_summary` - Patient-level cohort data
- `sandbox_us_west_2.hls_demo_omop_analytics.survival_statistics` - Pre-calculated Kaplan-Meier statistics

## Technical Decisions
- Diabetic patients identified using ICD-10 code E11.9 (Type 2 Diabetes Mellitus)
- Treatment groups: Metformin, Insulin glargine, Glipizide from drug_source_value
- Patients assigned to first treatment received among the three
- Mortality observed until end of observation_period or drug_exposure_end_date (whichever is later)
- Implemented using Delta Live Tables (DLT) with batch processing

**Acceptance Criteria:**
## Output Tables
- [ ] `diabetic_cohort_summary` table exists with schema:
  - person_id (BIGINT) - Primary Key
  - treatment_group (STRING)
  - treatment_start_date (DATE)
  - observation_end_date (DATE)
  - mortality_status (INT)
  - date_of_death (DATE, NULLABLE)

- [ ] `survival_statistics` table exists with schema:
  - treatment_group (STRING)
  - time_point (INT)
  - survival_probability (DOUBLE)
  - lower_ci (DOUBLE)
  - upper_ci (DOUBLE)
  - num_at_risk (BIGINT)
  - num_events (BIGINT)

## Data Quality
- [ ] No null values in critical columns (person_id, condition_start_date, drug_exposure_start_date)
- [ ] All three treatment groups represented in output
- [ ] Survival probabilities between 0 and 1

## Pipeline
- [ ] DLT pipeline created and running
- [ ] Code complete and reviewed

## Pod Context

- **Pod ID**: demo-pod
- **UC Catalog**: sandbox_us_west_2
- **UC Schema**: lakefoundry
- **UC Volume**: artifacts

## Working Directory

Files should be created in the current sandbox directory.
