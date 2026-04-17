# Diabetic Patient Treatment Outcome Comparison Pipeline - Implementation Plan

**Goal:** Build a DLT pipeline that identifies diabetic patients, assigns them to treatment groups (Metformin, Insulin glargine, Glipizide), and generates Kaplan-Meier survival statistics.

**Architecture:** Python-based Spark Declarative Pipeline (SDP) with materialized views. Source data from OMOP CDM tables in `databricks_observational_medical_outcomes_partnership_omop_common_data_model_cdm.patient_risk_altered_omop`. Output to `lakefoundry_dev.hls_demo_omop_analytics`. Pipeline uses batch processing with two output tables: `diabetic_cohort_summary` (patient-level) and `survival_statistics` (Kaplan-Meier curves by treatment group).

**Data Notes:**
- Diabetes identified via SNOMED code `44054006` (Type 2 diabetes mellitus, concept_id 201826)
- Drugs matched via concept table joins on drug_source_value → concept_code where concept_name contains medication names
- Death table may have zero records - pipeline handles this gracefully with censoring logic

---

### Task 1: Create DLT Pipeline Source Code

**Depends On:** none

**Files:**
- Create: `src/diabetic_outcomes/transformations/diabetic_cohort_summary.py`
- Create: `src/diabetic_outcomes/transformations/survival_statistics.py`

**Requirements:**
- Use modern `pyspark.pipelines` API (`from pyspark import pipelines as dp`)
- `diabetic_cohort_summary` materialized view:
  - Join person, condition_occurrence, drug_exposure, death, observation_period tables
  - Filter diabetic patients: condition_source_value = '44054006' (SNOMED for Type 2 DM)
  - Identify treatment groups by joining drug_exposure → concept table where concept_name LIKE '%Metformin%', '%Insulin Glargine%', or '%Glipizide%'
  - Assign patient to FIRST treatment received among the three (by drug_exposure_start_date)
  - Calculate observation_end_date as GREATEST(observation_period_end_date, drug_exposure_end_date)
  - Set mortality_status = 1 if death record exists, 0 otherwise
  - Output schema: person_id (BIGINT), treatment_group (STRING), treatment_start_date (DATE), observation_end_date (DATE), mortality_status (INT), date_of_death (DATE nullable)
- `survival_statistics` materialized view:
  - Read from diabetic_cohort_summary
  - Calculate time_to_event in days from treatment_start_date to observation_end_date (or death)
  - Implement Kaplan-Meier estimator using window functions
  - Generate time_point (days), survival_probability, lower_ci, upper_ci (95% Greenwood CI), num_at_risk, num_events per treatment_group
  - Handle edge cases: zero deaths (survival = 1.0 throughout), single patient groups

**Acceptance Criteria:**
- [ ] Both Python files use `@dp.materialized_view()` decorator
- [ ] Source tables referenced with full catalog.schema.table paths
- [ ] No hardcoded credentials or paths
- [ ] Survival probabilities calculated correctly (monotonically decreasing)

**Skills:** spark-declarative-pipelines (read 5-python-api.md for dp API)

---

### Task 2: Create DAB Bundle Configuration

**Depends On:** 1

**Files:**
- Create: `databricks.yml`
- Create: `resources/diabetic_outcomes_pipeline.yml`

**Requirements:**
- Bundle name: `diabetic-outcomes-pipeline`
- Single dev target with mode: development
- Pipeline resource configuration:
  - name: `diabetic_outcomes_etl`
  - catalog: `lakefoundry_dev`
  - schema: `hls_demo_omop_analytics`
  - serverless: true
  - libraries pointing to `src/diabetic_outcomes/transformations/`
- Include pattern: `resources/*.yml`

**Acceptance Criteria:**
- [ ] `databricks bundle validate` passes
- [ ] Pipeline targets correct catalog/schema
- [ ] Serverless compute enabled

**Skills:** asset-bundles, spark-declarative-pipelines (read SDP_guidance.md)

---

### Task 3: Deploy and Run Pipeline

**Depends On:** 2

**Files:**
- Modify: None (deployment task)

**Requirements:**
- Run `databricks bundle deploy -t dev`
- Run `databricks bundle run diabetic_outcomes_etl -t dev`
- Wait for pipeline completion
- Verify output tables exist and have data

**Acceptance Criteria:**
- [ ] Pipeline deploys successfully
- [ ] Pipeline runs to completion without errors
- [ ] `diabetic_cohort_summary` table exists with correct schema
- [ ] `survival_statistics` table exists with correct schema
- [ ] All three treatment groups represented in output
- [ ] Survival probabilities between 0 and 1
- [ ] No null values in critical columns (person_id, treatment_group, treatment_start_date)

**Skills:** asset-bundles

---

### Task 4: Data Quality Validation

**Depends On:** 3

**Files:**
- Create: `tests/test_diabetic_outcomes.py`

**Requirements:**
- Write pytest tests that query the output tables via SQL
- Validate schema matches acceptance criteria
- Validate data quality:
  - No nulls in required columns
  - All three treatment groups present
  - Survival probabilities in [0, 1] range
  - num_at_risk decreases or stays same over time
  - num_events >= 0
- Tests should use `tool_dispatcher` or direct SQL execution

**Acceptance Criteria:**
- [ ] All tests pass
- [ ] Tests cover schema validation
- [ ] Tests cover data quality rules from spec

**Skills:** None (standard pytest)
