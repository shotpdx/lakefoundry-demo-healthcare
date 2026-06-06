# Healthcare & Life Sciences Demo

**AI-DLC Stage: Ideation**

Use case: Real World Evidence (RWE) - Cohort analysis, treatment comparison, outcomes research

## Demo Flow

1. User describes RWE use case to ideation agent
2. Agent discovers OMOP tables in Unity Catalog
3. Agent generates spec for cohort building and analysis pipeline
4. Hand off to execution plane

## Available Data

**Catalog:** `sandbox_us_west_2.hls_demo_omop`

| Table | Rows | Description |
|-------|------|-------------|
| person | 10,000 | Patient demographics |
| condition_occurrence | 16,348 | Diagnoses (diabetes, HTN, heart failure, COPD, CKD) |
| drug_exposure | 12,206 | Medications (metformin, insulin, statins, ACE inhibitors) |
| procedure_occurrence | 9,772 | Labs/procedures (HbA1c, lipid panels, ECGs, echo) |
| visit_occurrence | 68,024 | Outpatient, inpatient, ER encounters |
| observation_period | 10,000 | Patient observation windows |
| death | 201 | Mortality outcomes |

## Sample RWE Questions for Demo

- "Build a cohort of diabetic patients and compare outcomes by treatment"
- "Analyze heart failure patients - what medications correlate with better outcomes?"
- "Create a propensity-matched comparison of metformin vs sulfonylureas"

## To Run Demo

```bash
# Start ideation conversation
lakefoundry ideate --pod-id hls-demo --catalog sandbox_us_west_2 --schema hls_demo_omop
```

Part of the Lakefoundry AI-DLC demo suite.

## Current medallion transformation scope

Task 1 completes the local Bronze and Silver transformation contract for the diabetic outcomes pipeline.

### Bronze OMOP assets

The Bronze layer materializes the OMOP source tables needed by the downstream diabetic cohort logic at operational grain:

- `bronze_omop_person`
- `bronze_omop_condition_occurrence`
- `bronze_omop_drug_exposure`
- `bronze_omop_death`
- `bronze_omop_observation_period`
- `bronze_omop_concept`

Each Bronze table preserves source-grain records and appends traceability metadata:

- `bronze_source_table`
- `bronze_source_key`
- `bronze_ingested_at`

For the current task scope, the Silver cohort directly depends on condition, drug exposure, death, observation period, and concept Bronze tables. `bronze_omop_person` is still materialized because it is part of the curated OMOP Bronze surface for future person-level enrichment.

### Silver cohort asset

The Silver cohort output is `silver_diabetic_treatment_cohort`. It implements:

- deterministic one-row-per-person identity resolution for diabetes diagnosis selection
- deterministic first-treatment selection per person
- conformed treatment grouping for Metformin, Insulin Glargine, and Glipizide
- follow-up validation using observation windows and death dates
- audit and lineage columns back to Bronze records for diabetes, drug exposure, observation period, and death inputs
- `patient_identity_key` and `silver_conformed_at` fields for identity and audit tracking

### Source assumptions for reviewers

- Diabetes inclusion is currently based on `condition_source_value = '44054006'`.
- Treatment conformance uses OMOP concept joins on `drug_source_value = concept_code`, with pattern-based grouping over concept name, concept code, and raw source value.
- The Silver cohort keeps only records with non-negative follow-up where `observation_end_date >= treatment_start_date`.
- Deployment and Databricks workspace execution are intentionally deferred; this task only claims local code and test evidence.

### Local verification for Task 1

- `pytest tests/test_transformations.py tests/test_data_quality.py -x`

