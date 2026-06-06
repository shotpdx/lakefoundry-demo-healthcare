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

### Gold analytics assets

Task 2 adds the Gold survival analytics contract derived directly from `silver_diabetic_treatment_cohort`.

#### `gold_diabetic_treatment_survival_curve`

Business-friendly Kaplan-Meier style survival curve by `treatment_group` with reviewer-facing fields:

- `treatment_group`
- `silver_source_table`
- `silver_source_key` as `silver_diabetic_treatment_cohort::<treatment_group>`
- `silver_lineage_layer` (`silver`)
- `gold_analytics_version` (`v1`)
- `time_to_event_days`
- `survival_probability`
- `survival_percent`
- `lower_ci`
- `upper_ci`
- `confidence_interval_width`
- `num_at_risk`
- `num_events`

#### `gold_diabetic_treatment_survival_summary`

Business-ready treatment rollup for downstream SQL or dashboard use:

- `treatment_group`
- `silver_source_table`
- `silver_source_key`
- `silver_lineage_layer`
- `gold_analytics_version`
- `cohort_size`
- `total_events`
- `event_rate`
- `event_rate_percent`
- `avg_follow_up_days`
- `median_follow_up_days`
- `max_follow_up_days`
- `median_survival_days`
- `median_survival_reached`
- `latest_time_point_days`
- `latest_survival_probability`
- `latest_survival_percent`
- `latest_num_at_risk`
- `latest_num_events`

These Gold outputs intentionally expose explicit lineage markers back to the Silver cohort so reviewers can validate that the analytics surface is queryable and traceable without requiring workspace deployment evidence in this task.

### Source assumptions for reviewers

- Diabetes inclusion is currently based on `condition_source_value = '44054006'`.
- Treatment conformance uses OMOP concept joins on `drug_source_value = concept_code`, with pattern-based grouping over concept name, concept code, and raw source value.
- The Silver cohort keeps only records with non-negative follow-up where `observation_end_date >= treatment_start_date`.
- Gold lineage is currently treatment-group scoped through `silver_source_table`, `silver_source_key`, `silver_lineage_layer`, and `gold_analytics_version`.
- Task 3 owns deployment and workspace verification. The Databricks bundle now passes `databricks bundle validate` and deploys the medallion pipeline into `${var.catalog}.${var.schema}` with pipeline configuration keys `source_catalog` and `source_schema` for the upstream OMOP source location.
- Workspace execution still depends on the configured source catalog/schema being readable and containing the expected OMOP tables (`person`, `condition_occurrence`, `drug_exposure`, `death`, `observation_period`, `concept`). Preserve failed run evidence if that upstream source is unavailable and update the bundle target variables before retrying.

### Local quality expectations for Task 2

Local tests now validate:

- Gold curve metric correctness for event counting, same-day outcomes, and event-only latest-point selection
- Gold summary business metrics such as event rate percent, event-driven median survival, and latest survival rollups
- explicit Gold-to-Silver lineage markers and stable business-friendly column contracts
- Silver quality thresholds for lineage completeness, mortality consistency, and valid follow-up bounds
- Gold quality thresholds for probability ranges, confidence interval ordering, event-row latest point selection, median-on-event correctness, at-risk/event count sanity, and percent-to-ratio consistency

### Local verification for Tasks 1-2

- `pytest tests/test_transformations.py tests/test_data_quality.py -x`
- `databricks bundle validate`
- `databricks bundle deploy`
- `databricks bundle run diabetic_outcomes_pipeline`
- If the pipeline run fails before materializing Bronze/Silver/Gold assets, keep the failed update log as acceptance evidence and correct the bundle workspace/source configuration before rerunning.

