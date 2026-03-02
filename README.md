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
