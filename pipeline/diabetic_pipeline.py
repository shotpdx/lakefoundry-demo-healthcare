# Databricks notebook source
# MAGIC %pip install lifelines

# COMMAND ----------
# MAGIC %md
# MAGIC # Diabetic Patient Treatment Outcome Comparison Pipeline
# MAGIC 
# MAGIC Delta Live Tables pipeline that:
# MAGIC 1. Reads OMOP source tables from `sandbox_us_west_2.hls_demo_omop`
# MAGIC 2. Identifies diabetic patients (ICD-10 E11.9)
# MAGIC 3. Assigns treatment groups (Metformin, Insulin glargine, Glipizide)
# MAGIC 4. Computes mortality outcomes within observation windows
# MAGIC 5. Pre-calculates Kaplan-Meier survival statistics

# COMMAND ----------

import sys
import os

# Add the bundle files root to sys.path so `pipeline` package can be imported.
# When deployed via DAB, this notebook lives at:
#   /Workspace/.../files/pipeline/diabetic_pipeline
# We need /Workspace/.../files/ on the path.
try:
    # In Databricks DLT, __file__ is defined as the workspace path of the notebook
    _notebook_path = __file__
    _notebook_dir = os.path.dirname(_notebook_path)
    _bundle_root = os.path.dirname(_notebook_dir)
    if _bundle_root not in sys.path:
        sys.path.insert(0, _bundle_root)
except Exception:
    # Fallback: try to derive path from dbutils
    try:
        _nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        _bundle_root = "/Workspace" + os.path.dirname(os.path.dirname(_nb_path))
        if _bundle_root not in sys.path:
            sys.path.insert(0, _bundle_root)
    except Exception:
        pass

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, LongType, StringType, DateType, IntegerType,
    DoubleType,
)
import pandas as pd

from pipeline.diabetic_cohort import (
    identify_diabetic_patients,
    assign_treatment_group,
    compute_observation_end,
    compute_mortality_status,
    TREATMENT_DRUGS,
)
from pipeline.survival import compute_survival_statistics

# ── Source catalog / schema ──────────────────────────────────────────────────
SOURCE_CATALOG = "sandbox_us_west_2"
SOURCE_SCHEMA = "hls_demo_omop"


def _src(table: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table}"


# ── Bronze: read source tables ───────────────────────────────────────────────

@dlt.table(
    name="bronze_condition_occurrence",
    comment="Raw condition_occurrence from OMOP CDM",
)
def bronze_condition_occurrence():
    return spark.read.table(_src("condition_occurrence"))


@dlt.table(
    name="bronze_drug_exposure",
    comment="Raw drug_exposure from OMOP CDM",
)
def bronze_drug_exposure():
    return spark.read.table(_src("drug_exposure"))


@dlt.table(
    name="bronze_observation_period",
    comment="Raw observation_period from OMOP CDM",
)
def bronze_observation_period():
    return spark.read.table(_src("observation_period"))


@dlt.table(
    name="bronze_death",
    comment="Raw death table from OMOP CDM",
)
def bronze_death():
    return spark.read.table(_src("death"))


# ── Silver: diabetic_cohort_summary ─────────────────────────────────────────

_COHORT_SCHEMA = StructType([
    StructField("person_id", LongType(), False),
    StructField("treatment_group", StringType(), False),
    StructField("treatment_start_date", DateType(), False),
    StructField("observation_end_date", DateType(), False),
    StructField("mortality_status", IntegerType(), False),
    StructField("date_of_death", DateType(), True),
])


@dlt.table(
    name="diabetic_cohort_summary",
    comment="One row per diabetic patient with treatment group and mortality outcome",
    schema=_COHORT_SCHEMA,
)
@dlt.expect_all({
    "person_id_not_null": "person_id IS NOT NULL",
    "treatment_group_not_null": "treatment_group IS NOT NULL",
    "treatment_start_date_not_null": "treatment_start_date IS NOT NULL",
    "observation_end_date_not_null": "observation_end_date IS NOT NULL",
    "mortality_status_not_null": "mortality_status IS NOT NULL",
    "mortality_status_valid": "mortality_status IN (0, 1)",
})
def diabetic_cohort_summary():
    # Pre-filter in Spark before converting to pandas (reduces driver memory pressure).
    # Read directly from source to avoid DLT dependency scheduling issues with bronze views.
    conditions_pd = (
        spark.read.table(_src("condition_occurrence"))
        .filter(F.col("condition_source_value") == "E11.9")
        .toPandas()
    )
    drugs_pd = (
        spark.read.table(_src("drug_exposure"))
        .filter(F.col("drug_source_value").isin(TREATMENT_DRUGS))
        .toPandas()
    )
    obs_pd = spark.read.table(_src("observation_period")).toPandas()
    death_pd = spark.read.table(_src("death")).toPandas()

    # 1. Identify diabetic patients (E11.9)
    diabetic = identify_diabetic_patients(conditions_pd)
    diabetic_ids = set(diabetic["person_id"].tolist())

    # 2. Filter drugs to diabetic patients only
    drugs_diabetic = drugs_pd[drugs_pd["person_id"].isin(diabetic_ids)].copy()
    obs_diabetic = obs_pd[obs_pd["person_id"].isin(diabetic_ids)].copy()

    # 3. Assign treatment groups (first treatment within observation period)
    treatment = assign_treatment_group(drugs_diabetic, obs_diabetic)
    if treatment.empty:
        return spark.createDataFrame([], _COHORT_SCHEMA)

    treatment_ids = set(treatment["person_id"].tolist())

    # 4. Join observation period end date onto treatment
    obs_end_raw = obs_diabetic[["person_id", "observation_period_end_date"]]
    dup_count = len(obs_end_raw) - obs_end_raw.drop_duplicates("person_id").shape[0]
    if dup_count > 0:
        print(f"[diabetic_cohort_summary] Dropping {dup_count} duplicate observation period rows")
    obs_end = obs_end_raw.drop_duplicates("person_id")
    treatment = treatment.merge(obs_end, on="person_id", how="left")

    # 5. Compute observation_end_date = max(obs_period_end, drug_exposure_end)
    treatment["observation_end_date"] = treatment.apply(
        lambda r: compute_observation_end(
            r["treatment_start_date"],
            r["drug_exposure_end_date"],
            r["observation_period_end_date"],
        ),
        axis=1,
    )

    # 6. Join death info
    death_relevant = death_pd[death_pd["person_id"].isin(treatment_ids)][
        ["person_id", "death_date"]
    ].drop_duplicates("person_id")
    treatment = treatment.merge(death_relevant, on="person_id", how="left")

    # 7. Compute mortality_status
    treatment["mortality_status"] = treatment.apply(
        lambda r: compute_mortality_status(
            r["observation_end_date"],
            r["death_date"] if pd.notna(r.get("death_date")) else None,
        ),
        axis=1,
    )
    # Replace NaN with None for date_of_death (Spark DateType cannot accept float NaN)
    treatment["date_of_death"] = treatment["death_date"].where(
        treatment["death_date"].notna(), other=None
    ) if "death_date" in treatment.columns else None

    # 8. Select and cast final columns
    result = treatment[[
        "person_id", "treatment_group", "treatment_start_date",
        "observation_end_date", "mortality_status", "date_of_death",
    ]].copy()
    # Ensure non-nullable columns have no nulls before casting
    assert result["person_id"].notna().all(), "person_id has unexpected null values"
    result["person_id"] = result["person_id"].astype("int64")
    result["mortality_status"] = result["mortality_status"].fillna(0).astype("int32")

    return spark.createDataFrame(result, schema=_COHORT_SCHEMA)


# ── Gold: survival_statistics ────────────────────────────────────────────────

_SURVIVAL_SCHEMA = StructType([
    StructField("treatment_group", StringType(), False),
    StructField("time_point", IntegerType(), False),
    StructField("survival_probability", DoubleType(), False),
    StructField("lower_ci", DoubleType(), False),
    StructField("upper_ci", DoubleType(), False),
    StructField("num_at_risk", LongType(), False),
    StructField("num_events", LongType(), False),
])


@dlt.table(
    name="survival_statistics",
    comment="Pre-calculated Kaplan-Meier survival statistics per treatment group",
    schema=_SURVIVAL_SCHEMA,
)
@dlt.expect_all({
    "treatment_group_not_null": "treatment_group IS NOT NULL",
    "time_point_non_negative": "time_point >= 0",
    "survival_probability_valid": "survival_probability >= 0 AND survival_probability <= 1",
})
def survival_statistics():
    # Read from the UC table directly to avoid DLT dependency scheduling issues
    cohort_pd = spark.read.table(
        f"{SOURCE_CATALOG}.hls_demo_omop_analytics.diabetic_cohort_summary"
    ).toPandas()
    stats_pd = compute_survival_statistics(cohort_pd)
    if stats_pd.empty:
        return spark.createDataFrame([], _SURVIVAL_SCHEMA)

    stats_pd["time_point"] = stats_pd["time_point"].astype("int32")
    stats_pd["num_at_risk"] = stats_pd["num_at_risk"].astype("int64")
    stats_pd["num_events"] = stats_pd["num_events"].astype("int64")

    return spark.createDataFrame(stats_pd, schema=_SURVIVAL_SCHEMA)
