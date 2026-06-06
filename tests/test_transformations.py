"""Unit tests for transformation logic.

These tests require a Spark session and are skipped in environments
where Spark is not available or properly configured.
"""
import sys
from pathlib import Path
import types
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Mock pyspark.pipelines if not available
if 'pyspark.pipelines' not in sys.modules:
    sys.modules['pyspark.pipelines'] = types.SimpleNamespace(materialized_view=lambda **kwargs: (lambda f: f))

# Try to import SparkSession
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkSession = None

from diabetic_outcomes.transformations import bronze_omop as bronze
from diabetic_outcomes.transformations import diabetic_cohort_summary as dcs
from diabetic_outcomes.transformations import survival_statistics as ss


@pytest.fixture
def spark_session():
    """Create a Spark session for testing, skip if not available."""
    if not SPARK_AVAILABLE:
        pytest.skip("PySpark not available")
    try:
        return SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    except Exception as e:
        pytest.skip(f"Could not create Spark session: {e}")


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_bronze_condition_occurrence_preserves_operational_grain(spark_session, monkeypatch):
    spark = spark_session

    source_rows = [
        (1001, 1, 999, "2020-01-01", "2020-01-02", bronze.DIABETES_CONDITION_CODE if hasattr(bronze, "DIABETES_CONDITION_CODE") else "44054006"),
        (1002, 1, 999, "2020-02-01", None, "999999"),
    ]
    source_df = spark.createDataFrame(
        source_rows,
        [
            "condition_occurrence_id",
            "person_id",
            "condition_concept_id",
            "condition_start_date",
            "condition_end_date",
            "condition_source_value",
        ],
    )

    class Reader:
        def table(self, name):
            assert name == bronze._source("condition_occurrence")
            return source_df

    monkeypatch.setattr(bronze, "spark", MagicMock(read=Reader()))
    result = bronze.bronze_omop_condition_occurrence().collect()

    assert len(result) == 2
    assert {row.condition_occurrence_id for row in result} == {1001, 1002}
    assert all(row.bronze_source_table == "condition_occurrence" for row in result)
    assert {row.bronze_source_key for row in result} == {"1001", "1002"}
    assert all(row.bronze_ingested_at is not None for row in result)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_silver_diabetic_treatment_cohort_respects_observation_bounds(spark_session):
    spark = spark_session

    condition = spark.createDataFrame(
        [
            (101, 1, "2020-01-01", dcs.DIABETES_CONDITION_CODE, "condition-101", "2024-01-01 01:00:00"),
            (201, 2, "2020-03-01", dcs.DIABETES_CONDITION_CODE, "condition-201", "2024-01-01 02:00:00"),
        ],
        [
            "condition_occurrence_id",
            "person_id",
            "condition_start_date",
            "condition_source_value",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    drug = spark.createDataFrame(
        [
            (1001, 1, "metformin rxnorm", "2020-01-01", "2020-12-31", "drug-1001", "2024-01-02 01:00:00"),
            (2001, 2, "glipizide rxnorm", "2020-03-01", "2020-04-01", "drug-2001", "2024-01-02 02:00:00"),
        ],
        [
            "drug_exposure_id",
            "person_id",
            "drug_source_value",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    death = spark.createDataFrame(
        [(1, None, None, None, None), (2, "2020-07-01", None, "death-2", "2024-01-04 01:00:00")],
        ["person_id", "death_date", "death_datetime", "bronze_source_key", "bronze_ingested_at"],
    )
    observation = spark.createDataFrame(
        [
            (301, 1, "2019-01-01", "2020-06-01", "obs-301", "2024-01-03 01:00:00"),
            (302, 1, "2019-06-01", "2020-05-15", "obs-302", "2024-01-03 02:00:00"),
            (401, 2, "2020-01-01", "2020-05-01", "obs-401", "2024-01-03 03:00:00"),
        ],
        [
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    concept = spark.createDataFrame(
        [
            ("metformin rxnorm", "Metformin HCl"),
            ("glipizide rxnorm", "Glipizide"),
        ],
        ["concept_code", "concept_name"],
    )

    result = dcs.build_diabetic_cohort_summary(condition, drug, death, observation, concept).collect()
    assert len(result) == 2

    person_one = next(row for row in result if row.person_id == 1)
    assert person_one.observation_end_date.isoformat() == "2020-06-01"

    person_two = next(row for row in result if row.person_id == 2)
    assert person_two.observation_end_date.isoformat() == "2020-05-01"
    assert person_two.mortality_status == 0
    assert person_two.date_of_death is None

    assert all(r.patient_identity_key for r in result)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_silver_diabetic_treatment_cohort_filters_invalid_follow_up_and_keeps_one_row_per_person(spark_session):
    spark = spark_session

    condition = spark.createDataFrame(
        [
            (101, 1, "2020-01-15", dcs.DIABETES_CONDITION_CODE, "condition-101", "2024-01-01 01:00:00"),
            (102, 1, "2020-01-01", dcs.DIABETES_CONDITION_CODE, "condition-102", "2024-01-01 00:00:00"),
            (201, 2, "2020-02-01", dcs.DIABETES_CONDITION_CODE, "condition-201", "2024-01-01 02:00:00"),
            (301, 3, "2020-03-01", dcs.DIABETES_CONDITION_CODE, "condition-301", "2024-01-01 03:00:00"),
        ],
        [
            "condition_occurrence_id",
            "person_id",
            "condition_start_date",
            "condition_source_value",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    drug = spark.createDataFrame(
        [
            (1001, 1, "Metformin 500 MG Oral Tablet", "2020-01-10", "2020-01-31", "drug-1001", "2024-01-02 01:00:00"),
            (1002, 1, "GLIPIZIDE ER", "2020-01-20", "2020-02-15", "drug-1002", "2024-01-02 02:00:00"),
            (2001, 2, "insulin glargine prefilled pen", "2020-02-01", "2020-02-28", "drug-2001", "2024-01-02 03:00:00"),
            (3001, 3, "GLIPIZIDE ER", "2020-03-01", "2020-03-31", "drug-3001", "2024-01-02 04:00:00"),
        ],
        [
            "drug_exposure_id",
            "person_id",
            "drug_source_value",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    death = spark.createDataFrame(
        [
            (2, "2020-02-15", None, "death-2", "2024-01-04 01:00:00"),
            (2, "2020-02-14", None, "death-older-2", "2024-01-03 01:00:00"),
        ],
        ["person_id", "death_date", "death_datetime", "bronze_source_key", "bronze_ingested_at"],
    )
    observation = spark.createDataFrame(
        [
            (301, 1, "2019-01-01", "2020-06-01", "obs-301", "2024-01-03 01:00:00"),
            (302, 1, "2019-06-01", "2020-06-01", "obs-302", "2024-01-03 02:00:00"),
            (401, 2, "2020-01-01", "2020-05-01", "obs-401", "2024-01-03 02:00:00"),
            (501, 3, "2020-01-01", "2020-02-15", "obs-501", "2024-01-03 03:00:00"),
        ],
        [
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    concept = spark.createDataFrame([], "concept_code string, concept_name string")

    result = dcs.build_diabetic_cohort_summary(condition, drug, death, observation, concept).collect()

    assert {(r.person_id, r.treatment_group) for r in result} == {
        (1, "Metformin"),
        (2, "Insulin Glargine"),
    }
    assert len(result) == len({r.person_id for r in result})

    person_one = next(row for row in result if row.person_id == 1)
    assert person_one.diabetes_condition_occurrence_id == 102
    assert person_one.drug_exposure_id == 1001
    assert person_one.diabetes_condition_bronze_source_key == "condition-102"
    assert person_one.drug_exposure_bronze_source_key == "drug-1001"
    assert person_one.observation_period_bronze_source_key == "obs-302"
    assert person_one.observation_period_bronze_table == bronze.BRONZE_OBSERVATION_PERIOD
    assert person_one.death_bronze_table is None

    person_two = next(row for row in result if row.person_id == 2)
    assert person_two.mortality_status == 1
    assert person_two.date_of_death.isoformat() == "2020-02-15"
    assert person_two.death_bronze_source_key == "death-2"
    assert person_two.patient_identity_key != person_one.patient_identity_key
    assert all(r.observation_end_date >= r.treatment_start_date for r in result)
    assert all(r.person_id != 3 for r in result)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_bronze_omop_death_source_key_includes_datetime(spark_session, monkeypatch):
    spark = spark_session
    source_df = spark.createDataFrame(
        [
            (1, "2020-01-01", "2020-01-01 00:00:00"),
            (1, "2020-01-01", "2020-01-01 12:00:00"),
        ],
        ["person_id", "death_date", "death_datetime"],
    )

    class Reader:
        def table(self, name):
            assert name == bronze._source("death")
            return source_df

    monkeypatch.setattr(bronze, "spark", MagicMock(read=Reader()))
    result = bronze.bronze_omop_death().collect()

    assert len(result) == 2
    assert len({row.bronze_source_key for row in result}) == 2
    assert all("2020-01-01" in row.bronze_source_key for row in result)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_gold_survival_curve_counts_events_exactly_at_time_point(spark_session):
    spark = spark_session
    cohort = spark.createDataFrame([
        (1, "identity-1", "Metformin", "2020-01-01", "2020-01-10", 1, "2020-01-10"),
        (2, "identity-2", "Metformin", "2020-01-01", "2020-01-20", 0, None),
        (3, "identity-3", "Metformin", "2020-01-01", "2020-01-10", 1, "2020-01-10"),
    ], ["person_id", "patient_identity_key", "treatment_group", "treatment_start_date", "observation_end_date", "mortality_status", "date_of_death"])

    out = ss.build_survival_statistics(cohort).collect()
    rows = {r.time_to_event_days: r for r in out}
    assert rows[9].num_events == 2
    assert rows[19].num_events == 0
    assert rows[9].silver_source_table == dcs.SILVER_DIABETIC_TREATMENT_COHORT
    assert rows[9].survival_percent == pytest.approx(rows[9].survival_probability * 100.0)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_gold_survival_curve_excludes_non_positive_follow_up_windows(spark_session):
    spark = spark_session
    cohort = spark.createDataFrame([
        (1, "identity-1", "Metformin", "2020-01-01", "2020-01-01", 1, "2020-01-01"),
        (2, "identity-2", "Metformin", "2020-01-01", "2020-01-05", 1, "2020-01-05"),
        (3, "identity-3", "Metformin", "2020-01-01", "2020-01-01", 0, None),
        (4, "identity-4", "Metformin", "2020-01-01", "2020-01-10", 0, None),
    ], ["person_id", "patient_identity_key", "treatment_group", "treatment_start_date", "observation_end_date", "mortality_status", "date_of_death"])

    out = ss.build_survival_statistics(cohort).collect()
    rows = {r.time_to_event_days: r for r in out}

    assert set(rows) == {4, 9}
    assert rows[4].num_events == 1
    assert rows[4].num_at_risk == 2
    assert rows[9].num_events == 0
    assert rows[9].num_at_risk == 1


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_gold_survival_summary_provides_business_ready_treatment_rollup(spark_session):
    spark = spark_session
    cohort = spark.createDataFrame([
        (1, "identity-1", "Metformin", "2020-01-01", "2020-01-10", 1, "2020-01-10"),
        (2, "identity-2", "Metformin", "2020-01-01", "2020-01-20", 0, None),
        (3, "identity-3", "Glipizide", "2020-01-01", "2020-01-08", 1, "2020-01-08"),
        (4, "identity-4", "Glipizide", "2020-01-01", "2020-01-12", 1, "2020-01-12"),
    ], ["person_id", "patient_identity_key", "treatment_group", "treatment_start_date", "observation_end_date", "mortality_status", "date_of_death"])

    out = {row.treatment_group: row for row in ss.build_survival_summary(cohort).collect()}

    metformin = out["Metformin"]
    assert metformin.silver_source_table == dcs.SILVER_DIABETIC_TREATMENT_COHORT
    assert metformin.cohort_size == 2
    assert metformin.total_events == 1
    assert metformin.event_rate == pytest.approx(0.5)
    assert metformin.event_rate_percent == pytest.approx(50.0)
    assert metformin.latest_time_point_days == 19
    assert metformin.latest_survival_probability == pytest.approx(0.5)
    assert metformin.latest_survival_percent == pytest.approx(50.0)
    assert metformin.median_survival_days == 9
    assert metformin.median_survival_reached is True

    glipizide = out["Glipizide"]
    assert glipizide.cohort_size == 2
    assert glipizide.total_events == 2
    assert glipizide.event_rate == pytest.approx(1.0)
    assert glipizide.median_survival_days == 7
    assert glipizide.median_survival_reached is True
    assert glipizide.latest_num_at_risk == 1
    assert glipizide.latest_num_events == 1


def test_gold_asset_names_follow_medallion_conventions():
    assert ss.GOLD_TREATMENT_SURVIVAL_CURVE == "gold_diabetic_treatment_survival_curve"
    assert ss.GOLD_TREATMENT_SURVIVAL_SUMMARY == "gold_diabetic_treatment_survival_summary"
    assert ss.GOLD_TREATMENT_SURVIVAL_CURVE.startswith("gold_")
    assert ss.GOLD_TREATMENT_SURVIVAL_SUMMARY.startswith("gold_")
