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
        (1001, 1, 999, "2020-01-01", "2020-01-02", dcs.DIABETES_CONDITION_CODE),
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
    result_df = bronze.bronze_omop_condition_occurrence()
    result = result_df.collect()

    assert len(result) == 2
    assert {row.condition_occurrence_id for row in result} == {1001, 1002}
    assert result_df.columns == [
        "condition_occurrence_id",
        "person_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_end_date",
        "condition_source_value",
        "bronze_source_table",
        "bronze_source_key",
        "bronze_ingested_at",
    ]
    assert all(row.bronze_source_table == "condition_occurrence" for row in result)
    assert {row.bronze_source_key for row in result} == {"1001", "1002"}
    assert all(row.bronze_ingested_at is not None for row in result)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_bronze_person_and_concept_support_silver_identity_and_treatment_conformance(spark_session, monkeypatch):
    spark = spark_session
    person_df = spark.createDataFrame(
        [(1, 8507, 1980, 8527, 38003564)],
        ["person_id", "gender_concept_id", "year_of_birth", "race_concept_id", "ethnicity_concept_id"],
    )
    concept_df = spark.createDataFrame(
        [(111, "metformin", "Metformin Hydrochloride", "RxNorm")],
        ["concept_id", "concept_code", "concept_name", "vocabulary_id"],
    )

    class Reader:
        def table(self, name):
            if name == bronze._source("person"):
                return person_df
            if name == bronze._source("concept"):
                return concept_df
            raise AssertionError(f"Unexpected table request: {name}")

    monkeypatch.setattr(bronze, "spark", MagicMock(read=Reader()))

    bronze_person = bronze.bronze_omop_person().collect()
    bronze_concept = bronze.bronze_omop_concept().collect()

    assert bronze_person[0].person_id == 1
    assert bronze_person[0].bronze_source_table == "person"
    assert bronze_person[0].bronze_source_key == "1"
    assert bronze_concept[0].concept_id == 111
    assert bronze_concept[0].concept_code == "metformin"
    assert bronze_concept[0].bronze_source_table == "concept"
    assert bronze_concept[0].bronze_source_key == "111"


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_bronze_table_names_cover_all_required_silver_inputs():
    assert bronze.BRONZE_TABLE_NAMES == (
        bronze.BRONZE_PERSON,
        bronze.BRONZE_CONDITION_OCCURRENCE,
        bronze.BRONZE_DRUG_EXPOSURE,
        bronze.BRONZE_DEATH,
        bronze.BRONZE_OBSERVATION_PERIOD,
        bronze.BRONZE_CONCEPT,
    )
    assert set(bronze.BRONZE_TABLE_NAMES) >= {
        bronze.BRONZE_CONDITION_OCCURRENCE,
        bronze.BRONZE_DRUG_EXPOSURE,
        bronze.BRONZE_DEATH,
        bronze.BRONZE_OBSERVATION_PERIOD,
        bronze.BRONZE_CONCEPT,
    }
    assert bronze.BRONZE_PERSON in bronze.BRONZE_TABLE_NAMES
    assert len(bronze.BRONZE_TABLE_NAMES) == len(set(bronze.BRONZE_TABLE_NAMES))
    assert all(name.startswith("bronze_omop_") for name in bronze.BRONZE_TABLE_NAMES)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_silver_diabetic_cohort_includes_lineage_and_audit_columns(spark_session):
    spark = spark_session

    condition = spark.createDataFrame(
        [(101, 1, "2020-01-01", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-101", "2024-01-01 01:00:00")],
        [
            "condition_occurrence_id",
            "person_id",
            "condition_start_date",
            "condition_source_value",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    drug = spark.createDataFrame(
        [(1001, 1, "metformin", "2020-01-05", "2020-02-01", bronze.BRONZE_DRUG_EXPOSURE, "drug-1001", "2024-01-02 01:00:00")],
        [
            "drug_exposure_id",
            "person_id",
            "drug_source_value",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    death = spark.createDataFrame([], "person_id bigint, death_date string, death_datetime string, bronze_source_table string, bronze_source_key string, bronze_ingested_at string")
    observation = spark.createDataFrame(
        [(301, 1, "2019-01-01", "2020-06-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-301", "2024-01-03 01:00:00")],
        [
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    concept = spark.createDataFrame(
        [(111, "metformin", "Metformin Hydrochloride", "RxNorm")],
        ["concept_id", "concept_code", "concept_name", "vocabulary_id"],
    )

    result_df = dcs.build_diabetic_cohort_summary(condition.alias("c"), drug.alias("d"), death.alias("de"), observation.alias("o"), concept.alias("co"))
    row = result_df.collect()[0]

    assert row.patient_identity_key
    assert row.diabetes_condition_bronze_table == bronze.BRONZE_CONDITION_OCCURRENCE
    assert row.diabetes_condition_bronze_source_key == "condition-101"
    assert row.drug_exposure_bronze_table == bronze.BRONZE_DRUG_EXPOSURE
    assert row.drug_exposure_bronze_source_key == "drug-1001"
    assert row.observation_period_bronze_table == bronze.BRONZE_OBSERVATION_PERIOD
    assert row.observation_period_bronze_source_key == "obs-301"
    assert row.silver_conformed_at is not None
    assert result_df.columns[-1] == "silver_conformed_at"
    assert "death_bronze_table" in result_df.columns
    assert row.death_bronze_table is None
    assert row.treatment_group == "Metformin"
    assert row.treatment_concept_name == "Metformin Hydrochloride"
    assert row.treatment_source_value == "metformin"


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_silver_identity_resolution_is_deterministic_for_duplicate_diabetes_and_treatment_events(spark_session):
    spark = spark_session

    condition = spark.createDataFrame(
        [
            (102, 7, "2020-01-03", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-102", "2024-01-01 02:00:00"),
            (101, 7, "2020-01-03", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-101", "2024-01-01 01:00:00"),
        ],
        [
            "condition_occurrence_id",
            "person_id",
            "condition_start_date",
            "condition_source_value",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    drug = spark.createDataFrame(
        [
            (1002, 7, "metformin", "2020-01-10", "2020-02-01", bronze.BRONZE_DRUG_EXPOSURE, "drug-1002", "2024-01-02 02:00:00"),
            (1001, 7, "metformin", "2020-01-10", "2020-02-01", bronze.BRONZE_DRUG_EXPOSURE, "drug-1001", "2024-01-02 01:00:00"),
        ],
        [
            "drug_exposure_id",
            "person_id",
            "drug_source_value",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    death = spark.createDataFrame([], "person_id bigint, death_date string, death_datetime string, bronze_source_table string, bronze_source_key string, bronze_ingested_at string")
    observation = spark.createDataFrame(
        [(501, 7, "2019-01-01", "2020-06-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-501", "2024-01-03 01:00:00")],
        [
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    concept = spark.createDataFrame(
        [(111, "metformin", "Metformin Hydrochloride", "RxNorm")],
        ["concept_id", "concept_code", "concept_name", "vocabulary_id"],
    )

    result = dcs.build_diabetic_cohort_summary(condition.alias("c"), drug.alias("d"), death.alias("de"), observation.alias("o"), concept.alias("co")).collect()

    assert len(result) == 1
    row = result[0]
    assert row.person_id == 7
    assert row.diabetes_condition_occurrence_id == 101
    assert row.drug_exposure_id == 1001
    assert row.diabetes_condition_bronze_source_key == "condition-101"
    assert row.drug_exposure_bronze_source_key == "drug-1001"
    assert row.patient_identity_key == result[0].patient_identity_key


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_silver_unknown_treatments_are_excluded_from_cohort(spark_session):
    spark = spark_session

    condition = spark.createDataFrame(
        [(101, 1, "2020-01-01", dcs.DIABETES_CONDITION_CODE, "condition-101", "2024-01-01 01:00:00")],
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
        [(1001, 1, "unmapped drug", "2020-01-05", "2020-02-01", "drug-1001", "2024-01-02 01:00:00")],
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
    death = spark.createDataFrame([], "person_id bigint, death_date string, death_datetime string, bronze_source_table string, bronze_source_key string, bronze_ingested_at string")
    observation = spark.createDataFrame(
        [(301, 1, "2019-01-01", "2020-06-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-301", "2024-01-03 01:00:00")],
        [
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    concept = spark.createDataFrame([], "concept_id bigint, concept_code string, concept_name string, vocabulary_id string")

    result = dcs.build_diabetic_cohort_summary(condition.alias("c"), drug.alias("d"), death.alias("de"), observation.alias("o"), concept.alias("co")).collect()
    assert result == []


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_silver_diabetic_treatment_cohort_respects_observation_bounds(spark_session):
    spark = spark_session

    condition = spark.createDataFrame(
        [
            (101, 1, "2020-01-01", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-101", "2024-01-01 01:00:00"),
            (201, 2, "2020-03-01", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-201", "2024-01-01 02:00:00"),
        ],
        [
            "condition_occurrence_id",
            "person_id",
            "condition_start_date",
            "condition_source_value",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    drug = spark.createDataFrame(
        [
            (1001, 1, "metformin rxnorm", "2020-01-01", "2020-12-31", bronze.BRONZE_DRUG_EXPOSURE, "drug-1001", "2024-01-02 01:00:00"),
            (2001, 2, "glipizide rxnorm", "2020-03-01", "2020-04-01", bronze.BRONZE_DRUG_EXPOSURE, "drug-2001", "2024-01-02 02:00:00"),
        ],
        [
            "drug_exposure_id",
            "person_id",
            "drug_source_value",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    death = spark.createDataFrame(
        [
            (1, None, None, None, None, None),
            (2, "2020-07-01", None, bronze.BRONZE_DEATH, "death-2", "2024-01-04 01:00:00"),
        ],
        ["person_id", "death_date", "death_datetime", "bronze_source_table", "bronze_source_key", "bronze_ingested_at"],
    )
    observation = spark.createDataFrame(
        [
            (301, 1, "2019-01-01", "2020-06-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-301", "2024-01-03 01:00:00"),
            (302, 1, "2019-06-01", "2020-05-15", bronze.BRONZE_OBSERVATION_PERIOD, "obs-302", "2024-01-03 02:00:00"),
            (401, 2, "2020-01-01", "2020-05-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-401", "2024-01-03 03:00:00"),
        ],
        [
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    concept = spark.createDataFrame(
        [
            (111, "metformin rxnorm", "Metformin HCl", "RxNorm"),
            (222, "glipizide rxnorm", "Glipizide", "RxNorm"),
        ],
        ["concept_id", "concept_code", "concept_name", "vocabulary_id"],
    )

    result = dcs.build_diabetic_cohort_summary(condition.alias("c"), drug.alias("d"), death.alias("de"), observation.alias("o"), concept.alias("co")).collect()
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
            (101, 1, "2020-01-15", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-101", "2024-01-01 01:00:00"),
            (102, 1, "2020-01-01", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-102", "2024-01-01 00:00:00"),
            (201, 2, "2020-02-01", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-201", "2024-01-01 02:00:00"),
            (301, 3, "2020-03-01", dcs.DIABETES_CONDITION_CODE, bronze.BRONZE_CONDITION_OCCURRENCE, "condition-301", "2024-01-01 03:00:00"),
        ],
        [
            "condition_occurrence_id",
            "person_id",
            "condition_start_date",
            "condition_source_value",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    drug = spark.createDataFrame(
        [
            (1001, 1, "Metformin 500 MG Oral Tablet", "2020-01-10", "2020-01-31", bronze.BRONZE_DRUG_EXPOSURE, "drug-1001", "2024-01-02 01:00:00"),
            (1002, 1, "GLIPIZIDE ER", "2020-01-20", "2020-02-15", bronze.BRONZE_DRUG_EXPOSURE, "drug-1002", "2024-01-02 02:00:00"),
            (2001, 2, "insulin glargine prefilled pen", "2020-02-01", "2020-02-28", bronze.BRONZE_DRUG_EXPOSURE, "drug-2001", "2024-01-02 03:00:00"),
            (3001, 3, "GLIPIZIDE ER", "2020-03-01", "2020-03-31", bronze.BRONZE_DRUG_EXPOSURE, "drug-3001", "2024-01-02 04:00:00"),
        ],
        [
            "drug_exposure_id",
            "person_id",
            "drug_source_value",
            "drug_exposure_start_date",
            "drug_exposure_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    death = spark.createDataFrame(
        [
            (2, "2020-02-15", None, bronze.BRONZE_DEATH, "death-2", "2024-01-04 01:00:00"),
            (2, "2020-02-14", None, bronze.BRONZE_DEATH, "death-older-2", "2024-01-03 01:00:00"),
        ],
        ["person_id", "death_date", "death_datetime", "bronze_source_table", "bronze_source_key", "bronze_ingested_at"],
    )
    observation = spark.createDataFrame(
        [
            (301, 1, "2019-01-01", "2020-06-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-301", "2024-01-03 01:00:00"),
            (302, 1, "2019-06-01", "2020-06-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-302", "2024-01-03 02:00:00"),
            (401, 2, "2020-01-01", "2020-05-01", bronze.BRONZE_OBSERVATION_PERIOD, "obs-401", "2024-01-03 02:00:00"),
            (501, 3, "2020-01-01", "2020-02-15", bronze.BRONZE_OBSERVATION_PERIOD, "obs-501", "2024-01-03 03:00:00"),
        ],
        [
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            "bronze_source_table",
            "bronze_source_key",
            "bronze_ingested_at",
        ],
    )
    concept = spark.createDataFrame([], "concept_id bigint, concept_code string, concept_name string, vocabulary_id string")

    result = dcs.build_diabetic_cohort_summary(condition.alias("c"), drug.alias("d"), death.alias("de"), observation.alias("o"), concept.alias("co")).collect()

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
    assert rows[9].silver_source_key == f"{dcs.SILVER_DIABETIC_TREATMENT_COHORT}::Metformin"
    assert rows[9].silver_lineage_layer == "silver"
    assert rows[9].gold_analytics_version == "v1"
    assert rows[9].survival_percent == pytest.approx(rows[9].survival_probability * 100.0)


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_gold_survival_curve_keeps_same_day_outcomes_and_excludes_negative_follow_up(spark_session):
    spark = spark_session
    cohort = spark.createDataFrame([
        (1, "identity-1", "Metformin", "2020-01-01", "2020-01-01", 1, "2020-01-01"),
        (2, "identity-2", "Metformin", "2020-01-01", "2020-01-05", 1, "2020-01-05"),
        (3, "identity-3", "Metformin", "2020-01-01", "2020-01-01", 0, None),
        (4, "identity-4", "Metformin", "2020-01-01", "2020-01-10", 0, None),
        (5, "identity-5", "Metformin", "2020-01-02", "2020-01-01", 0, None),
    ], ["person_id", "patient_identity_key", "treatment_group", "treatment_start_date", "observation_end_date", "mortality_status", "date_of_death"])

    out = ss.build_survival_statistics(cohort).collect()
    rows = {r.time_to_event_days: r for r in out}

    assert set(rows) == {0, 4, 9}
    assert rows[0].num_events == 1
    assert rows[0].num_at_risk == 4
    assert rows[4].num_events == 1
    assert rows[4].num_at_risk == 2
    assert rows[9].num_events == 0
    assert rows[9].num_at_risk == 1


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_gold_survival_summary_uses_deterministic_latest_metrics_on_ties(spark_session, monkeypatch):
    spark = spark_session
    cohort = spark.createDataFrame([
        (1, "identity-1", "Metformin", "2020-01-01", "2020-01-10", 0, None),
        (2, "identity-2", "Metformin", "2020-01-01", "2020-01-10", 1, "2020-01-10"),
    ], ["person_id", "patient_identity_key", "treatment_group", "treatment_start_date", "observation_end_date", "mortality_status", "date_of_death"])

    tied_curve = spark.createDataFrame([
        ("Metformin", dcs.SILVER_DIABETIC_TREATMENT_COHORT, "silver_diabetic_treatment_cohort::Metformin", "silver", "v1", 9, 0.4, 40.0, 0.2, 0.6, 0.4, 1, 1),
        ("Metformin", dcs.SILVER_DIABETIC_TREATMENT_COHORT, "silver_diabetic_treatment_cohort::Metformin", "silver", "v1", 9, 0.8, 80.0, 0.7, 0.9, 0.2, 2, 0),
        ("Metformin", dcs.SILVER_DIABETIC_TREATMENT_COHORT, "silver_diabetic_treatment_cohort::Metformin", "silver", "v1", 5, 0.9, 90.0, 0.8, 1.0, 0.2, 2, 0),
    ], [
        "treatment_group",
        "silver_source_table",
        "silver_source_key",
        "silver_lineage_layer",
        "gold_analytics_version",
        "time_to_event_days",
        "survival_probability",
        "survival_percent",
        "lower_ci",
        "upper_ci",
        "confidence_interval_width",
        "num_at_risk",
        "num_events",
    ])

    monkeypatch.setattr(ss, "build_survival_statistics", lambda _: tied_curve)

    row = ss.build_survival_summary(cohort).collect()[0]

    assert row.latest_time_point_days == 9
    assert row.latest_survival_probability == pytest.approx(0.8)
    assert row.latest_survival_percent == pytest.approx(80.0)
    assert row.latest_num_at_risk == 2
    assert row.latest_num_events == 0


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
    assert metformin.silver_source_key == f"{dcs.SILVER_DIABETIC_TREATMENT_COHORT}::Metformin"
    assert metformin.silver_lineage_layer == "silver"
    assert metformin.gold_analytics_version == "v1"
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


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_gold_survival_assets_publish_business_friendly_column_contracts(spark_session):
    spark = spark_session
    cohort = spark.createDataFrame([
        (1, "identity-1", "Metformin", "2020-01-01", "2020-01-10", 1, "2020-01-10"),
        (2, "identity-2", "Metformin", "2020-01-01", "2020-01-20", 0, None),
    ], ["person_id", "patient_identity_key", "treatment_group", "treatment_start_date", "observation_end_date", "mortality_status", "date_of_death"])

    curve_df = ss.build_survival_statistics(cohort)
    summary_df = ss.build_survival_summary(cohort)

    assert curve_df.columns == list(ss.SURVIVAL_CURVE_COLUMNS)
    assert summary_df.columns == list(ss.SUMMARY_COLUMNS)
    assert {"survival_percent", "confidence_interval_width", "num_at_risk", "num_events"}.issubset(curve_df.columns)
    assert {"cohort_size", "event_rate_percent", "median_survival_reached", "latest_survival_percent"}.issubset(summary_df.columns)


def test_gold_asset_names_follow_medallion_conventions():
    assert ss.GOLD_TREATMENT_SURVIVAL_CURVE == "gold_diabetic_treatment_survival_curve"
    assert ss.GOLD_TREATMENT_SURVIVAL_SUMMARY == "gold_diabetic_treatment_survival_summary"
    assert ss.GOLD_TREATMENT_SURVIVAL_CURVE.startswith("gold_")
    assert ss.GOLD_TREATMENT_SURVIVAL_SUMMARY.startswith("gold_")
    assert ss.GOLD_SURVIVAL_LINEAGE_LAYER == "silver"
    assert ss.GOLD_SURVIVAL_ANALYTICS_VERSION == "v1"
