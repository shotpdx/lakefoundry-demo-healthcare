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
def test_diabetic_cohort_summary_uses_latest_observation_period(spark_session, monkeypatch):
    spark = spark_session

    tables = {
        dcs._source("person"): spark.createDataFrame([(1,), (2,)], ["person_id"]),
        dcs._source("condition_occurrence"): spark.createDataFrame([(1, dcs.DIABETES_CONDITION_CODE)], ["person_id", "condition_source_value"]),
        dcs._source("drug_exposure"): spark.createDataFrame([
            (1, "metformin rxnorm", "2020-01-01", "2020-02-01"),
            (2, "glipizide rxnorm", "2020-03-01", "2020-04-01"),
        ], ["person_id", "drug_source_value", "drug_exposure_start_date", "drug_exposure_end_date"]),
        dcs._source("death"): spark.createDataFrame([(1, None), (2, None)], ["person_id", "death_date"]),
        dcs._source("observation_period"): spark.createDataFrame([
            (1, "2020-06-01"),
            (1, "2020-12-31"),
            (2, "2020-05-01"),
        ], ["person_id", "observation_period_end_date"]),
        dcs._source("concept"): spark.createDataFrame([
            ("metformin rxnorm", "Metformin HCl"),
            ("glipizide rxnorm", "Glipizide"),
        ], ["concept_code", "concept_name"]),
    }

    class Reader:
        def table(self, name):
            return tables[name]

    monkeypatch.setattr(dcs, 'spark', MagicMock(read=Reader()))
    result = dcs.diabetic_cohort_summary().collect()
    assert len(result) == 2
    assert {r.observation_end_date.isoformat() for r in result} == {"2020-12-31", "2020-05-01"}


@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not available")
def test_survival_statistics_counts_events_exactly_at_time_point(spark_session, monkeypatch):
    spark = spark_session
    cohort = spark.createDataFrame([
        (1, "Metformin", "2020-01-01", "2020-01-10", 1, "2020-01-10"),
        (2, "Metformin", "2020-01-01", "2020-01-20", 0, None),
        (3, "Metformin", "2020-01-01", "2020-01-10", 1, "2020-01-10"),
    ], ["person_id", "treatment_group", "treatment_start_date", "observation_end_date", "mortality_status", "date_of_death"])

    class Reader:
        def table(self, name):
            return cohort

    monkeypatch.setattr(ss, 'spark', MagicMock(read=Reader()))
    out = ss.survival_statistics().collect()
    rows = {r.time_point: r for r in out}
    assert rows[9].num_events == 2
    assert rows[19].num_events == 0
