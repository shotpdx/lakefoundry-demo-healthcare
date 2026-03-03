"""Unit tests for Kaplan-Meier survival statistics computation."""
import pytest
from datetime import date
import pandas as pd
from pipeline.survival import compute_survival_statistics


def _make_cohort(records):
    """Helper: records = list of (treatment_group, treatment_start_date, observation_end_date, mortality_status, date_of_death)"""
    return pd.DataFrame(records, columns=[
        "treatment_group", "treatment_start_date", "observation_end_date",
        "mortality_status", "date_of_death"
    ])


def test_compute_survival_statistics_returns_expected_columns():
    cohort = _make_cohort([
        ("Metformin", date(2020, 1, 1), date(2022, 1, 1), 0, None),
        ("Metformin", date(2020, 1, 1), date(2021, 6, 1), 1, date(2021, 6, 1)),
        ("Glipizide", date(2020, 3, 1), date(2022, 3, 1), 0, None),
    ])
    result = compute_survival_statistics(cohort)
    expected_cols = {
        "treatment_group", "time_point", "survival_probability",
        "lower_ci", "upper_ci", "num_at_risk", "num_events"
    }
    assert expected_cols.issubset(set(result.columns))


def test_compute_survival_statistics_has_row_per_treatment_timepoint():
    cohort = _make_cohort([
        ("Metformin", date(2020, 1, 1), date(2022, 1, 1), 0, None),
        ("Metformin", date(2020, 1, 1), date(2021, 6, 1), 1, date(2021, 6, 1)),
    ])
    result = compute_survival_statistics(cohort)
    # All rows should be for Metformin
    assert (result["treatment_group"] == "Metformin").all()
    # time_point should be non-negative integers (days from treatment start)
    assert (result["time_point"] >= 0).all()


def test_compute_survival_statistics_survival_probability_between_0_and_1():
    cohort = _make_cohort([
        ("Insulin glargine", date(2020, 1, 1), date(2022, 1, 1), 0, None),
        ("Insulin glargine", date(2020, 2, 1), date(2021, 2, 1), 1, date(2021, 2, 1)),
        ("Insulin glargine", date(2020, 3, 1), date(2022, 3, 1), 0, None),
    ])
    result = compute_survival_statistics(cohort)
    assert (result["survival_probability"] >= 0).all()
    assert (result["survival_probability"] <= 1).all()


def test_compute_survival_statistics_multiple_groups():
    cohort = _make_cohort([
        ("Metformin", date(2020, 1, 1), date(2022, 1, 1), 0, None),
        ("Metformin", date(2020, 1, 1), date(2021, 6, 1), 1, date(2021, 6, 1)),
        ("Glipizide", date(2020, 3, 1), date(2022, 3, 1), 0, None),
        ("Glipizide", date(2020, 3, 1), date(2021, 9, 1), 1, date(2021, 9, 1)),
    ])
    result = compute_survival_statistics(cohort)
    groups = result["treatment_group"].unique()
    assert "Metformin" in groups
    assert "Glipizide" in groups


def test_compute_survival_statistics_empty_cohort_returns_empty():
    cohort = _make_cohort([])
    result = compute_survival_statistics(cohort)
    assert result.empty
