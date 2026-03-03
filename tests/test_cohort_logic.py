"""Unit tests for diabetic cohort logic (pure Python / pandas, no Spark)."""
import pytest
from datetime import date
import pandas as pd
from pipeline.diabetic_cohort import (
    identify_diabetic_patients,
    assign_treatment_group,
    compute_observation_end,
    compute_mortality_status,
)


# ── identify_diabetic_patients ──────────────────────────────────────────────

def test_identify_diabetic_patients_returns_e119():
    conditions = pd.DataFrame({
        "person_id": [1, 2, 3],
        "condition_source_value": ["E11.9", "J45.0", "E11.9"],
        "condition_start_date": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
    })
    result = identify_diabetic_patients(conditions)
    assert set(result["person_id"].tolist()) == {1, 3}


def test_identify_diabetic_patients_excludes_non_diabetic():
    conditions = pd.DataFrame({
        "person_id": [1],
        "condition_source_value": ["I10"],
        "condition_start_date": [date(2020, 1, 1)],
    })
    result = identify_diabetic_patients(conditions)
    assert result.empty


# ── assign_treatment_group ──────────────────────────────────────────────────

def test_assign_treatment_group_picks_first_exposure():
    """Patient exposed to Metformin first, then Insulin glargine — should be Metformin."""
    drugs = pd.DataFrame({
        "person_id": [1, 1],
        "drug_source_value": ["Insulin glargine", "Metformin"],
        "drug_exposure_start_date": [date(2020, 6, 1), date(2020, 1, 1)],
        "drug_exposure_end_date": [date(2020, 12, 31), date(2020, 12, 31)],
    })
    obs = pd.DataFrame({
        "person_id": [1],
        "observation_period_start_date": [date(2019, 1, 1)],
        "observation_period_end_date": [date(2021, 12, 31)],
    })
    result = assign_treatment_group(drugs, obs)
    assert result.loc[result["person_id"] == 1, "treatment_group"].iloc[0] == "Metformin"
    assert result.loc[result["person_id"] == 1, "treatment_start_date"].iloc[0] == date(2020, 1, 1)


def test_assign_treatment_group_excludes_unknown_drugs():
    drugs = pd.DataFrame({
        "person_id": [1],
        "drug_source_value": ["Aspirin"],
        "drug_exposure_start_date": [date(2020, 1, 1)],
        "drug_exposure_end_date": [date(2020, 6, 1)],
    })
    obs = pd.DataFrame({
        "person_id": [1],
        "observation_period_start_date": [date(2019, 1, 1)],
        "observation_period_end_date": [date(2021, 12, 31)],
    })
    result = assign_treatment_group(drugs, obs)
    assert result.empty


def test_assign_treatment_group_only_within_observation():
    """Drug exposure starting before observation period should be excluded."""
    drugs = pd.DataFrame({
        "person_id": [1],
        "drug_source_value": ["Glipizide"],
        "drug_exposure_start_date": [date(2018, 1, 1)],  # before obs start
        "drug_exposure_end_date": [date(2018, 6, 1)],
    })
    obs = pd.DataFrame({
        "person_id": [1],
        "observation_period_start_date": [date(2019, 1, 1)],
        "observation_period_end_date": [date(2021, 12, 31)],
    })
    result = assign_treatment_group(drugs, obs)
    assert result.empty


# ── compute_observation_end ─────────────────────────────────────────────────

def test_compute_observation_end_uses_obs_end_when_later():
    result = compute_observation_end(
        treatment_start=date(2020, 1, 1),
        drug_exposure_end=date(2020, 6, 1),
        obs_period_end=date(2021, 12, 31),
    )
    assert result == date(2021, 12, 31)


def test_compute_observation_end_uses_drug_end_when_later():
    result = compute_observation_end(
        treatment_start=date(2020, 1, 1),
        drug_exposure_end=date(2022, 6, 1),
        obs_period_end=date(2021, 12, 31),
    )
    assert result == date(2022, 6, 1)


# ── compute_mortality_status ────────────────────────────────────────────────

def test_compute_mortality_status_death_within_window():
    result = compute_mortality_status(
        observation_end=date(2021, 12, 31),
        death_date=date(2021, 6, 1),
    )
    assert result == 1


def test_compute_mortality_status_death_outside_window():
    result = compute_mortality_status(
        observation_end=date(2021, 12, 31),
        death_date=date(2023, 1, 1),
    )
    assert result == 0


def test_compute_mortality_status_no_death():
    result = compute_mortality_status(
        observation_end=date(2021, 12, 31),
        death_date=None,
    )
    assert result == 0
