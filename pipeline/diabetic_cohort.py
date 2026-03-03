"""
Pure-Python cohort logic helpers.
These functions accept pandas DataFrames so they can be unit-tested without Spark.
The DLT notebook calls these helpers after converting Spark DataFrames to pandas.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

TREATMENT_DRUGS = ["Metformin", "Insulin glargine", "Glipizide"]

def identify_diabetic_patients(conditions: pd.DataFrame) -> pd.DataFrame:
    """Return rows from *conditions* where condition_source_value == 'E11.9'."""
    return conditions[conditions["condition_source_value"] == "E11.9"].copy()


def assign_treatment_group(
    drugs: pd.DataFrame,
    obs_periods: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each patient, find the first exposure to a defined treatment drug
    that starts within their observation period.

    Returns a DataFrame with columns:
        person_id, treatment_group, treatment_start_date, drug_exposure_end_date
    """
    # Keep only defined treatment drugs
    filtered = drugs[drugs["drug_source_value"].isin(TREATMENT_DRUGS)].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=["person_id", "treatment_group", "treatment_start_date", "drug_exposure_end_date"]
        )

    # Join with observation periods
    merged = filtered.merge(obs_periods, on="person_id", how="inner")

    # Keep only exposures that start within the observation period
    merged = merged[
        (merged["drug_exposure_start_date"] >= merged["observation_period_start_date"])
        & (merged["drug_exposure_start_date"] <= merged["observation_period_end_date"])
    ]
    if merged.empty:
        return pd.DataFrame(
            columns=["person_id", "treatment_group", "treatment_start_date", "drug_exposure_end_date"]
        )

    # Sort by start date and keep the first treatment per patient
    merged = merged.sort_values("drug_exposure_start_date")
    first = merged.groupby("person_id", as_index=False).first()

    return first[["person_id", "drug_source_value", "drug_exposure_start_date", "drug_exposure_end_date"]].rename(
        columns={
            "drug_source_value": "treatment_group",
            "drug_exposure_start_date": "treatment_start_date",
        }
    )


def compute_observation_end(
    treatment_start: date,
    drug_exposure_end: date,
    obs_period_end: date,
) -> date:
    """Return the later of obs_period_end and drug_exposure_end."""
    return max(obs_period_end, drug_exposure_end)


def compute_mortality_status(
    observation_end: date,
    death_date: Optional[date],
) -> int:
    """Return 1 if death occurred on or before observation_end, else 0."""
    if death_date is None:
        return 0
    return 1 if death_date <= observation_end else 0
