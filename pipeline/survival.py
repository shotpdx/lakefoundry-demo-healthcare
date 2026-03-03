"""
Kaplan-Meier survival statistics computation.
Accepts a pandas DataFrame (diabetic_cohort_summary) and returns a
survival_statistics DataFrame ready for storage in Delta.
"""
from __future__ import annotations

import pandas as pd
from lifelines import KaplanMeierFitter


def compute_survival_statistics(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Kaplan-Meier survival statistics per treatment group.

    Parameters
    ----------
    cohort : pd.DataFrame
        Must contain columns:
            treatment_group, treatment_start_date, observation_end_date,
            mortality_status, date_of_death

    Returns
    -------
    pd.DataFrame with columns:
        treatment_group, time_point (int days), survival_probability,
        lower_ci, upper_ci, num_at_risk, num_events
    """
    if cohort.empty:
        return pd.DataFrame(columns=[
            "treatment_group", "time_point", "survival_probability",
            "lower_ci", "upper_ci", "num_at_risk", "num_events"
        ])

    # Compute duration (days from treatment_start_date to observation_end_date)
    cohort = cohort.copy()
    cohort["duration"] = (
        pd.to_datetime(cohort["observation_end_date"])
        - pd.to_datetime(cohort["treatment_start_date"])
    ).dt.days.clip(lower=0)

    results = []
    for group, df in cohort.groupby("treatment_group"):
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=df["duration"],
            event_observed=df["mortality_status"],
            label=group,
        )
        timeline = kmf.survival_function_.index.astype(int)
        sf = kmf.survival_function_[group].values
        ci = kmf.confidence_interval_

        # Compute num_at_risk and num_events at each timeline point
        at_risk = [int((df["duration"] >= t).sum()) for t in timeline]
        events = [int(((df["duration"] <= t) & (df["mortality_status"] == 1)).sum()) for t in timeline]

        group_df = pd.DataFrame({
            "treatment_group": group,
            "time_point": timeline,
            "survival_probability": sf,
            "lower_ci": ci.iloc[:, 0].values,
            "upper_ci": ci.iloc[:, 1].values,
            "num_at_risk": at_risk,
            "num_events": events,
        })
        results.append(group_df)

    return pd.concat(results, ignore_index=True)
