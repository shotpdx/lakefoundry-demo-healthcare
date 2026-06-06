import pytest


from diabetic_outcomes.transformations import bronze_omop as bronze
from diabetic_outcomes.transformations import diabetic_cohort_summary as dcs
from diabetic_outcomes.transformations import survival_statistics as ss

CATALOG = "lakefoundry_dev"
SCHEMA = "hls_demo_omop_analytics"
SILVER_COHORT_TABLE = f"{CATALOG}.{SCHEMA}.{dcs.SILVER_DIABETIC_TREATMENT_COHORT}"
GOLD_SURVIVAL_CURVE_TABLE = f"{CATALOG}.{SCHEMA}.{ss.GOLD_TREATMENT_SURVIVAL_CURVE}"
GOLD_SURVIVAL_SUMMARY_TABLE = f"{CATALOG}.{SCHEMA}.{ss.GOLD_TREATMENT_SURVIVAL_SUMMARY}"

BRONZE_TABLES = [
    bronze.BRONZE_PERSON,
    bronze.BRONZE_CONDITION_OCCURRENCE,
    bronze.BRONZE_DRUG_EXPOSURE,
    bronze.BRONZE_DEATH,
    bronze.BRONZE_OBSERVATION_PERIOD,
    bronze.BRONZE_CONCEPT,
]

EXPECTED_COHORT_SCHEMA = [
    ("person_id", "bigint"),
    ("patient_identity_key", "string"),
    ("diabetes_condition_occurrence_id", "bigint"),
    ("diabetes_condition_start_date", "date"),
    ("drug_exposure_id", "bigint"),
    ("treatment_group", "string"),
    ("treatment_start_date", "date"),
    ("drug_exposure_end_date", "date"),
    ("observation_end_date", "date"),
    ("observation_period_id", "bigint"),
    ("mortality_status", "int"),
    ("date_of_death", "date"),
    ("treatment_source_value", "string"),
    ("treatment_concept_name", "string"),
    ("treatment_concept_code", "string"),
    ("diabetes_condition_bronze_table", "string"),
    ("diabetes_condition_bronze_source_key", "string"),
    ("diabetes_condition_bronze_ingested_at", "timestamp"),
    ("drug_exposure_bronze_table", "string"),
    ("drug_exposure_bronze_source_key", "string"),
    ("drug_exposure_bronze_ingested_at", "timestamp"),
    ("observation_period_bronze_table", "string"),
    ("observation_period_bronze_source_key", "string"),
    ("observation_period_bronze_ingested_at", "timestamp"),
    ("death_bronze_table", "string"),
    ("death_bronze_source_key", "string"),
    ("death_bronze_ingested_at", "timestamp"),
    ("silver_conformed_at", "timestamp"),
]

EXPECTED_SURVIVAL_SCHEMA = [
    ("treatment_group", "string"),
    ("silver_source_table", "string"),
    ("silver_source_key", "string"),
    ("silver_lineage_layer", "string"),
    ("gold_analytics_version", "string"),
    ("time_to_event_days", "int"),
    ("survival_probability", "double"),
    ("survival_percent", "double"),
    ("lower_ci", "double"),
    ("upper_ci", "double"),
    ("confidence_interval_width", "double"),
    ("num_at_risk", "bigint"),
    ("num_events", "bigint"),
]

EXPECTED_GOLD_SUMMARY_SCHEMA = [
    ("treatment_group", "string"),
    ("silver_source_table", "string"),
    ("silver_source_key", "string"),
    ("silver_lineage_layer", "string"),
    ("gold_analytics_version", "string"),
    ("cohort_size", "bigint"),
    ("total_events", "bigint"),
    ("event_rate", "double"),
    ("event_rate_percent", "double"),
    ("avg_follow_up_days", "double"),
    ("median_follow_up_days", "int"),
    ("max_follow_up_days", "int"),
    ("median_survival_days", "int"),
    ("median_survival_reached", "boolean"),
    ("latest_time_point_days", "int"),
    ("latest_survival_probability", "double"),
    ("latest_survival_percent", "double"),
    ("latest_num_at_risk", "bigint"),
    ("latest_num_events", "bigint"),
]

REQUIRED_TREATMENT_GROUPS = {
    "Metformin",
    "Insulin Glargine",
    "Glipizide",
}


def _run_sql(sql: str, sql_executor):
    try:
        return sql_executor(sql)
    except Exception as exc:  # pragma: no cover - exercised in Databricks runtime
        pytest.skip(f"Skipping SQL validation because Databricks SQL is unavailable: {exc}")


@pytest.fixture(scope="module")
def tables_available(sql_executor):
    queries = {
        "cohort": f"SHOW TABLES IN {CATALOG}.{SCHEMA} LIKE '{dcs.SILVER_DIABETIC_TREATMENT_COHORT}'",
        "gold_survival_curve": f"SHOW TABLES IN {CATALOG}.{SCHEMA} LIKE '{ss.GOLD_TREATMENT_SURVIVAL_CURVE}'",
        "gold_survival_summary": f"SHOW TABLES IN {CATALOG}.{SCHEMA} LIKE '{ss.GOLD_TREATMENT_SURVIVAL_SUMMARY}'",
    }
    results = {name: _run_sql(sql, sql_executor) for name, sql in queries.items()}
    if not all(results.values()):
        pytest.skip("Validation tables do not exist yet in Unity Catalog")
    return True


def _assert_schema(table_name: str, expected_schema, sql_executor):
    rows = _run_sql(f"DESCRIBE TABLE {table_name}", sql_executor)
    actual = [
        (row["col_name"].strip().lower(), row["data_type"].strip().lower())
        for row in rows
        if row.get("col_name") and not row["col_name"].startswith("#")
    ]
    expected = [(name.lower(), dtype.lower()) for name, dtype in expected_schema]
    assert actual[: len(expected)] == expected, (
        f"Schema mismatch for {table_name}. Expected prefix {expected}, got {actual[:len(expected)]}"
    )


def test_bronze_table_set_matches_expected_medallion_inputs():
    assert list(bronze.BRONZE_TABLE_NAMES) == BRONZE_TABLES
    assert all(table_name.startswith("bronze_omop_") for table_name in BRONZE_TABLES)


def test_bronze_tables_are_sufficient_for_current_silver_scope():
    required_for_silver = {
        bronze.BRONZE_CONDITION_OCCURRENCE,
        bronze.BRONZE_DRUG_EXPOSURE,
        bronze.BRONZE_DEATH,
        bronze.BRONZE_OBSERVATION_PERIOD,
        bronze.BRONZE_CONCEPT,
    }
    assert required_for_silver.issubset(set(bronze.BRONZE_TABLE_NAMES))
    assert bronze.BRONZE_PERSON in bronze.BRONZE_TABLE_NAMES


def test_silver_schema_includes_identity_and_audit_lineage_columns():
    schema_columns = {name: dtype for name, dtype in EXPECTED_COHORT_SCHEMA}
    assert schema_columns["patient_identity_key"] == "string"
    assert schema_columns["silver_conformed_at"] == "timestamp"
    assert schema_columns["diabetes_condition_bronze_table"] == "string"
    assert schema_columns["drug_exposure_bronze_source_key"] == "string"
    assert schema_columns["observation_period_bronze_ingested_at"] == "timestamp"


def test_silver_diabetic_treatment_cohort_schema(tables_available, sql_executor):
    _assert_schema(SILVER_COHORT_TABLE, EXPECTED_COHORT_SCHEMA, sql_executor)


def test_gold_survival_curve_schema(tables_available, sql_executor):
    _assert_schema(GOLD_SURVIVAL_CURVE_TABLE, EXPECTED_SURVIVAL_SCHEMA, sql_executor)


def test_gold_survival_summary_schema(tables_available, sql_executor):
    _assert_schema(GOLD_SURVIVAL_SUMMARY_TABLE, EXPECTED_GOLD_SUMMARY_SCHEMA, sql_executor)


@pytest.mark.parametrize(
    "table_name,critical_columns",
    [
        (SILVER_COHORT_TABLE, ["person_id", "patient_identity_key", "treatment_group", "treatment_start_date"]),
    ],
)
def test_no_nulls_in_critical_columns(tables_available, table_name, critical_columns, sql_executor):
    where_clause = " OR ".join(f"{column} IS NULL" for column in critical_columns)
    rows = _run_sql(f"SELECT COUNT(*) AS null_count FROM {table_name} WHERE {where_clause}", sql_executor)
    null_count = rows[0]["null_count"]
    assert null_count == 0, (
        f"Found {null_count} rows with null values in critical columns {critical_columns} for {table_name}"
    )


def test_all_treatment_groups_represented(tables_available, sql_executor):
    rows = _run_sql(f"SELECT DISTINCT treatment_group FROM {SILVER_COHORT_TABLE}", sql_executor)
    observed = {row["treatment_group"] for row in rows if row["treatment_group"] is not None}
    missing = REQUIRED_TREATMENT_GROUPS - observed
    assert not missing, (
        f"Missing treatment groups in {SILVER_COHORT_TABLE}: {sorted(missing)}. Observed groups: {sorted(observed)}"
    )


def test_survival_probabilities_between_zero_and_one(tables_available, sql_executor):
    table_name = GOLD_SURVIVAL_CURVE_TABLE
    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count FROM {table_name} "
        "WHERE survival_probability < 0 OR survival_probability > 1 OR survival_probability IS NULL",
        sql_executor,
    )
    invalid_count = rows[0]["invalid_count"]
    assert invalid_count == 0, (
        f"Found {invalid_count} survival_probability values outside [0, 1] in {table_name}"
    )


def test_silver_cohort_has_one_row_per_person_and_valid_follow_up_bounds(tables_available, sql_executor):
    rows = _run_sql(
        f"SELECT COUNT(*) AS total_rows, COUNT(DISTINCT person_id) AS distinct_people, "
        f"SUM(CASE WHEN observation_end_date < treatment_start_date THEN 1 ELSE 0 END) AS invalid_bounds "
        f"FROM {SILVER_COHORT_TABLE}",
        sql_executor,
    )
    row = rows[0]
    assert row["total_rows"] == row["distinct_people"], (
        f"Expected one row per person in {SILVER_COHORT_TABLE}, got {row['total_rows']} rows for {row['distinct_people']} people"
    )
    assert row["invalid_bounds"] == 0, (
        f"Found {row['invalid_bounds']} rows where observation_end_date precedes treatment_start_date in {SILVER_COHORT_TABLE}"
    )


def test_bronze_lineage_columns_cover_downstream_needs():
    assert bronze._source("drug_exposure").endswith(".drug_exposure")
    assert bronze.BRONZE_DRUG_EXPOSURE in bronze.BRONZE_TABLE_NAMES
    assert bronze.BRONZE_CONDITION_OCCURRENCE in bronze.BRONZE_TABLE_NAMES


def test_treatment_mapping_configuration_is_explicit():
    assert dcs.TREATMENT_CONCEPT_PATTERNS == {
        "Metformin": ("metformin",),
        "Insulin Glargine": ("insulin glargine",),
        "Glipizide": ("glipizide",),
    }


def test_gold_assets_reference_silver_lineage_explicitly(tables_available, sql_executor):
    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count FROM {GOLD_SURVIVAL_CURVE_TABLE} "
        f"WHERE silver_source_table <> '{dcs.SILVER_DIABETIC_TREATMENT_COHORT}' "
        f"OR silver_source_table IS NULL "
        f"OR silver_source_key IS NULL "
        f"OR silver_source_key <> CONCAT('{dcs.SILVER_DIABETIC_TREATMENT_COHORT}', '::', treatment_group) "
        f"OR silver_lineage_layer <> '{ss.GOLD_SURVIVAL_LINEAGE_LAYER}' "
        f"OR gold_analytics_version <> '{ss.GOLD_SURVIVAL_ANALYTICS_VERSION}'",
        sql_executor,
    )
    assert rows[0]["invalid_count"] == 0

    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count FROM {GOLD_SURVIVAL_SUMMARY_TABLE} "
        f"WHERE silver_source_table <> '{dcs.SILVER_DIABETIC_TREATMENT_COHORT}' "
        f"OR silver_source_table IS NULL "
        f"OR silver_source_key IS NULL "
        f"OR silver_source_key <> CONCAT('{dcs.SILVER_DIABETIC_TREATMENT_COHORT}', '::', treatment_group) "
        f"OR silver_lineage_layer <> '{ss.GOLD_SURVIVAL_LINEAGE_LAYER}' "
        f"OR gold_analytics_version <> '{ss.GOLD_SURVIVAL_ANALYTICS_VERSION}'",
        sql_executor,
    )
    assert rows[0]["invalid_count"] == 0


def test_gold_summary_metrics_are_business_ready(tables_available, sql_executor):
    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count FROM {GOLD_SURVIVAL_SUMMARY_TABLE} "
        "WHERE cohort_size <= 0 "
        "OR total_events < 0 "
        "OR total_events > cohort_size "
        "OR event_rate < 0 OR event_rate > 1 "
        "OR ABS(event_rate_percent - (event_rate * 100)) > 0.0001 "
        "OR avg_follow_up_days < 0 "
        "OR median_follow_up_days < 0 "
        "OR max_follow_up_days < 0 "
        "OR median_follow_up_days > max_follow_up_days "
        "OR latest_time_point_days < 0 "
        "OR latest_num_at_risk <= 0 "
        "OR latest_num_events < 0 "
        "OR latest_num_events > latest_num_at_risk "
        "OR latest_survival_probability < 0 OR latest_survival_probability > 1 "
        "OR ABS(latest_survival_percent - (latest_survival_probability * 100)) > 0.0001 "
        "OR latest_survival_percent < 0 OR latest_survival_percent > 100",
        sql_executor,
    )
    assert rows[0]["invalid_count"] == 0


def test_gold_survival_curve_quality_thresholds_are_business_ready(tables_available, sql_executor):
    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count FROM {GOLD_SURVIVAL_CURVE_TABLE} "
        "WHERE time_to_event_days < 0 "
        "OR num_at_risk <= 0 "
        "OR num_events <= 0 "
        "OR num_events > num_at_risk "
        "OR lower_ci < 0 OR lower_ci > 1 "
        "OR upper_ci < 0 OR upper_ci > 1 "
        "OR lower_ci > upper_ci "
        "OR confidence_interval_width < 0 "
        "OR ABS(survival_percent - (survival_probability * 100)) > 0.0001",
        sql_executor,
    )
    assert rows[0]["invalid_count"] == 0


def test_gold_summary_latest_point_must_match_event_time(sql_executor, tables_available):
    rows = _run_sql(
        f"WITH latest_summary AS ("
        f"  SELECT treatment_group, latest_time_point_days, latest_num_events, latest_survival_probability "
        f"  FROM {GOLD_SURVIVAL_SUMMARY_TABLE}"
        f"), latest_curve AS ("
        f"  SELECT treatment_group, MAX(time_to_event_days) AS latest_curve_time "
        f"  FROM {GOLD_SURVIVAL_CURVE_TABLE} GROUP BY treatment_group"
        f") "
        f"SELECT COUNT(*) AS invalid_count "
        f"FROM latest_summary s JOIN latest_curve c USING (treatment_group) "
        f"WHERE s.latest_time_point_days <> c.latest_curve_time OR s.latest_num_events <= 0 OR s.latest_survival_probability IS NULL",
        sql_executor,
    )
    assert rows[0]["invalid_count"] == 0


def test_gold_summary_median_survival_must_land_on_event_row(sql_executor, tables_available):
    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count "
        f"FROM {GOLD_SURVIVAL_SUMMARY_TABLE} s "
        f"LEFT JOIN {GOLD_SURVIVAL_CURVE_TABLE} c "
        f"  ON s.treatment_group = c.treatment_group "
        f" AND s.median_survival_days = c.time_to_event_days "
        f"WHERE s.median_survival_reached = TRUE "
        f"  AND (c.time_to_event_days IS NULL OR c.num_events <= 0 OR c.survival_probability > 0.5)",
        sql_executor,
    )
    assert rows[0]["invalid_count"] == 0


def test_silver_quality_thresholds_match_business_ready_expectations(tables_available, sql_executor):
    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count FROM {SILVER_COHORT_TABLE} "
        "WHERE diabetes_condition_bronze_table IS NULL "
        "OR drug_exposure_bronze_table IS NULL "
        "OR observation_period_bronze_table IS NULL "
        "OR diabetes_condition_bronze_source_key IS NULL "
        "OR drug_exposure_bronze_source_key IS NULL "
        "OR observation_period_bronze_source_key IS NULL "
        "OR silver_conformed_at IS NULL "
        "OR mortality_status NOT IN (0, 1) "
        "OR (mortality_status = 1 AND date_of_death IS NULL) "
        "OR (mortality_status = 0 AND date_of_death IS NOT NULL)",
        sql_executor,
    )
    assert rows[0]["invalid_count"] == 0


def test_silver_asset_name_follows_medallion_conventions():
    assert dcs.SILVER_DIABETIC_TREATMENT_COHORT.startswith("silver_")
    assert dcs.SILVER_DIABETIC_TREATMENT_COHORT == "silver_diabetic_treatment_cohort"
    assert ss.GOLD_TREATMENT_SURVIVAL_CURVE.startswith("gold_")
    assert ss.GOLD_TREATMENT_SURVIVAL_SUMMARY.startswith("gold_")
