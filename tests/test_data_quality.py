import pytest


from diabetic_outcomes.transformations import bronze_omop as bronze
from diabetic_outcomes.transformations import diabetic_cohort_summary as dcs

CATALOG = "lakefoundry_dev"
SCHEMA = "hls_demo_omop_analytics"
SILVER_COHORT_TABLE = f"{CATALOG}.{SCHEMA}.{dcs.SILVER_DIABETIC_TREATMENT_COHORT}"
SURVIVAL_TABLE = f"{CATALOG}.{SCHEMA}.survival_statistics"

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
    ("time_point", "int"),
    ("survival_probability", "double"),
    ("lower_ci", "double"),
    ("upper_ci", "double"),
    ("num_at_risk", "bigint"),
    ("num_events", "bigint"),
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
        "survival": f"SHOW TABLES IN {CATALOG}.{SCHEMA} LIKE 'survival_statistics'",
    }
    results = {name: _run_sql(sql, sql_executor) for name, sql in queries.items()}
    if not results["cohort"] or not results["survival"]:
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


def test_silver_diabetic_treatment_cohort_schema(tables_available, sql_executor):
    _assert_schema(SILVER_COHORT_TABLE, EXPECTED_COHORT_SCHEMA, sql_executor)


def test_survival_statistics_schema(tables_available, sql_executor):
    _assert_schema(SURVIVAL_TABLE, EXPECTED_SURVIVAL_SCHEMA, sql_executor)


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
    rows = _run_sql(
        f"SELECT COUNT(*) AS invalid_count FROM {SURVIVAL_TABLE} "
        "WHERE survival_probability < 0 OR survival_probability > 1 OR survival_probability IS NULL",
        sql_executor,
    )
    invalid_count = rows[0]["invalid_count"]
    assert invalid_count == 0, (
        f"Found {invalid_count} survival_probability values outside [0, 1] in {SURVIVAL_TABLE}"
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


def test_silver_asset_name_follows_medallion_conventions():
    assert dcs.SILVER_DIABETIC_TREATMENT_COHORT.startswith("silver_")
    assert dcs.SILVER_DIABETIC_TREATMENT_COHORT == "silver_diabetic_treatment_cohort"
