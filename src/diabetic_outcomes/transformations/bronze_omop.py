from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SOURCE_CATALOG = "databricks_observational_medical_outcomes_partnership_omop_common_data_model_cdm"
SOURCE_SCHEMA = "patient_risk_altered_omop"
SOURCE_PREFIX = f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}"

BRONZE_PERSON = "bronze_omop_person"
BRONZE_CONDITION_OCCURRENCE = "bronze_omop_condition_occurrence"
BRONZE_DRUG_EXPOSURE = "bronze_omop_drug_exposure"
BRONZE_DEATH = "bronze_omop_death"
BRONZE_OBSERVATION_PERIOD = "bronze_omop_observation_period"
BRONZE_CONCEPT = "bronze_omop_concept"

BRONZE_TABLE_NAMES = (
    BRONZE_PERSON,
    BRONZE_CONDITION_OCCURRENCE,
    BRONZE_DRUG_EXPOSURE,
    BRONZE_DEATH,
    BRONZE_OBSERVATION_PERIOD,
    BRONZE_CONCEPT,
)


def _source(table_name: str) -> str:
    return f"{SOURCE_PREFIX}.{table_name}"


def _with_bronze_metadata(df: DataFrame, source_table: str, key_columns: tuple[str, ...]) -> DataFrame:
    current_ts = F.current_timestamp()
    source_key = F.concat_ws("||", *[F.coalesce(F.col(column).cast("string"), F.lit("")) for column in key_columns])
    return (
        df.withColumn("bronze_source_table", F.lit(source_table))
        .withColumn("bronze_source_key", source_key)
        .withColumn("bronze_ingested_at", current_ts)
    )


@dp.materialized_view(
    name=BRONZE_PERSON,
    comment="Operational-grain OMOP person records with bronze lineage metadata.",
)
def bronze_omop_person() -> DataFrame:
    person = spark.read.table(_source("person")).select(
        F.col("person_id").cast("bigint").alias("person_id"),
        F.col("gender_concept_id").cast("bigint").alias("gender_concept_id"),
        F.col("year_of_birth").cast("int").alias("year_of_birth"),
        F.col("race_concept_id").cast("bigint").alias("race_concept_id"),
        F.col("ethnicity_concept_id").cast("bigint").alias("ethnicity_concept_id"),
    )
    return _with_bronze_metadata(person, "person", ("person_id",))


@dp.materialized_view(
    name=BRONZE_CONDITION_OCCURRENCE,
    comment="Operational-grain OMOP condition occurrence records with source identifiers.",
)
def bronze_omop_condition_occurrence() -> DataFrame:
    condition = spark.read.table(_source("condition_occurrence")).select(
        F.col("condition_occurrence_id").cast("bigint").alias("condition_occurrence_id"),
        F.col("person_id").cast("bigint").alias("person_id"),
        F.col("condition_concept_id").cast("bigint").alias("condition_concept_id"),
        F.col("condition_start_date").cast("date").alias("condition_start_date"),
        F.col("condition_end_date").cast("date").alias("condition_end_date"),
        F.col("condition_source_value").alias("condition_source_value"),
    )
    return _with_bronze_metadata(condition, "condition_occurrence", ("condition_occurrence_id",))


@dp.materialized_view(
    name=BRONZE_DRUG_EXPOSURE,
    comment="Operational-grain OMOP drug exposure records with source identifiers.",
)
def bronze_omop_drug_exposure() -> DataFrame:
    drug = spark.read.table(_source("drug_exposure")).select(
        F.col("drug_exposure_id").cast("bigint").alias("drug_exposure_id"),
        F.col("person_id").cast("bigint").alias("person_id"),
        F.col("drug_concept_id").cast("bigint").alias("drug_concept_id"),
        F.col("drug_exposure_start_date").cast("date").alias("drug_exposure_start_date"),
        F.col("drug_exposure_end_date").cast("date").alias("drug_exposure_end_date"),
        F.col("drug_source_value").alias("drug_source_value"),
    )
    return _with_bronze_metadata(drug, "drug_exposure", ("drug_exposure_id",))


@dp.materialized_view(
    name=BRONZE_DEATH,
    comment="Operational-grain OMOP death records with source identifiers.",
)
def bronze_omop_death() -> DataFrame:
    death = spark.read.table(_source("death")).select(
        F.col("person_id").cast("bigint").alias("person_id"),
        F.col("death_date").cast("date").alias("death_date"),
        F.col("death_datetime").cast("timestamp").alias("death_datetime"),
    )
    return _with_bronze_metadata(death, "death", ("person_id", "death_date", "death_datetime"))


@dp.materialized_view(
    name=BRONZE_OBSERVATION_PERIOD,
    comment="Operational-grain OMOP observation period records with source identifiers.",
)
def bronze_omop_observation_period() -> DataFrame:
    observation = spark.read.table(_source("observation_period")).select(
        F.col("observation_period_id").cast("bigint").alias("observation_period_id"),
        F.col("person_id").cast("bigint").alias("person_id"),
        F.col("observation_period_start_date").cast("date").alias("observation_period_start_date"),
        F.col("observation_period_end_date").cast("date").alias("observation_period_end_date"),
    )
    return _with_bronze_metadata(observation, "observation_period", ("observation_period_id",))


@dp.materialized_view(
    name=BRONZE_CONCEPT,
    comment="Operational-grain OMOP concept records required for treatment mapping.",
)
def bronze_omop_concept() -> DataFrame:
    concept = spark.read.table(_source("concept")).select(
        F.col("concept_id").cast("bigint").alias("concept_id"),
        F.col("concept_code").alias("concept_code"),
        F.col("concept_name").alias("concept_name"),
        F.col("vocabulary_id").alias("vocabulary_id"),
    )
    return _with_bronze_metadata(concept, "concept", ("concept_id",))
