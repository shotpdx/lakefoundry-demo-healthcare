from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from diabetic_outcomes.transformations.bronze_omop import (
    BRONZE_CONDITION_OCCURRENCE,
    BRONZE_CONCEPT,
    BRONZE_DEATH,
    BRONZE_DRUG_EXPOSURE,
    BRONZE_OBSERVATION_PERIOD,
)

SILVER_DIABETIC_TREATMENT_COHORT = "silver_diabetic_treatment_cohort"
DIABETES_CONDITION_CODE = "44054006"
TREATMENT_CONCEPT_PATTERNS = {
    "Metformin": ("metformin",),
    "Insulin Glargine": ("insulin glargine",),
    "Glipizide": ("glipizide",),
}


def _bronze(table_name: str) -> str:
    return table_name


def _treatment_group_expression() -> F.Column:
    concept_name = F.lower(F.coalesce(F.col("co.concept_name"), F.lit("")))
    concept_code = F.lower(F.coalesce(F.col("co.concept_code"), F.lit("")))
    drug_source_value = F.lower(F.coalesce(F.col("d.drug_source_value"), F.lit("")))

    treatment_group = F.lit("Unknown")
    for label, patterns in reversed(tuple(TREATMENT_CONCEPT_PATTERNS.items())):
        pattern_match = F.lit(False)
        for pattern in patterns:
            pattern_match = pattern_match | concept_name.contains(pattern) | concept_code.contains(pattern) | drug_source_value.contains(pattern)
        treatment_group = F.when(pattern_match, F.lit(label)).otherwise(treatment_group)
    return treatment_group


def _diabetes_patients(condition: DataFrame) -> DataFrame:
    return (
        condition.filter(F.col("condition_source_value") == F.lit(DIABETES_CONDITION_CODE))
        .select(
            F.col("person_id").cast("bigint").alias("person_id"),
            F.col("condition_occurrence_id").cast("bigint").alias("diabetes_condition_occurrence_id"),
            F.col("condition_start_date").cast("date").alias("diabetes_condition_start_date"),
            F.col("bronze_source_key").alias("diabetes_condition_bronze_source_key"),
            F.col("bronze_ingested_at").alias("diabetes_condition_bronze_ingested_at"),
        )
        .withColumn(
            "diabetes_identity_rank",
            F.row_number().over(
                Window.partitionBy("person_id").orderBy(
                    F.col("diabetes_condition_start_date").asc_nulls_last(),
                    F.col("diabetes_condition_occurrence_id").asc_nulls_last(),
                )
            ),
        )
        .filter(F.col("diabetes_identity_rank") == 1)
        .drop("diabetes_identity_rank")
    )


def _conformed_treatments(drug: DataFrame, concept: DataFrame) -> DataFrame:
    return (
        drug.join(concept, F.col("d.drug_source_value") == F.col("co.concept_code"), "left")
        .withColumn("treatment_group", _treatment_group_expression())
        .filter(F.col("treatment_group") != F.lit("Unknown"))
        .select(
            F.col("d.person_id").cast("bigint").alias("person_id"),
            F.col("d.drug_exposure_id").cast("bigint").alias("drug_exposure_id"),
            F.col("treatment_group"),
            F.to_date(F.col("d.drug_exposure_start_date")).alias("treatment_start_date"),
            F.to_date(F.col("d.drug_exposure_end_date")).alias("drug_exposure_end_date"),
            F.col("d.drug_source_value").alias("treatment_source_value"),
            F.col("co.concept_name").alias("treatment_concept_name"),
            F.col("co.concept_code").alias("treatment_concept_code"),
            F.col("d.bronze_source_key").alias("drug_exposure_bronze_source_key"),
            F.col("d.bronze_ingested_at").alias("drug_exposure_bronze_ingested_at"),
        )
    )


@dp.materialized_view(
    name=SILVER_DIABETIC_TREATMENT_COHORT,
    comment="Silver conformed diabetic treatment cohort with identity resolution and bronze lineage.",
)
def diabetic_cohort_summary() -> DataFrame:
    condition = spark.read.table(_bronze(BRONZE_CONDITION_OCCURRENCE)).alias("c")
    drug = spark.read.table(_bronze(BRONZE_DRUG_EXPOSURE)).alias("d")
    death = spark.read.table(_bronze(BRONZE_DEATH)).alias("de")
    observation = spark.read.table(_bronze(BRONZE_OBSERVATION_PERIOD)).alias("o")
    concept = spark.read.table(_bronze(BRONZE_CONCEPT)).alias("co")

    diabetes_patients = _diabetes_patients(condition)
    treatments = _conformed_treatments(drug, concept)

    first_treatment_window = Window.partitionBy("person_id").orderBy(
        F.col("treatment_start_date").asc_nulls_last(),
        F.col("drug_exposure_id").asc_nulls_last(),
        F.col("treatment_group").asc(),
    )

    first_treatment = (
        treatments.withColumn("silver_treatment_rank", F.row_number().over(first_treatment_window))
        .filter(F.col("silver_treatment_rank") == 1)
        .drop("silver_treatment_rank")
    )

    observation_summary = observation.groupBy("person_id").agg(
        F.max("observation_period_end_date").alias("observation_period_end_date"),
        F.max("observation_period_id").alias("observation_period_id"),
        F.max("bronze_source_key").alias("observation_period_bronze_source_key"),
        F.max("bronze_ingested_at").alias("observation_period_bronze_ingested_at"),
    )

    death_summary = death.groupBy("person_id").agg(
        F.max("death_date").alias("death_date"),
        F.max("bronze_source_key").alias("death_bronze_source_key"),
        F.max("bronze_ingested_at").alias("death_bronze_ingested_at"),
    )

    cohort = (
        diabetes_patients.join(first_treatment, "person_id", "inner")
        .join(observation_summary, "person_id", "left")
        .join(death_summary, "person_id", "left")
        .select(
            F.col("person_id").cast("bigint").alias("person_id"),
            F.sha2(F.concat_ws("||", F.lit("person"), F.col("person_id").cast("string")), 256).alias("patient_identity_key"),
            F.col("diabetes_condition_occurrence_id").cast("bigint").alias("diabetes_condition_occurrence_id"),
            F.col("diabetes_condition_start_date").cast("date").alias("diabetes_condition_start_date"),
            F.col("drug_exposure_id").cast("bigint").alias("drug_exposure_id"),
            F.col("treatment_group"),
            F.col("treatment_start_date").cast("date").alias("treatment_start_date"),
            F.col("drug_exposure_end_date").cast("date").alias("drug_exposure_end_date"),
            F.greatest(
                F.col("observation_period_end_date").cast("date"),
                F.col("drug_exposure_end_date").cast("date"),
            ).alias("observation_end_date"),
            F.col("observation_period_id").cast("bigint").alias("observation_period_id"),
            F.when(F.col("death_date").isNotNull(), F.lit(1)).otherwise(F.lit(0)).cast("int").alias("mortality_status"),
            F.col("death_date").cast("date").alias("date_of_death"),
            F.col("treatment_source_value"),
            F.col("treatment_concept_name"),
            F.col("treatment_concept_code"),
            F.lit(BRONZE_CONDITION_OCCURRENCE).alias("diabetes_condition_bronze_table"),
            F.col("diabetes_condition_bronze_source_key"),
            F.col("diabetes_condition_bronze_ingested_at"),
            F.lit(BRONZE_DRUG_EXPOSURE).alias("drug_exposure_bronze_table"),
            F.col("drug_exposure_bronze_source_key"),
            F.col("drug_exposure_bronze_ingested_at"),
            F.lit(BRONZE_OBSERVATION_PERIOD).alias("observation_period_bronze_table"),
            F.col("observation_period_bronze_source_key"),
            F.col("observation_period_bronze_ingested_at"),
            F.when(F.col("death_bronze_source_key").isNotNull(), F.lit(BRONZE_DEATH)).otherwise(F.lit(None).cast("string")).alias("death_bronze_table"),
            F.col("death_bronze_source_key"),
            F.col("death_bronze_ingested_at"),
            F.current_timestamp().alias("silver_conformed_at"),
        )
    )

    return cohort


def silver_diabetic_treatment_cohort() -> DataFrame:
    return diabetic_cohort_summary()
