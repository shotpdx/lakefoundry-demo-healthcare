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


@dp.materialized_view(
    name="diabetic_cohort_summary",
    comment="Diabetic cohort treatment summary for survival analysis",
)
def diabetic_cohort_summary() -> DataFrame:
    condition = spark.read.table(_bronze(BRONZE_CONDITION_OCCURRENCE)).alias("c")
    drug = spark.read.table(_bronze(BRONZE_DRUG_EXPOSURE)).alias("d")
    death = spark.read.table(_bronze(BRONZE_DEATH)).alias("de")
    observation = spark.read.table(_bronze(BRONZE_OBSERVATION_PERIOD)).alias("o")
    concept = spark.read.table(_bronze(BRONZE_CONCEPT)).alias("co")

    diabetes_patients = (
        condition.filter(F.col("condition_source_value") == F.lit(DIABETES_CONDITION_CODE))
        .select("person_id")
        .distinct()
    )

    treatments = (
        drug.join(concept, F.col("d.drug_source_value") == F.col("co.concept_code"), "left")
        .withColumn("treatment_group", _treatment_group_expression())
        .filter(F.col("treatment_group") != F.lit("Unknown"))
        .select(
            F.col("d.person_id").alias("person_id"),
            F.col("treatment_group"),
            F.to_date(F.col("d.drug_exposure_start_date")).alias("treatment_start_date"),
            F.to_date(F.col("d.drug_exposure_end_date")).alias("drug_exposure_end_date"),
        )
    )

    first_treatment_window = Window.partitionBy("person_id").orderBy(
        F.col("treatment_start_date").asc_nulls_last(), F.col("treatment_group").asc()
    )

    first_treatment = (
        treatments.withColumn("rn", F.row_number().over(first_treatment_window))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    cohort = (
        diabetes_patients.join(first_treatment, "person_id", "inner")
        .join(
            observation.groupBy("person_id").agg(F.max("observation_period_end_date").alias("observation_period_end_date")),
            "person_id",
            "left",
        )
        .join(death.select("person_id", "death_date"), "person_id", "left")
        .select(
            F.col("person_id").cast("bigint").alias("person_id"),
            F.col("treatment_group"),
            F.col("treatment_start_date").cast("date").alias("treatment_start_date"),
            F.greatest(
                F.col("observation_period_end_date").cast("date"),
                F.col("drug_exposure_end_date").cast("date"),
            ).alias("observation_end_date"),
            F.when(F.col("death_date").isNotNull(), F.lit(1)).otherwise(F.lit(0)).cast("int").alias("mortality_status"),
            F.col("death_date").cast("date").alias("date_of_death"),
        )
    )

    return cohort
