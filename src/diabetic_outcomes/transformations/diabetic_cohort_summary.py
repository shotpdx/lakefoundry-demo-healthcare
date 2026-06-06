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


def _bronze(table_name: str) -> str:
    return table_name


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
        drug.join(concept, F.col("d.drug_source_value") == F.col("co.concept_code"), "inner")
        .withColumn(
            "treatment_group",
            F.when(F.lower(F.col("co.concept_name")).contains("metformin"), F.lit("Metformin"))
            .when(F.lower(F.col("co.concept_name")).contains("insulin glargine"), F.lit("Insulin Glargine"))
            .when(F.lower(F.col("co.concept_name")).contains("glipizide"), F.lit("Glipizide")),
        )
        .filter(F.col("treatment_group").isNotNull())
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
