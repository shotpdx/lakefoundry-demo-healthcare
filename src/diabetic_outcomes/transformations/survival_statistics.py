from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


def _km_ci_expr(survival_col: str, variance_col: str, z: float, upper: bool) -> F.Column:
    se = F.sqrt(F.col(variance_col))
    if upper:
        return F.least(F.lit(1.0), F.col(survival_col) + F.lit(z) * se)
    return F.greatest(F.lit(0.0), F.col(survival_col) - F.lit(z) * se)


@dp.materialized_view(
    name="survival_statistics",
    comment="Kaplan-Meier survival statistics by treatment group",
)
def survival_statistics() -> DataFrame:
    cohort = spark.read.table("diabetic_cohort_summary")

    events = (
        cohort.select(
            "treatment_group",
            F.col("person_id").cast("bigint").alias("person_id"),
            F.when(F.col("mortality_status") == 1, F.datediff(F.col("date_of_death"), F.col("treatment_start_date"))).otherwise(
                F.datediff(F.col("observation_end_date"), F.col("treatment_start_date"))
            ).cast("int").alias("time_to_event"),
            F.col("mortality_status").cast("int").alias("event"),
        )
        .filter(F.col("time_to_event").isNotNull())
        .withColumn("time_to_event", F.greatest(F.col("time_to_event"), F.lit(0)))
    )

    time_grid = events.select("treatment_group", F.col("time_to_event").alias("time_point")).distinct()
    at_risk = (
        time_grid.alias("tg")
        .join(events.alias("e"), F.col("tg.treatment_group") == F.col("e.treatment_group"))
        .groupBy(F.col("tg.treatment_group").alias("treatment_group"), F.col("tg.time_point").alias("time_point"))
        .agg(
            F.countDistinct(F.when(F.col("e.time_to_event") >= F.col("tg.time_point"), F.col("e.person_id"))).alias("num_at_risk"),
            F.sum(F.when((F.col("e.time_to_event") == F.col("tg.time_point")) & (F.col("e.event") == 1), F.lit(1)).otherwise(F.lit(0))).alias("num_events"),
        )
    )

    window = Window.partitionBy("treatment_group").orderBy("time_point").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    km = (
        at_risk.withColumn("hazard", F.when(F.col("num_at_risk") > 0, F.col("num_events") / F.col("num_at_risk")).otherwise(F.lit(0.0)))
        .withColumn("survival_probability", F.exp(F.sum(F.log(F.greatest(F.lit(1e-12), F.lit(1.0) - F.col("hazard")))).over(window)))
        .withColumn(
            "greenwood_term",
            F.when(
                (F.col("num_at_risk") > F.col("num_events")) & (F.col("num_at_risk") > 0),
                F.col("num_events") / (F.col("num_at_risk") * (F.col("num_at_risk") - F.col("num_events"))),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("greenwood_var", F.pow(F.col("survival_probability"), 2) * F.sum("greenwood_term").over(window))
        .withColumn("lower_ci", _km_ci_expr("survival_probability", "greenwood_var", 1.96, upper=False))
        .withColumn("upper_ci", _km_ci_expr("survival_probability", "greenwood_var", 1.96, upper=True))
    )

    return km.select(
        "treatment_group",
        F.col("time_point").cast("int").alias("time_point"),
        F.col("survival_probability").cast("double").alias("survival_probability"),
        F.col("lower_ci").cast("double").alias("lower_ci"),
        F.col("upper_ci").cast("double").alias("upper_ci"),
        F.col("num_at_risk").cast("long").alias("num_at_risk"),
        F.col("num_events").cast("long").alias("num_events"),
    )
