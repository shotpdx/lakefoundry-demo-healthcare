from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window

from diabetic_outcomes.transformations.diabetic_cohort_summary import SILVER_DIABETIC_TREATMENT_COHORT

GOLD_TREATMENT_SURVIVAL_CURVE = "gold_diabetic_treatment_survival_curve"
GOLD_TREATMENT_SURVIVAL_SUMMARY = "gold_diabetic_treatment_survival_summary"


def _km_ci_expr(survival_col: str, variance_col: str, z: float, upper: bool) -> F.Column:
    se = F.sqrt(F.col(variance_col))
    if upper:
        return F.least(F.lit(1.0), F.col(survival_col) + F.lit(z) * se)
    return F.greatest(F.lit(0.0), F.col(survival_col) - F.lit(z) * se)


def _follow_up_end_date() -> F.Column:
    death_date = F.col("date_of_death").cast("date")
    observation_end_date = F.col("observation_end_date").cast("date")

    return F.when(F.col("mortality_status") == 1, death_date).otherwise(observation_end_date)


def _valid_follow_up_time() -> F.Column:
    return F.datediff(_follow_up_end_date(), F.col("treatment_start_date").cast("date"))


def _gold_survival_events(cohort: DataFrame) -> DataFrame:
    return (
        cohort.select(
            F.col("person_id").cast("bigint").alias("person_id"),
            F.col("treatment_group").alias("treatment_group"),
            _valid_follow_up_time().cast("int").alias("time_to_event_days"),
            F.col("mortality_status").cast("int").alias("event_indicator"),
        )
        .filter(F.col("time_to_event_days").isNotNull())
        .filter(F.col("time_to_event_days") > 0)
    )


def build_survival_statistics(cohort: DataFrame) -> DataFrame:
    events = _gold_survival_events(cohort)

    time_grid = events.select("treatment_group", F.col("time_to_event_days").alias("time_point")).distinct()
    at_risk = (
        time_grid.alias("tg")
        .join(events.alias("e"), F.col("tg.treatment_group") == F.col("e.treatment_group"))
        .groupBy(F.col("tg.treatment_group").alias("treatment_group"), F.col("tg.time_point").alias("time_point"))
        .agg(
            F.countDistinct(F.when(F.col("e.time_to_event_days") >= F.col("tg.time_point"), F.col("e.person_id"))).alias("num_at_risk"),
            F.sum(F.when((F.col("e.time_to_event_days") == F.col("tg.time_point")) & (F.col("e.event_indicator") == 1), F.lit(1)).otherwise(F.lit(0))).alias("num_events"),
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
        .withColumn("survival_percent", (F.col("survival_probability") * F.lit(100.0)).cast("double"))
        .withColumn("lower_ci", _km_ci_expr("survival_probability", "greenwood_var", 1.96, upper=False))
        .withColumn("upper_ci", _km_ci_expr("survival_probability", "greenwood_var", 1.96, upper=True))
        .withColumn("confidence_interval_width", (F.col("upper_ci") - F.col("lower_ci")).cast("double"))
    )

    return km.select(
        "treatment_group",
        F.lit(SILVER_DIABETIC_TREATMENT_COHORT).alias("silver_source_table"),
        F.col("time_point").cast("int").alias("time_to_event_days"),
        F.col("survival_probability").cast("double").alias("survival_probability"),
        F.col("survival_percent").cast("double").alias("survival_percent"),
        F.col("lower_ci").cast("double").alias("lower_ci"),
        F.col("upper_ci").cast("double").alias("upper_ci"),
        F.col("confidence_interval_width").cast("double").alias("confidence_interval_width"),
        F.col("num_at_risk").cast("long").alias("num_at_risk"),
        F.col("num_events").cast("long").alias("num_events"),
    )


def build_survival_summary(cohort: DataFrame) -> DataFrame:
    events = _gold_survival_events(cohort)
    curve = build_survival_statistics(cohort).alias("curve")

    group_window = Window.partitionBy("treatment_group")
    median_window = Window.partitionBy("treatment_group").orderBy(F.col("time_to_event_days").asc())

    median_candidates = (
        curve.filter(F.col("survival_probability") <= F.lit(0.5))
        .withColumn("median_rank", F.row_number().over(median_window))
        .filter(F.col("median_rank") == 1)
        .select(
            "treatment_group",
            F.col("time_to_event_days").cast("int").alias("median_survival_days"),
        )
    )

    return (
        events.groupBy("treatment_group")
        .agg(
            F.countDistinct("person_id").cast("long").alias("cohort_size"),
            F.sum("event_indicator").cast("long").alias("total_events"),
            F.avg(F.col("time_to_event_days").cast("double")).alias("avg_follow_up_days"),
            F.expr("percentile_approx(time_to_event_days, 0.5)").cast("int").alias("median_follow_up_days"),
            F.max("time_to_event_days").cast("int").alias("max_follow_up_days"),
        )
        .join(
            curve.groupBy("treatment_group")
            .agg(
                F.max("time_to_event_days").cast("int").alias("latest_time_point_days"),
                F.max_by(F.col("survival_probability"), F.col("time_to_event_days")).cast("double").alias("latest_survival_probability"),
                F.max_by(F.col("survival_percent"), F.col("time_to_event_days")).cast("double").alias("latest_survival_percent"),
                F.max_by(F.col("num_at_risk"), F.col("time_to_event_days")).cast("long").alias("latest_num_at_risk"),
                F.max_by(F.col("num_events"), F.col("time_to_event_days")).cast("long").alias("latest_num_events"),
            ),
            "treatment_group",
            "inner",
        )
        .join(median_candidates, "treatment_group", "left")
        .withColumn("event_rate", F.when(F.col("cohort_size") > 0, F.col("total_events") / F.col("cohort_size")).otherwise(F.lit(0.0)))
        .withColumn("event_rate_percent", (F.col("event_rate") * F.lit(100.0)).cast("double"))
        .withColumn("median_survival_reached", F.col("median_survival_days").isNotNull())
        .select(
            "treatment_group",
            F.lit(SILVER_DIABETIC_TREATMENT_COHORT).alias("silver_source_table"),
            F.col("cohort_size").cast("long").alias("cohort_size"),
            F.col("total_events").cast("long").alias("total_events"),
            F.col("event_rate").cast("double").alias("event_rate"),
            F.col("event_rate_percent").cast("double").alias("event_rate_percent"),
            F.col("avg_follow_up_days").cast("double").alias("avg_follow_up_days"),
            F.col("median_follow_up_days").cast("int").alias("median_follow_up_days"),
            F.col("max_follow_up_days").cast("int").alias("max_follow_up_days"),
            F.col("median_survival_days").cast("int").alias("median_survival_days"),
            F.col("median_survival_reached").cast("boolean").alias("median_survival_reached"),
            F.col("latest_time_point_days").cast("int").alias("latest_time_point_days"),
            F.col("latest_survival_probability").cast("double").alias("latest_survival_probability"),
            F.col("latest_survival_percent").cast("double").alias("latest_survival_percent"),
            F.col("latest_num_at_risk").cast("long").alias("latest_num_at_risk"),
            F.col("latest_num_events").cast("long").alias("latest_num_events"),
        )
    )


@dp.materialized_view(
    name=GOLD_TREATMENT_SURVIVAL_CURVE,
    comment="Gold Kaplan-Meier survival curve by diabetic treatment group sourced from the silver cohort.",
)
def gold_diabetic_treatment_survival_curve() -> DataFrame:
    cohort = spark.read.table(SILVER_DIABETIC_TREATMENT_COHORT)
    return build_survival_statistics(cohort)


@dp.materialized_view(
    name=GOLD_TREATMENT_SURVIVAL_SUMMARY,
    comment="Gold business-ready survival summary by diabetic treatment group sourced from the silver cohort.",
)
def gold_diabetic_treatment_survival_summary() -> DataFrame:
    cohort = spark.read.table(SILVER_DIABETIC_TREATMENT_COHORT)
    return build_survival_summary(cohort)


@dp.materialized_view(
    name="survival_statistics",
    comment="Compatibility alias for the gold diabetic treatment survival curve.",
)
def survival_statistics() -> DataFrame:
    cohort = spark.read.table(SILVER_DIABETIC_TREATMENT_COHORT)
    return build_survival_statistics(cohort)
