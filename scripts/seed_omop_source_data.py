#!/usr/bin/env python3
"""Seed a deterministic minimal OMOP-compatible source dataset in Unity Catalog.

Creates or replaces the six source tables required by the diabetic outcomes
Bronze layer using a small, deterministic dataset that exercises downstream
Silver and Gold behavior.

Usage:
    python scripts/seed_omop_source_data.py
    python scripts/seed_omop_source_data.py --catalog cme_outcomes_uswest --schema omop_seed
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

DEFAULT_CATALOG = "cme_outcomes_uswest"
DEFAULT_SCHEMA = "omop_seed"


@dataclass(frozen=True)
class SeedTable:
    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


PERSON = SeedTable(
    name="person",
    columns=(
        "person_id",
        "gender_concept_id",
        "year_of_birth",
        "race_concept_id",
        "ethnicity_concept_id",
    ),
    rows=(
        (1, 8507, 1975, 8527, 38003564),
        (2, 8532, 1968, 8516, 38003563),
        (3, 8507, 1982, 8527, 38003564),
    ),
)

CONDITION_OCCURRENCE = SeedTable(
    name="condition_occurrence",
    columns=(
        "condition_occurrence_id",
        "person_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_end_date",
        "condition_source_value",
    ),
    rows=(
        (101, 1, 201826, "2020-01-05", "2020-01-10", "44054006"),
        (102, 2, 201826, "2020-02-10", "2020-02-14", "44054006"),
        (103, 3, 201826, "2020-03-15", "2020-03-20", "44054006"),
    ),
)

DRUG_EXPOSURE = SeedTable(
    name="drug_exposure",
    columns=(
        "drug_exposure_id",
        "person_id",
        "drug_concept_id",
        "drug_exposure_start_date",
        "drug_exposure_end_date",
        "drug_source_value",
    ),
    rows=(
        (201, 1, 1503297, "2020-01-07", "2020-02-07", "metformin"),
        (202, 2, 19059796, "2020-02-12", "2020-03-12", "insulin_glargine"),
        (203, 3, 1559684, "2020-03-18", "2020-04-18", "glipizide"),
    ),
)

DEATH = SeedTable(
    name="death",
    columns=("person_id", "death_date", "death_datetime"),
    rows=(
        (1, "2020-06-01", "2020-06-01 08:30:00"),
        (2, None, None),
        (3, "2020-05-01", "2020-05-01 13:15:00"),
    ),
)

OBSERVATION_PERIOD = SeedTable(
    name="observation_period",
    columns=(
        "observation_period_id",
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    ),
    rows=(
        (301, 1, "2019-01-01", "2020-12-31"),
        (302, 2, "2019-06-01", "2021-01-31"),
        (303, 3, "2019-09-01", "2020-06-15"),
    ),
)

CONCEPT = SeedTable(
    name="concept",
    columns=("concept_id", "concept_code", "concept_name", "vocabulary_id"),
    rows=(
        (1503297, "metformin", "Metformin Hydrochloride", "RxNorm"),
        (19059796, "insulin_glargine", "Insulin Glargine", "RxNorm"),
        (1559684, "glipizide", "Glipizide", "RxNorm"),
    ),
)

SEED_TABLES = (
    PERSON,
    CONDITION_OCCURRENCE,
    DRUG_EXPOSURE,
    DEATH,
    OBSERVATION_PERIOD,
    CONCEPT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="Destination Unity Catalog catalog")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Destination Unity Catalog schema")
    return parser.parse_args()


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


COLUMN_TYPES = {
    "person": ("BIGINT", "BIGINT", "INT", "BIGINT", "BIGINT"),
    "condition_occurrence": ("BIGINT", "BIGINT", "BIGINT", "DATE", "DATE", "STRING"),
    "drug_exposure": ("BIGINT", "BIGINT", "BIGINT", "DATE", "DATE", "STRING"),
    "death": ("BIGINT", "DATE", "TIMESTAMP"),
    "observation_period": ("BIGINT", "BIGINT", "DATE", "DATE"),
    "concept": ("BIGINT", "STRING", "STRING", "STRING"),
}


def create_table_ddl(catalog: str, schema: str, table: SeedTable) -> str:
    columns = ",\n  ".join(
        f"{name} {dtype}" for name, dtype in zip(table.columns, COLUMN_TYPES[table.name], strict=True)
    )
    return f"CREATE OR REPLACE TABLE {catalog}.{schema}.{table.name} (\n  {columns}\n)"


def insert_statement(catalog: str, schema: str, table: SeedTable) -> str:
    values = ",\n  ".join(
        "(" + ", ".join(_sql_literal(value) for value in row) + ")" for row in table.rows
    )
    return (
        f"INSERT INTO {catalog}.{schema}.{table.name} ({', '.join(table.columns)}) VALUES\n  {values}"
    )


def _execute_sql(workspace_client: WorkspaceClient, warehouse_id: str, statement: str) -> None:
    response = workspace_client.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
    )
    status = getattr(response, "status", None)
    state = getattr(status, "state", None)
    if state not in {StatementState.SUCCEEDED, StatementState.CLOSED}:
        raise RuntimeError(f"SQL statement failed with state {state}: {statement}")


def seed_tables(workspace_client: WorkspaceClient, warehouse_id: str, catalog: str, schema: str, tables: Iterable[SeedTable] = SEED_TABLES) -> None:
    statements = [
        f"CREATE CATALOG IF NOT EXISTS {catalog}",
        f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}",
    ]

    for table in tables:
        statements.append(create_table_ddl(catalog, schema, table))
        statements.append(insert_statement(catalog, schema, table))

    for statement in statements:
        _execute_sql(workspace_client, warehouse_id, statement)

    for table in tables:
        print(f"Seeded {catalog}.{schema}.{table.name} with {len(table.rows)} rows")


def _resolve_warehouse_id(workspace_client: WorkspaceClient) -> str:
    warehouses = list(workspace_client.warehouses.list())
    for warehouse in warehouses:
        if getattr(warehouse, "state", None) in {"RUNNING", "STARTING"}:
            return warehouse.id
    if warehouses:
        return warehouses[0].id
    raise RuntimeError("No SQL warehouse available for seeding")


def main() -> int:
    args = parse_args()
    workspace_client = WorkspaceClient()
    warehouse_id = _resolve_warehouse_id(workspace_client)
    print(f"Using warehouse {warehouse_id} for OMOP seed")
    seed_tables(workspace_client=workspace_client, warehouse_id=warehouse_id, catalog=args.catalog, schema=args.schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
