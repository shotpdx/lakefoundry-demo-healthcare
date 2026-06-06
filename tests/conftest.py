"""Pytest fixtures for data quality tests.

Configure SQL executor via environment variable:
    LAKEFOUNDRY_SQL_EXECUTOR=module.path:function_name

Example:
    LAKEFOUNDRY_SQL_EXECUTOR=my_module:execute_sql
"""
import os
import sys
import types
from typing import Any, Callable

import pytest


try:
    from pyspark import pipelines as _pipelines  # noqa: F401
except Exception:
    sys.modules.setdefault(
        "pyspark.pipelines",
        types.SimpleNamespace(materialized_view=lambda **kwargs: (lambda func: func)),
    )


@pytest.fixture(scope="module")
def sql_executor() -> Callable[[str], Any]:
    """Provide a SQL executor function for data quality tests.
    
    The executor is configured via LAKEFOUNDRY_SQL_EXECUTOR environment variable
    in the format 'module.path:function_name'.
    
    Returns:
        A callable that accepts a SQL query string and returns results.
    
    Raises:
        pytest.skip: If the environment variable is not configured.
    """
    dbx = os.environ.get("LAKEFOUNDRY_SQL_EXECUTOR")
    if not dbx:
        pytest.skip("LAKEFOUNDRY_SQL_EXECUTOR is not configured for SQL validation tests")

    module_name, _, attr_name = dbx.partition(":")
    if not attr_name:
        pytest.skip(f"Invalid LAKEFOUNDRY_SQL_EXECUTOR format: {dbx}. Expected 'module:function'")
    
    try:
        module = __import__(module_name, fromlist=[attr_name])
        executor = getattr(module, attr_name, None)
        if executor is None:
            pytest.skip(f"Function '{attr_name}' not found in module '{module_name}'")
        return executor
    except ImportError as e:
        pytest.skip(f"Could not import module '{module_name}': {e}")
