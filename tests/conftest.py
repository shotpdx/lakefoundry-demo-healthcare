import os
from typing import Any

import pytest


@pytest.fixture(scope="module")
def sql_executor() -> Any:
    dbx = os.environ.get("LAKEFOUNDRY_SQL_EXECUTOR")
    if not dbx:
        pytest.skip("LAKEFOUNDRY_SQL_EXECUTOR is not configured for SQL validation tests")

    module_name, _, attr_name = dbx.partition(":")
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
