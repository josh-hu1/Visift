import numpy as np
import pandas as pd
import pytest

from tests.conftest import FIXTURES
from visift.semantic_types import classify_column

N = 200
_rng = np.random.default_rng(0)


def _ints(low, high, name):
    return pd.Series(_rng.integers(low, high, N), name=name)


def _choice(values, name):
    return pd.Series(_rng.choice(values, N), name=name)


CASES = [
    # (case id, series, expected semantic type)
    ("empty", pd.Series([np.nan] * N, name="blank"), "empty"),
    ("bool dtype", _choice([True, False], "active"), "boolean"),
    ("yes/no text", _choice(["yes", "no"], "subscribed"), "boolean"),
    ("Y/N text", _choice(["Y", "N"], "flag"), "boolean"),
    ("0/1 indicator", _ints(0, 2, "holiday"), "boolean"),
    ("parsed datetime", pd.Series(pd.date_range("2024-01-01", periods=N)), "datetime"),
    (
        "ISO date strings",
        pd.Series(pd.date_range("2024-01-01", periods=N).astype(str), name="date"),
        "datetime",
    ),
    (
        "sequential integer id",
        pd.Series(np.arange(1000, 1000 + N), name="customer_id"),
        "identifier",
    ),
    (
        "uuid-named strings",
        pd.Series([f"{i:08x}-aaaa" for i in range(N)], name="uuid"),
        "identifier",
    ),
    ("year", _ints(2010, 2024, "year"), "temporal"),
    ("numeric month", _ints(1, 13, "mnth"), "temporal"),
    (
        "month abbreviations",
        _choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun"], "month"),
        "categorical",
    ),
    ("numeric-coded season", _ints(1, 5, "season"), "categorical"),
    ("numeric-coded weekday", _ints(0, 7, "weekday"), "categorical"),
    ("zip code", _ints(90000, 96000, "zip_code"), "geographic"),
    ("1-5 rating", _ints(1, 6, "rating"), "ordinal"),
    ("small count", _ints(0, 8, "children"), "numeric_discrete"),
    (
        "continuous",
        pd.Series(_rng.normal(50, 10, N), name="price"),
        "numeric_continuous",
    ),
    ("string category", _choice(list("ABCD"), "segment"), "categorical"),
    (
        "free text",
        pd.Series([f"free text {i} words" for i in range(N)], name="comment"),
        "text",
    ),
]


@pytest.mark.parametrize(
    ("series", "expected"), [(s, e) for _, s, e in CASES], ids=[c[0] for c in CASES]
)
def test_classify_column(series, expected):
    assert classify_column(series) == expected


FIXTURE_EXPECTATIONS = {
    "sample_data.csv": {
        "customer_id": "identifier",
        "date": "datetime",
        "region": "categorical",
        "revenue": "numeric_continuous",
        "units": "numeric_discrete",
        "subscribed": "boolean",
        "notes": "text",
    },
    "tricky_data.csv": {
        "employee_id": "identifier",
        "year": "temporal",
        "zip_code": "geographic",
        "rating": "ordinal",
        "price": "numeric_continuous",
        "department": "categorical",
        "signup_date": "datetime",
        "description": "text",
        "is_active": "boolean",
    },
}


@pytest.mark.parametrize("filename", sorted(FIXTURE_EXPECTATIONS))
def test_fixture_tables(filename):
    df = pd.read_csv(FIXTURES / filename)
    inferred = {column: classify_column(df[column]) for column in df.columns}
    assert inferred == FIXTURE_EXPECTATIONS[filename]
