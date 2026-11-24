import numpy as np
import pandas as pd
import pytest

from visift.candidate_generator import generate_candidates, is_line_eligible_time_axis
from visift.profiler import profile_dataset


def _candidates(df):
    return generate_candidates(df, profile_dataset(df))


@pytest.fixture
def frame(rng):
    n = 120
    return pd.DataFrame(
        {
            "customer_id": np.arange(5000, 5000 + n),
            "date": pd.date_range("2023-01-01", periods=n),
            "mnth": rng.integers(1, 13, n),
            "year": rng.integers(2015, 2024, n),
            "region": rng.choice(["North", "South", "East"], n),
            "revenue": rng.normal(100, 20, n),
            "cost": rng.normal(60, 10, n),
            "churned": rng.choice([True, False], n),
        }
    )


def test_no_candidates_use_identifiers(frame):
    for candidate in _candidates(frame):
        assert "customer_id" not in (candidate.get("x"), candidate.get("y"))


def test_no_line_chart_on_cyclical_time_field(frame):
    lines = [c for c in _candidates(frame) if c["chart"] == "line"]
    line_axes = {c["x"] for c in lines}
    assert "mnth" not in line_axes
    assert {"date", "year"} <= line_axes


def test_expected_chart_families(frame):
    candidates = _candidates(frame)
    keys = {(c["chart"], c["x"], c.get("y"), c.get("aggregation")) for c in candidates}

    assert ("histogram", "revenue", None, None) in keys
    assert ("bar", "region", None, "count") in keys
    assert ("scatter", "revenue", "cost", None) in keys
    assert ("box", "region", "revenue", None) in keys
    assert ("bar", "region", "revenue", "mean") in keys
    assert ("line", "date", "revenue", None) in keys
    # Boolean outcome rate by a category.
    assert ("bar", "region", "churned", "mean") in keys


def test_two_column_candidates_are_not_duplicated(frame):
    keys = [
        (c["chart"], c["x"], c.get("y"), c.get("aggregation"))
        for c in _candidates(frame)
    ]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    ("name", "semantic_type", "eligible"),
    [
        ("date", "datetime", True),
        ("year", "temporal", True),
        ("month", "temporal", False),
        ("Day of Week", "temporal", False),
        ("hr", "temporal", False),
        ("revenue", "numeric_continuous", False),
    ],
)
def test_line_eligibility(name, semantic_type, eligible):
    assert is_line_eligible_time_axis(name, semantic_type) is eligible
