"""Each chart scorer should rank a planted pattern above a null and stay in 0-100."""

import numpy as np
import pandas as pd
import pytest

N = 400


def _base_frame(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, N)
    segment = rng.choice(list("ABCD"), N)
    segment_index = pd.Series(segment).map({"A": 0, "B": 1, "C": 2, "D": 3}).to_numpy()
    days = np.arange(N)
    return pd.DataFrame(
        {
            "x": x,
            "noise": rng.normal(0, 1, N),
            "linear": 2 * x + rng.normal(0, 0.5, N),
            "segment": segment,
            "by_segment": segment_index * 2 + rng.normal(0, 1, N),
            "bimodal": np.concatenate(
                [rng.normal(-4, 1, N // 2), rng.normal(4, 1, N - N // 2)]
            ),
            "converted": rng.random(N) < np.where(segment_index >= 2, 0.7, 0.1),
            "converted_null": rng.random(N) < 0.3,
            "skewed_category": rng.choice(list("ABCD"), N, p=[0.7, 0.1, 0.1, 0.1]),
            "date": pd.date_range("2020-01-01", periods=N, freq="D"),
            "trend": days * 0.5 + rng.normal(0, 5, N),
            "seasonal": 100 + 20 * np.sin(2 * np.pi * days / 30) + rng.normal(0, 3, N),
            "level_shift": np.where(days < N // 2, 100, 140) + rng.normal(0, 5, N),
            "flat": rng.normal(100, 5, N),
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _base_frame(42)


# Each case: (planted candidate, null candidate) as (chart, x, y, aggregation).
PAIRS = [
    pytest.param(
        ("scatter", "x", "linear", None),
        ("scatter", "x", "noise", None),
        id="scatter-linear",
    ),
    pytest.param(
        ("histogram", "bimodal", None, None),
        ("histogram", "noise", None, None),
        id="histogram-bimodal",
    ),
    pytest.param(
        ("box", "segment", "by_segment", None),
        ("box", "segment", "noise", None),
        id="box-group-shift",
    ),
    pytest.param(
        ("bar", "segment", "by_segment", "mean"),
        ("bar", "segment", "noise", "mean"),
        id="bar-mean",
    ),
    pytest.param(
        ("bar", "segment", "converted", "mean"),
        ("bar", "segment", "converted_null", "mean"),
        id="bar-binary-rate",
    ),
    pytest.param(
        ("bar", "skewed_category", None, "count"),
        ("bar", "segment", None, "count"),
        id="bar-count-imbalance",
    ),
    pytest.param(
        ("line", "date", "trend", None), ("line", "date", "flat", None), id="line-trend"
    ),
    pytest.param(
        ("line", "date", "seasonal", None),
        ("line", "date", "flat", None),
        id="line-seasonal",
    ),
    pytest.param(
        ("line", "date", "level_shift", None),
        ("line", "date", "flat", None),
        id="line-level-shift",
    ),
]


@pytest.mark.parametrize(("planted", "null"), PAIRS)
def test_planted_pattern_beats_null(score, frame, planted, null):
    strong = score(frame, *planted)["score"]
    weak = score(frame, *null)["score"]

    assert strong > weak + 20, (strong, weak)


NULL_CASES = [
    pytest.param(("scatter", "x", "noise", None), id="scatter"),
    pytest.param(("histogram", "noise", None, None), id="histogram"),
    pytest.param(("box", "segment", "noise", None), id="box"),
    pytest.param(("bar", "segment", "noise", "mean"), id="bar-mean"),
    pytest.param(("bar", "segment", "converted_null", "mean"), id="bar-binary-rate"),
    pytest.param(("bar", "segment", None, "count"), id="bar-count"),
    pytest.param(("line", "date", "flat", None), id="line"),
]


@pytest.mark.parametrize("null", NULL_CASES)
def test_null_false_positive_rate(score, null):
    """
    Pure-noise charts should rarely reach "Moderate" (>= 50) and never
    "Strong" (>= 65). One seed is not enough to check this: a single null
    can land in the tail (e.g. a box plot whose group IQRs differ by chance).
    """

    scores = np.array([score(_base_frame(seed), *null)["score"] for seed in range(50)])

    assert (scores >= 50).mean() <= 0.10, np.sort(scores)[-5:]
    assert scores.max() < 65, np.sort(scores)[-5:]


ALL_CANDIDATES = [
    ("scatter", "x", "linear", None),
    ("scatter", "x", "noise", None),
    ("histogram", "bimodal", None, None),
    ("histogram", "noise", None, None),
    ("box", "segment", "by_segment", None),
    ("bar", "segment", "by_segment", "mean"),
    ("bar", "segment", "converted", "mean"),
    ("bar", "segment", None, "count"),
    ("line", "date", "trend", None),
    ("line", "date", "flat", None),
]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scores_stay_in_range(score, seed):
    frame = _base_frame(seed)
    for chart, x, y, aggregation in ALL_CANDIDATES:
        result = score(frame, chart, x, y, aggregation)
        assert 0 <= result["score"] <= 100, (chart, x, y, result["score"])
        assert "reasons" in result and "components" in result


def test_scores_stay_in_range_on_degenerate_columns(score, rng):
    n = 50
    df = pd.DataFrame(
        {
            "constant": np.ones(n),
            "x": rng.normal(0, 1, n),
            "mostly_missing": [np.nan] * (n - 3) + [1.0, 2.0, 3.0],
            "group": rng.choice(["a", "b"], n),
            "date": pd.date_range("2024-01-01", periods=n),
        }
    )
    for chart, x, y, aggregation in [
        ("scatter", "x", "constant", None),
        ("scatter", "x", "mostly_missing", None),
        ("histogram", "constant", None, None),
        ("box", "group", "constant", None),
        ("bar", "group", "constant", "mean"),
        ("line", "date", "constant", None),
    ]:
        result = score(df, chart, x, y, aggregation)
        assert 0 <= result["score"] <= 100, (chart, x, y, result["score"])


def test_scoring_is_deterministic(score, frame):
    first = score(frame, "scatter", "x", "linear")["score"]
    second = score(frame, "scatter", "x", "linear")["score"]
    assert first == second
