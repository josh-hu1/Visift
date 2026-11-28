"""Binned-rate bars: generation, bins, score, takeaway and exported code."""

import numpy as np
import pandas as pd
import pytest

from visift.binning import binned_rate_table, format_bound
from visift.candidate_generator import generate_candidates
from visift.export import chart_code
from visift.insight import plain_english_insight
from visift.profiler import profile_dataset
from visift.scoring.bar import score_bar_chart
from visift.visualization.renderer import render_candidate, visualization_title


def _u_shape(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    rate = 0.05 + 0.35 * ((age - 54) / 36) ** 2  # high at both ends
    return pd.DataFrame(
        {
            "age": age,
            "subscribed": np.where(rng.random(n) < rate, "yes", "no"),
            "score": rng.integers(1, 5, n),  # numeric, < 10 distinct values
        }
    )


CANDIDATE = {
    "chart": "bar",
    "x": "age",
    "y": "subscribed",
    "aggregation": "binned_rate",
}


def test_generated_for_numeric_with_enough_values_vs_boolean():
    df = _u_shape()
    candidates = generate_candidates(df, profile_dataset(df))
    binned = [c for c in candidates if c.get("aggregation") == "binned_rate"]
    assert binned == [CANDIDATE]  # not for "score" (4 distinct values)


def test_bins_are_ordered_quintiles_with_value_range_labels():
    table = binned_rate_table(_u_shape(), "age", "subscribed")
    assert table is not None and len(table) == 5
    assert table["low"].is_monotonic_increasing
    assert table["bin"][0] == f"18–{format_bound(table['high'][0])}"
    assert table["rate"].between(0, 1).all()
    assert format_bound(1250.4) == "1,250" and format_bound(0.25) == "0.25"


def test_u_shape_scores_and_reads_as_both_ends():
    df = _u_shape()
    result = score_bar_chart(df, CANDIDATE, profile_dataset(df))
    assert result["score"] >= 50
    takeaway = plain_english_insight(df, CANDIDATE)
    assert takeaway.endswith("it is higher at both ends.")
    assert visualization_title(CANDIDATE) == "Average Subscribed by Age (binned)"


def test_too_few_values_scores_zero_signal():
    df = pd.DataFrame({"x": [1, 1, 1, 2, 2, 2] * 20, "y": [0, 1] * 60})
    result = score_bar_chart(
        df,
        {"chart": "bar", "x": "x", "y": "y", "aggregation": "binned_rate"},
        profile_dataset(df),
    )
    assert result["components"]["signal"] == 0


def test_exported_code_rebuilds_the_chart(tmp_path):
    df = _u_shape()
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    namespace: dict = {}
    exec(compile(chart_code(CANDIDATE, str(path), reloaded), "<s>", "exec"), namespace)  # noqa: S102
    want = render_candidate(reloaded, CANDIDATE).data[0]
    got = namespace["fig"].data[0]
    assert list(want.x) == list(got.x)
    assert np.allclose(want.y, got.y)


@pytest.mark.parametrize("n", [300, 1000])
def test_pure_noise_rarely_scores_moderate(n):
    # Guard on the calibration (see evaluation/binned_rate_calibration.py).
    scores = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(
            {"x": rng.normal(40, 10, n).round(1), "y": rng.random(n) < 0.3}
        )
        scores.append(
            score_bar_chart(df, {**CANDIDATE, "x": "x", "y": "y"}, profile_dataset(df))[
                "score"
            ]
        )
    assert max(scores) < 65
