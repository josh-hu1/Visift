import numpy as np
import pandas as pd
import pytest

from evaluation.real_world.baseline import baseline_analysis, effect_size
from visift.profiler import profile_dataset
from visift.redundancy import bias_corrected_cramers_v


@pytest.fixture
def frame(rng):
    n = 300
    x = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], n)
    shift = pd.Series(group).map({"a": 0.0, "b": 0.5, "c": 1.0}).to_numpy()
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n),
            "x": x,
            "y": 2 * x + rng.normal(size=n),
            "noise": rng.normal(size=n),
            "group": group,
            "value": shift + rng.normal(size=n),
            "converted": rng.random(n) < np.where(group == "c", 0.6, 0.2),
        }
    )


def _key(chart):
    return (chart["chart"], chart["x"], chart.get("y"), chart.get("aggregation"))


def test_strongest_correlation_ranks_first(frame):
    ranked = baseline_analysis(frame)["recommendations"]
    assert _key(ranked[0]) == ("scatter", "x", "y", None)


def test_single_variable_charts_rank_after_all_bivariate_charts(frame):
    ranked = baseline_analysis(frame)["recommendations"]
    first_univariate = next(i for i, c in enumerate(ranked) if c.get("y") is None)
    assert all(c.get("y") is None for c in ranked[first_univariate:])


def test_no_diversity_step(frame):
    keys = {_key(c) for c in baseline_analysis(frame)["recommendations"]}
    assert ("bar", "group", "value", "mean") in keys
    assert ("box", "group", "value", None) in keys


def test_effect_sizes_use_the_documented_measures(frame):
    types = {
        c["name"]: c["semantic_type"] for c in profile_dataset(frame)["column_profiles"]
    }
    rate = {"chart": "bar", "x": "group", "y": "converted", "aggregation": "mean"}
    assert effect_size(frame, rate, types) == pytest.approx(
        bias_corrected_cramers_v(frame["group"], frame["converted"])
    )
    scatter = {"chart": "scatter", "x": "x", "y": "y"}
    assert effect_size(frame, scatter, types) == pytest.approx(
        abs(frame["x"].corr(frame["y"], method="spearman"))
    )
    assert effect_size(frame, {"chart": "histogram", "x": "x"}, types) == 0.0


def test_ranking_is_deterministic(frame):
    first = baseline_analysis(frame)["recommendations"]
    second = baseline_analysis(frame)["recommendations"]
    assert [_key(c) for c in first] == [_key(c) for c in second]


def test_target_mode_lists_target_charts_first_then_the_rest(frame):
    plain = baseline_analysis(frame)["recommendations"]
    ranked = baseline_analysis(frame, target="converted")["recommendations"]

    def is_driver(chart):
        return chart.get("y") is not None and "converted" in (chart["x"], chart["y"])

    drivers = [c for c in ranked if is_driver(c)]
    assert drivers and ranked[: len(drivers)] == drivers
    scores = [c["score"] for c in drivers]
    assert scores == sorted(scores, reverse=True)

    # The rest keeps the no-target order.
    rest = [_key(c) for c in ranked[len(drivers) :]]
    assert rest == [_key(c) for c in plain if not is_driver(c)]
