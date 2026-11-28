import json
import time

import numpy as np
import pandas as pd
import pytest

from evaluation.datasets import build_benchmark_suite
from visift.candidate_generator import generate_candidates
from visift.datasets import load_sample
from visift.pipeline import analyze
from visift.profiler import profile_dataset
from visift.redundancy import (
    bias_corrected_cramers_v,
    duplicate_aliases,
    eta_squared,
    find_redundancies,
    flagged_pairs,
    normalized_conditional_entropy,
)
from visift.scoring.engine import diversify_recommendations, score_all_candidates

N = 500


def _find(df):
    return find_redundancies(df, profile_dataset(df))


def _kinds(df):
    return {(r["kind"], r["columns"]) for r in _find(df)}


def _correlated(rng, r):
    x = rng.normal(size=N)
    return x, r * x + np.sqrt(1 - r * r) * rng.normal(size=N)


# --------------------------------------------------
# Thresholds: twins vs strong-but-real relationships
# --------------------------------------------------


def test_numeric_twins_are_flagged(rng):
    x, twin = _correlated(rng, 0.995)
    df = pd.DataFrame({"temp": x, "feels_like": twin})
    (finding,) = _find(df)
    assert finding["kind"] == "near_duplicate_numeric"
    assert finding["value"] >= 0.97
    assert "move together" in finding["reason"]


@pytest.mark.parametrize("r", [0.8, 0.9, 0.95])
def test_strong_real_numeric_relationships_are_not_flagged(rng, r):
    x, y = _correlated(rng, r)
    assert _find(pd.DataFrame({"x": x, "y": y})) == []


def test_negative_twins_are_flagged(rng):
    x = rng.normal(size=N)
    df = pd.DataFrame({"x": x, "minus_x": -x + rng.normal(0, 0.01, N)})
    (finding,) = _find(df)
    assert "opposite directions" in finding["reason"]


def test_relabelled_category_is_a_categorical_duplicate(rng):
    region = rng.choice(["north", "south", "east", "west"], N)
    code = pd.Series(region).map({"north": "N", "south": "S", "east": "E", "west": "W"})
    df = pd.DataFrame({"region": region, "region_code": code})
    assert ("near_duplicate_categorical", ("region", "region_code")) in _kinds(df)


def test_strong_but_noisy_category_association_is_not_flagged(rng):
    a = rng.choice(list("ABC"), N)
    # 70% of rows copy a's label, 30% are random: a real, strong association.
    b = np.where(rng.random(N) < 0.7, a, rng.choice(list("ABC"), N))
    assert _find(pd.DataFrame({"a": a, "b": b})) == []


def test_near_deterministic_category_is_flagged(rng):
    # Twelve products roll up into four product lines: a coarsening.
    product = rng.choice([f"product_{chr(65 + i)}" for i in range(12)], N)
    line = pd.Series(product).map(lambda p: f"line_{(ord(p[-1]) - 65) // 3}")
    df = pd.DataFrame({"product": product, "product_line": line})
    kinds = _kinds(df)
    assert ("determined_categorical", ("product", "product_line")) in kinds
    # A coarsening is not a twin, and does not determine the finer column.
    assert ("near_duplicate_categorical", ("product", "product_line")) not in kinds
    assert ("determined_categorical", ("product_line", "product")) not in kinds


def test_near_deterministic_numeric_is_flagged(rng):
    group = rng.choice(["low", "mid", "high"], N)
    level = pd.Series(group).map({"low": 1.0, "mid": 5.0, "high": 9.0})
    df = pd.DataFrame({"group": group, "level": level + rng.normal(0, 0.1, N)})
    assert ("determined_numeric", ("group", "level")) in _kinds(df)


def test_planted_group_difference_is_not_flagged(rng):
    group = rng.choice(["a", "b", "c"], N)
    shift = pd.Series(group).map({"a": 0.0, "b": 1.0, "c": 2.0})
    df = pd.DataFrame({"group": group, "value": shift + rng.normal(0, 1, N)})
    assert _find(df) == []


def test_no_synthetic_benchmark_scenario_is_flagged():
    flagged = [
        scenario
        for scenario in build_benchmark_suite()
        if find_redundancies(scenario.dataframe, profile_dataset(scenario.dataframe))
    ]
    assert flagged == []


# --------------------------------------------------
# Folding and dataset notes
# --------------------------------------------------


@pytest.fixture
def twins(rng):
    days = np.arange(300)
    temp = 15 + 10 * np.sin(2 * np.pi * days / 365) + rng.normal(0, 1, 300)
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=300),
            "temp": temp,
            "atemp": temp * 0.9 + rng.normal(0, 0.2, 300),
            "rentals": 100 + 8 * temp + rng.normal(0, 20, 300),
            "season": np.where(days < 150, "spring", "summer"),
        }
    )


def test_twin_charts_fold_into_the_representative(twins):
    analysis = analyze(twins)
    recommendations = analysis["recommendations"]

    assert analysis["aliases"] == {"temp": "temp", "atemp": "temp"}

    # No ranked chart uses atemp as its primary view...
    for recommendation in recommendations:
        assert "atemp" not in (recommendation["x"], recommendation.get("y"))

    # ...but its versions survive as alternatives of the temp charts.
    line = next(
        r for r in recommendations if r["chart"] == "line" and r.get("y") == "temp"
    )
    assert any("atemp" in alt["title"].lower() for alt in line["alternative_charts"])


def test_chart_between_twins_moves_to_notes(twins):
    analysis = analyze(twins)

    noted = {(c["chart"], c["x"], c.get("y")) for c in analysis["noted_charts"]}
    assert ("scatter", "temp", "atemp") in noted

    for recommendation in analysis["recommendations"]:
        assert {recommendation["x"], recommendation.get("y")} != {"temp", "atemp"}

    (note,) = analysis["notes"]
    assert note["columns"] == ("temp", "atemp")
    assert not note["involves_target"]


def test_nothing_is_lost(twins):
    analysis = analyze(twins)

    ranked = sum(1 + len(r["alternative_charts"]) for r in analysis["recommendations"])
    assert ranked + len(analysis["noted_charts"]) == len(analysis["scored"])


def test_target_twin_is_noted_and_target_stays_representative(twins):
    analysis = analyze(twins, target="atemp")

    assert analysis["aliases"]["temp"] == "atemp"
    (note,) = analysis["notes"]
    assert note["involves_target"]
    assert "Your outcome atemp has a near-duplicate, temp" in note["reason"]

    for driver in analysis["drivers"]:
        assert {driver["x"], driver.get("y")} != {"temp", "atemp"}
    assert analysis["recommendations"] == analysis["drivers"] + analysis["others"]


def test_notes_in_target_mode_for_unrelated_target(twins):
    analysis = analyze(twins, target="rentals")
    assert [n["columns"] for n in analysis["notes"]] == [("temp", "atemp")]
    assert all(not n["involves_target"] for n in analysis["notes"])


def test_representative_prefers_fewest_missing_then_column_order(rng):
    x = rng.normal(size=N)
    df = pd.DataFrame({"a": x, "b": x * 2, "c": x * 3})
    assert set(duplicate_aliases(df, _find(df)).values()) == {"a"}

    df.loc[:9, "a"] = np.nan
    assert set(duplicate_aliases(df, _find(df)).values()) == {"b"}


def test_without_redundancies_analyze_matches_the_plain_pipeline(rng):
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "group": rng.choice(list("abc"), n),
        }
    )
    profile = profile_dataset(df)
    plain = diversify_recommendations(
        score_all_candidates(df, generate_candidates(df, profile), profile)
    )
    analysis = analyze(df)
    assert analysis["notes"] == []
    # json comparison: skipped MI null statistics are NaN, and NaN != NaN.
    assert json.dumps(analysis["recommendations"], default=str) == json.dumps(
        plain, default=str
    )


# --------------------------------------------------
# Joint determination by two columns
# --------------------------------------------------


def _calendar(rng, n=N, holiday_rate=0.05):
    """workingday = a weekday (1-5) that is not a holiday, as in Bike Sharing."""

    weekday = rng.integers(0, 7, n)
    holiday = ((weekday >= 1) & (weekday <= 5) & (rng.random(n) < holiday_rate)).astype(
        int
    )
    workingday = ((weekday >= 1) & (weekday <= 5) & (holiday == 0)).astype(int)
    return pd.DataFrame(
        {
            "weekday": pd.Series(weekday).map(lambda d: f"day{d}"),
            "holiday": holiday.astype(bool),
            "workingday": workingday.astype(bool),
            "noise": rng.choice(list("abc"), n),
        }
    )


def _joint(findings):
    return [r for r in findings if r["kind"].startswith("jointly_determined")]


def test_column_determined_by_two_others_is_flagged(rng):
    findings = _find(_calendar(rng))
    (joint,) = _joint(findings)

    assert joint["kind"] == "jointly_determined_categorical"
    assert set(joint["columns"]) == {"weekday", "holiday", "workingday"}
    # workingday has the most entropy, so it is the note's primary column;
    # holiday is just as determined by the other two.
    assert joint["columns"][-1] == "workingday"
    assert joint["determined"] == ("workingday", "holiday")
    assert joint["value"] <= 0.05
    assert "together" in joint["reason"]
    # No single column determines another, so nothing pairwise is reported.
    assert len(findings) == 1

    pairs = flagged_pairs(findings)
    assert frozenset(("weekday", "workingday")) in pairs
    assert frozenset(("holiday", "workingday")) in pairs
    assert not any("noise" in pair for pair in pairs)


def test_joint_determination_survives_missing_values(rng):
    df = _calendar(rng).astype(object)
    df.loc[rng.random(N) < 0.1, "weekday"] = None
    df.loc[rng.random(N) < 0.1, "workingday"] = None

    (joint,) = _joint(_find(df))
    assert joint["columns"][-1] == "workingday"


def test_sum_of_two_columns_is_flagged(rng):
    casual = rng.gamma(4, 200, N)
    registered = rng.gamma(4, 200, N)
    df = pd.DataFrame(
        {
            "casual": casual,
            "registered": registered,
            "cnt": casual + registered,
            "temp": rng.normal(20, 5, N),
        }
    )

    (joint,) = _joint(_find(df))

    assert joint["kind"] == "jointly_determined_numeric"
    assert joint["columns"] == ("casual", "registered", "cnt")
    assert joint["determined"] == ("cnt",)
    assert joint["statistic"] == "R²"
    assert "cnt ≈ 1·casual + 1·registered" in joint["reason"]

    pairs = flagged_pairs([joint])
    assert pairs == {frozenset(("cnt", "casual")), frozenset(("cnt", "registered"))}


def test_product_of_two_columns_is_flagged(rng):
    units = rng.integers(1, 60, N).astype(float)
    price = rng.lognormal(3, 0.5, N)
    df = pd.DataFrame({"units": units, "price": price, "revenue": units * price})

    (joint,) = _joint(_find(df))

    assert joint["columns"][-1] == "revenue"
    assert joint["statistic"] == "R² (log scale)"
    assert "revenue ≈ units^1 × price^1" in joint["reason"]


def test_charts_between_the_inputs_stay_ranked(rng):
    casual = rng.gamma(4, 200, N)
    registered = 0.5 * casual + rng.gamma(4, 200, N)
    df = pd.DataFrame(
        {"casual": casual, "registered": registered, "cnt": casual + registered}
    )

    analysis = analyze(df)
    ranked = {frozenset((r["x"], r.get("y"))) for r in analysis["recommendations"]}
    noted = {frozenset((c["x"], c.get("y"))) for c in analysis["noted_charts"]}

    assert frozenset(("casual", "registered")) in ranked
    assert frozenset(("casual", "cnt")) in noted
    assert frozenset(("registered", "cnt")) in noted


@pytest.mark.parametrize("n", [60, 150, 400, 2000])
def test_noise_is_not_jointly_determined(n):
    for seed in range(10):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(
            {
                **{f"c{k}": rng.choice(list("abcdefg")[: 2 + k], n) for k in range(6)},
                # A rare flag: plug-in entropy in small cells looks low by chance.
                "rare": rng.random(n) < 3 / n,
                **{f"x{k}": rng.normal(size=n) for k in range(4)},
            }
        )
        assert _joint(_find(df)) == [], (n, seed)


def test_single_determinant_is_reported_pairwise_not_jointly(rng):
    x = rng.normal(size=N)
    df = pd.DataFrame(
        {"x": x, "twin": 2 * x + rng.normal(0, 0.01, N), "other": rng.normal(size=N)}
    )
    kinds = [r["kind"] for r in _find(df)]
    assert kinds == ["near_duplicate_numeric"]

    product = rng.choice([f"p{i}" for i in range(12)], N)
    line = pd.Series(product).map(lambda p: f"line_{int(p[1:]) // 3}")
    df = pd.DataFrame(
        {"product": product, "line": line, "store": rng.choice(list("ab"), N)}
    )
    assert [r["kind"] for r in _find(df)] == ["determined_categorical"]


def test_target_in_a_jointly_determined_triple(rng):
    df = _calendar(rng)
    df["rentals"] = 100 + 50 * df["workingday"] + rng.normal(0, 20, N)

    analysis = analyze(df, target="workingday")
    (note,) = analysis["notes"]
    assert note["involves_target"]

    for driver in analysis["drivers"]:
        pair = {driver["x"], driver.get("y")}
        assert pair not in ({"weekday", "workingday"}, {"holiday", "workingday"})


def test_bike_sharing_weekday_vs_workingday_leaves_the_ranking():
    analysis = analyze(load_sample("bike"))

    ranked = {frozenset((r["x"], r.get("y"))) for r in analysis["recommendations"]}
    noted = {frozenset((c["x"], c.get("y"))) for c in analysis["noted_charts"]}

    assert frozenset(("weekday", "workingday")) not in ranked
    assert frozenset(("weekday", "workingday")) in noted
    assert [n["kind"] for n in analysis["notes"]] == [
        "near_duplicate_numeric",
        "jointly_determined_categorical",
    ]


# --------------------------------------------------
# Statistics on codes match pd.crosstab
# --------------------------------------------------


def _entropy_bits(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def test_statistics_match_crosstab_definitions(rng):
    n = 300
    x = pd.Series(rng.choice(list("abcde"), n), dtype=object)
    y = pd.Series(rng.choice(list("xyz"), n), dtype=object)
    x[rng.random(n) < 0.1] = None
    y[rng.random(n) < 0.1] = None
    values = pd.Series(rng.normal(size=n))
    values[rng.random(n) < 0.1] = np.nan

    table = pd.crosstab(x, y).to_numpy(dtype=float)
    total = table.sum()
    h_y_given_x = sum(r.sum() / total * _entropy_bits(r) for r in table)
    expected_ratio = h_y_given_x / _entropy_bits(table.sum(axis=0))
    assert normalized_conditional_entropy(x, y) == pytest.approx(expected_ratio)

    rows, cols = table.shape
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0) / total
    phi2 = ((table - expected) ** 2 / expected).sum() / total
    phi2c = max(0.0, phi2 - (cols - 1) * (rows - 1) / (total - 1))
    k = min(cols - (cols - 1) ** 2 / (total - 1), rows - (rows - 1) ** 2 / (total - 1))
    assert bias_corrected_cramers_v(x, y) == pytest.approx(np.sqrt(phi2c / (k - 1)))

    pair = pd.DataFrame({"g": x, "v": values}).dropna()
    means = pair.groupby("g")["v"].transform("mean")
    expected_eta = ((means - pair["v"].mean()) ** 2).sum() / (
        (pair["v"] - pair["v"].mean()) ** 2
    ).sum()
    assert eta_squared(x, values) == pytest.approx(expected_eta)


def test_wide_categorical_table_is_fast():
    # pd.crosstab per pair took about 15 s here; codes and bincount take ~1 s.
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"q{i}": rng.choice(list("abcde"), 2000) for i in range(40)})
    profile = profile_dataset(df)

    start = time.perf_counter()
    assert find_redundancies(df, profile) == []
    assert time.perf_counter() - start < 6


# --------------------------------------------------
# Edge cases
# --------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) ** 2}),
        pd.DataFrame({"a": list("xy") * 25, "b": list("xxyy") * 12 + ["x", "y"]}),
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]}),
        pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0]}),
        pd.DataFrame({"const": [1.0] * 40, "other": np.arange(40.0)}),
    ],
    ids=["all-numeric", "all-categorical", "tiny", "single-column", "constant"],
)
def test_no_crash_on_edge_cases(frame):
    analysis = analyze(frame)
    assert isinstance(analysis["notes"], list)
