import numpy as np
import pandas as pd
import pytest

from evaluation.real_world import benchmark as real_world
from evaluation.real_world.expectations import (
    BANK_MARKETING,
    BIKE_SHARING,
    DATASET_DIR,
    PENGUINS,
    TITANIC,
)
from tests.conftest import FIXTURES, REPO_ROOT
from visift.candidate_generator import generate_candidates
from visift.profiler import profile_dataset
from visift.scoring.engine import diversify_recommendations, score_all_candidates
from visift.target import (
    involves_target,
    rank_with_target,
    split_by_target,
    suggest_target,
)


def _keys(candidates):
    return {(c["chart"], c["x"], c.get("y"), c.get("aggregation")) for c in candidates}


@pytest.fixture
def frame(rng):
    n = 200
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n),
            "region": rng.choice(["North", "South", "East"], n),
            "plan": rng.choice(["basic", "plus", "pro"], n),
            "revenue": rng.normal(100, 20, n),
            "tenure": rng.normal(24, 6, n),
            "has_promo": rng.choice([True, False], n),
            "churned": rng.choice([True, False], n),
        }
    )


# --------------------------------------------------
# No-target mode is unchanged
# --------------------------------------------------


def test_target_none_is_the_default(frame):
    profile = profile_dataset(frame)
    assert generate_candidates(frame, profile) == generate_candidates(
        frame, profile, target=None
    )


@pytest.mark.parametrize("target", ["churned", "revenue", "region"])
def test_target_mode_only_adds_target_candidates(frame, target):
    profile = profile_dataset(frame)
    base = generate_candidates(frame, profile)
    with_target = generate_candidates(frame, profile, target=target)

    assert with_target[: len(base)] == base
    for candidate in with_target[len(base) :]:
        assert target in (candidate["x"], candidate.get("y"))


def test_unknown_target_is_rejected(frame):
    with pytest.raises(ValueError):
        generate_candidates(frame, profile_dataset(frame), target="nope")


# --------------------------------------------------
# Every target-involving chart type is generated
# --------------------------------------------------


def test_binary_target_candidates(frame):
    keys = _keys(generate_candidates(frame, profile_dataset(frame), target="churned"))

    for category in ("region", "plan"):
        assert ("bar", category, "churned", "mean") in keys
    for numeric in ("revenue", "tenure"):
        assert ("box", "churned", numeric, None) in keys
    # Boolean predictor vs boolean target only exists with a target.
    assert ("bar", "has_promo", "churned", "mean") in keys


def test_numeric_target_candidates(frame):
    keys = _keys(generate_candidates(frame, profile_dataset(frame), target="revenue"))

    assert ("scatter", "revenue", "tenure", None) in keys
    for category in ("region", "plan", "churned"):
        assert ("box", category, "revenue", None) in keys
        assert ("bar", category, "revenue", "mean") in keys
    assert ("line", "date", "revenue", None) in keys


def test_categorical_target_candidates(frame):
    keys = _keys(generate_candidates(frame, profile_dataset(frame), target="plan"))

    for numeric in ("revenue", "tenure"):
        assert ("box", "plan", numeric, None) in keys


# --------------------------------------------------
# Drivers
# --------------------------------------------------


@pytest.fixture
def ranked(frame):
    profile = profile_dataset(frame)
    candidates = generate_candidates(frame, profile, target="churned")
    return diversify_recommendations(score_all_candidates(frame, candidates, profile))


def test_drivers_only_contain_target_charts(ranked):
    drivers, rest = split_by_target(ranked, "churned")

    assert drivers
    assert all(involves_target(r, "churned") for r in drivers)
    assert not any(involves_target(r, "churned") for r in rest)
    # Univariate charts of the target itself are not drivers.
    assert all(r.get("y") is not None for r in drivers)


def test_split_keeps_score_order_and_loses_nothing(ranked):
    drivers, rest = split_by_target(ranked, "churned")

    assert len(drivers) + len(rest) == len(ranked)
    for part in (drivers, rest):
        scores = [r["score"] for r in part]
        assert scores == sorted(scores, reverse=True)
    assert rank_with_target(ranked, "churned") == drivers + rest


# --------------------------------------------------
# suggest_target
# --------------------------------------------------


def _suggest(df):
    return suggest_target(profile_dataset(df))[0]


def test_suggests_synthetic_sales_subscribed():
    assert _suggest(pd.read_csv(REPO_ROOT / "synthetic_sales.csv")) == "subscribed"


def test_exact_outcome_name_beats_weaker_words(rng):
    # Bank Marketing shape: a "default" flag and the real outcome "y".
    n = 100
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 70, n),
            "default": rng.choice(["yes", "no"], n),
            "y": rng.choice(["yes", "no"], n),
        }
    )
    column, reason = suggest_target(profile_dataset(df))
    assert column == "y"
    assert "outcome" in reason


def test_falls_back_to_trailing_boolean(rng):
    n = 100
    df = pd.DataFrame({"x": rng.normal(size=n), "flag": rng.choice([True, False], n)})
    assert _suggest(df) == "flag"


def test_never_guesses_numeric_or_unnamed_targets(rng):
    n = 100
    df = pd.DataFrame(
        {"species": rng.choice(list("ABC"), n), "total": rng.normal(size=n)}
    )
    column, reason = suggest_target(profile_dataset(df))
    assert column is None
    assert reason


REAL_WORLD_SUGGESTIONS = [
    # Penguins' species and Bike Sharing's cnt are documented targets, but
    # nothing in their names or types says "outcome", so none is suggested.
    (PENGUINS, None),
    (BIKE_SHARING, None),
    (TITANIC, "survived"),
    (BANK_MARKETING, "y"),
]


@pytest.mark.parametrize(
    ("spec", "expected"),
    REAL_WORLD_SUGGESTIONS,
    ids=[spec.name for spec, _ in REAL_WORLD_SUGGESTIONS],
)
def test_suggest_target_on_real_world_datasets(spec, expected):
    path = FIXTURES / spec.filename
    if not path.exists():
        path = DATASET_DIR / spec.filename
    if not path.exists():
        pytest.skip(
            f"{spec.filename} not downloaded "
            "(python -m evaluation.real_world.download_datasets)"
        )
    assert _suggest(pd.read_csv(path)) == expected


def test_penguins_target_mode_regression(monkeypatch):
    """With target=species, all four Penguins insights stay in the top 5."""

    monkeypatch.setattr(real_world, "DATASET_DIR", FIXTURES)

    summary, *_ = real_world.evaluate_dataset(PENGUINS, target=PENGUINS.target)

    assert summary["recall_at_5"] == pytest.approx(1.0)
    assert np.isfinite(summary["mean_reciprocal_expected_rank"])
