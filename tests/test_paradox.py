import json

import numpy as np
import pandas as pd
import pytest

from evaluation.simpson_calibration import NULL_KINDS, null_dataset, planted_reversal
from tests.conftest import FIXTURES
from visift.paradox import MIN_ROWS_PER_GROUP, find_paradoxes
from visift.pipeline import analyze
from visift.profiler import profile_dataset
from visift.redundancy import find_redundancies
from visift.scoring.engine import diversify_recommendations
from visift.visualization.renderer import render_candidate


def _find(df):
    profile = profile_dataset(df)
    return find_paradoxes(df, profile, find_redundancies(df, profile))


@pytest.mark.parametrize("groups", [2, 3, 5, 8])
def test_planted_reversal_is_detected(groups):
    df = planted_reversal(groups, rho=0.6, separation=2.0, seed=0)
    (finding,) = _find(df)

    assert finding["kind"] == "reversal"
    assert (finding["x"], finding["y"], finding["group"]) == ("x", "y", "group")
    assert finding["overall"] < 0 < finding["within"]
    assert len(finding["groups"]) == groups
    assert all(g["rho"] > 0 for g in finding["groups"])
    assert "within each group" in finding["sentence"]


@pytest.mark.parametrize("kind", NULL_KINDS)
@pytest.mark.parametrize("seed", range(3))
def test_nulls_are_not_flagged(kind, seed):
    assert _find(null_dataset(kind, 3, seed)) == []


def _weak_reversal(noise_columns):
    rng = np.random.default_rng(7)
    n_per, rho, separation = 25, 0.45, 0.9
    index = np.repeat(np.arange(3), n_per)
    dx = rng.normal(size=index.size)
    dy = rho * dx + np.sqrt(1 - rho**2) * rng.normal(size=index.size)
    df = pd.DataFrame(
        {
            "x": index * separation + dx,
            "y": -index * separation + dy,
            "group": np.repeat(["a", "b", "c"], n_per),
        }
    )
    for j in range(noise_columns):
        df[f"noise_{j}"] = rng.normal(size=index.size)
    return df


def test_multiple_testing_correction_scales_with_checks():
    # Significant on its own (one check)...
    alone = _find(_weak_reversal(0))
    assert [f["kind"] for f in alone] == ["reversal"]

    # ...but not once 12 unrelated columns add dozens more checks.
    crowded = _find(_weak_reversal(12))
    assert not any(f["x"] == "x" and f["y"] == "y" for f in crowded)


def test_small_groups_are_not_used():
    # A perfect reversal, but each group has fewer rows than the minimum.
    df = planted_reversal(3, rho=0.9, separation=3.0, seed=1)
    small = df.groupby("group").head(MIN_ROWS_PER_GROUP - 1).reset_index(drop=True)
    assert _find(small) == []
    assert _find(df.groupby("group").head(MIN_ROWS_PER_GROUP).reset_index(drop=True))


def test_redundant_pairs_are_skipped():
    df = planted_reversal(3, rho=0.6, separation=2.0, seed=2)
    # x_copy is a near-duplicate of x, so x vs x_copy must never be checked
    # and the planted x-y reversal must still be found.
    df["x_copy"] = df["x"] + np.random.default_rng(0).normal(0, 1e-3, len(df))
    pairs = {(f["x"], f["y"]) for f in _find(df)}
    assert ("x", "x_copy") not in pairs
    assert ("x", "y") in pairs


def test_grouping_that_strongly_explains_x_is_still_checked():
    # Well-separated groups explain >= 95% of x's variance (the 2.2 redundancy
    # rule reports this as near-deterministic). That is exactly when a
    # reversal is clearest, so the grouping must not be skipped.
    df = planted_reversal(8, rho=0.6, separation=2.0, seed=0)
    profile = profile_dataset(df)
    redundancies = find_redundancies(df, profile)
    assert any(r["kind"] == "determined_numeric" for r in redundancies)
    assert [f["group"] for f in find_paradoxes(df, profile, redundancies)] == ["group"]


def test_grouped_scatter_renders():
    df = planted_reversal(3, rho=0.6, separation=2.0, seed=0)
    figure = render_candidate(
        df, {"chart": "scatter", "x": "x", "y": "y", "group": "group"}
    )
    names = [trace.name for trace in figure.data]
    assert {"g0", "g1", "g2"} <= set(names)
    assert "All rows (overall trend)" in names


def test_penguins_findings_and_badges():
    df = pd.read_csv(FIXTURES / "penguins.csv")
    analysis = analyze(df)

    found = {(p["x"], p["y"], p["group"]) for p in analysis["paradoxes"]}
    assert ("bill_length_mm", "bill_depth_mm", "species") in found
    assert all(p["kind"] == "reversal" for p in analysis["paradoxes"])

    badged = [r for r in analysis["recommendations"] if r.get("paradox")]
    assert badged and all(r["chart"] == "scatter" for r in badged)
    for recommendation in badged:
        grouped = [a for a in recommendation["alternative_charts"] if "group" in a]
        assert len(grouped) == 1 and "score" not in grouped[0]


def test_findings_do_not_change_scores_or_ranks():
    df = pd.read_csv(FIXTURES / "penguins.csv")
    analysis = analyze(df)

    def strip(recommendation):
        clean = {k: v for k, v in recommendation.items() if k != "paradox"}
        clean["alternative_charts"] = [
            a for a in recommendation["alternative_charts"] if "group" not in a
        ]
        return clean

    stripped = [strip(r) for r in analysis["recommendations"]]

    # Same ranking as the pipeline without the paradox step: diversify the
    # scored charts that were not set aside in the dataset notes.
    noted = {json.dumps(c, default=str) for c in analysis["noted_charts"]}
    ranked = [c for c in analysis["scored"] if json.dumps(c, default=str) not in noted]
    plain = diversify_recommendations(ranked, analysis["aliases"])

    assert json.dumps(stripped, default=str) == json.dumps(plain, default=str)


def test_works_with_a_target():
    df = pd.read_csv(FIXTURES / "penguins.csv")
    with_target = analyze(df, target="species")
    without = analyze(df)
    assert with_target["paradoxes"] == without["paradoxes"]
