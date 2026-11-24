import pandas as pd
import pytest

from evaluation.real_world import benchmark as real_world
from evaluation.real_world.expectations import PENGUINS
from tests.conftest import FIXTURES, REPO_ROOT
from visift.candidate_generator import generate_candidates
from visift.profiler import profile_dataset
from visift.scoring.engine import diversify_recommendations, score_all_candidates
from visift.visualization.renderer import render_candidate


def test_synthetic_sales_pipeline():
    df = pd.read_csv(REPO_ROOT / "synthetic_sales.csv")
    profile = profile_dataset(df)
    candidates = generate_candidates(df, profile)
    scored = score_all_candidates(df, candidates, profile)
    recommendations = diversify_recommendations(scored)

    assert len(candidates) > 20
    scores = [r["score"] for r in scored]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores)

    top = recommendations[0]
    assert top["score"] > 60
    assert top["reasons"]

    for recommendation in recommendations[:5]:
        figure = render_candidate(df, recommendation)
        assert figure is not None
        assert len(figure.data) > 0


def test_penguins_regression(monkeypatch):
    """Recall@10 on Palmer Penguins must not drop below the current 100%."""

    assert (FIXTURES / PENGUINS.filename).exists()
    monkeypatch.setattr(real_world, "DATASET_DIR", FIXTURES)

    summary, expectation_df, *_ = real_world.evaluate_dataset(PENGUINS)

    assert summary["recall_at_10"] == pytest.approx(1.0), expectation_df[
        ["expected_insight", "rank"]
    ]
    assert summary["semantic_accuracy"] == pytest.approx(1.0)


FAMILY_CANDIDATES = [
    {"chart": "scatter", "x": "x", "y": "linear"},
    {"chart": "histogram", "x": "bimodal"},
    {"chart": "box", "x": "segment", "y": "by_segment"},
    {"chart": "bar", "x": "segment", "y": "by_segment", "aggregation": "mean"},
    {"chart": "bar", "x": "segment", "y": "converted", "aggregation": "mean"},
    {"chart": "bar", "x": "skewed_category", "aggregation": "count"},
    {"chart": "line", "x": "date", "y": "trend"},
    {"chart": "line", "x": "date", "y": "seasonal"},
]


@pytest.mark.parametrize(
    "candidate",
    FAMILY_CANDIDATES,
    ids=[f"{c['chart']}-{c.get('y', c['x'])}" for c in FAMILY_CANDIDATES],
)
def test_every_chart_family_renders_and_explains(candidate):
    from tests.test_scorers import _base_frame
    from visift.scoring.engine import (
        chart_display_name,
        score_candidate,
        summary_reasons,
        visualization_title,
    )

    df = _base_frame(7)
    scored = score_candidate(df, candidate, profile_dataset(df))

    reasons = summary_reasons(scored)
    assert reasons and all(isinstance(r, str) and r for r in reasons)
    assert visualization_title(scored)
    assert chart_display_name(scored["chart"])

    figure = render_candidate(df, scored)
    assert len(figure.data) > 0
