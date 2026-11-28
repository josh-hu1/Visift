import pandas as pd
import pytest

from tests.conftest import REPO_ROOT
from visift.explain import pattern_factor, score_breakdown
from visift.pipeline import analyze

SAMPLES = {
    "data/samples/penguins.csv": "species",
    "data/samples/titanic.csv": "survived",
    "data/samples/bike_sharing_day.csv": "cnt",
    "data/samples/bank_marketing_sample.csv": "y",
    "synthetic_sales.csv": None,
}

CASES = [
    pytest.param(path, target, id=f"{path.split('/')[-1]}-{target or 'none'}")
    for path, outcome in SAMPLES.items()
    for target in {None, outcome}
]


@pytest.mark.slow
@pytest.mark.parametrize(("path", "target"), CASES)
def test_breakdown_is_exact_for_every_recommendation(path, target):
    df = pd.read_csv(REPO_ROOT / path)
    recommendations = analyze(df, target=target)["recommendations"]
    assert recommendations

    for candidate in recommendations:
        breakdown = score_breakdown(candidate)
        values = [segment["value"] for segment in breakdown["segments"]]

        assert all(value >= 0 for value in values), candidate
        assert sum(values) == pytest.approx(100, abs=1e-9), candidate
        assert breakdown["segments"][-1]["value"] == candidate["score"]

        # The formula itself reproduces the score (components are rounded).
        recomputed = breakdown["suitability"] * breakdown["pattern_factor"]
        assert recomputed == pytest.approx(candidate["score"], abs=0.01), candidate


def test_worked_example_from_the_plan():
    candidate = {
        "score": 82.45,
        "components": {
            "semantic_fit": 100.0,
            "readability": 100.0,
            "data_quality": 98.0,
            "sample_support": 100.0,
            "signal": 73.5,
            "visualization_quality": 99.6,
        },
    }
    breakdown = score_breakdown(candidate)
    assert breakdown["pattern_factor"] == pytest.approx(0.82775)
    assert breakdown["formula"] == (
        "82.5 = 99.6 chart suitability × 0.83 pattern factor (0.35 + 0.65 × 73.5/100)"
    )
    labels = [segment["label"] for segment in breakdown["segments"]]
    assert labels[-2:] == ["Pattern shortfall", "Score"]


def test_pattern_factor_floor_explains_noise_scores():
    # No pattern at all still keeps 35% of suitability: noise sits near 35.
    assert pattern_factor(0) == pytest.approx(0.35)
    assert pattern_factor(100) == pytest.approx(1.0)
