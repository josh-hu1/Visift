import pytest

from visift.scoring.engine import (
    candidate_signature,
    diversify_recommendations,
    recommendation_strength,
)


def test_candidate_signature_ignores_orientation():
    a = {"chart": "scatter", "x": "rating", "y": "revenue"}
    b = {"chart": "box", "x": "revenue", "y": "rating"}
    assert candidate_signature(a) == candidate_signature(b) == ("rating", "revenue")


def test_diversify_collapses_same_variables_into_alternatives():
    ranked = [
        {"chart": "box", "x": "region", "y": "revenue", "score": 80.0},
        {"chart": "scatter", "x": "units", "y": "revenue", "score": 70.0},
        {
            "chart": "bar",
            "x": "region",
            "y": "revenue",
            "aggregation": "mean",
            "score": 60.0,
        },
        {"chart": "histogram", "x": "revenue", "score": 50.0},
    ]

    result = diversify_recommendations(ranked)

    assert [(r["chart"], r["x"]) for r in result] == [
        ("box", "region"),
        ("scatter", "units"),
        ("histogram", "revenue"),
    ]
    alternatives = result[0]["alternative_charts"]
    assert len(alternatives) == 1
    assert alternatives[0]["chart"] == "bar"
    assert alternatives[0]["score"] == 60.0
    assert result[1]["alternative_charts"] == []


def test_diversify_does_not_mutate_input():
    ranked = [
        {"chart": "box", "x": "a", "y": "b", "score": 2.0},
        {"chart": "bar", "x": "a", "y": "b", "aggregation": "mean", "score": 1.0},
    ]
    diversify_recommendations(ranked)
    assert "alternative_charts" not in ranked[0]


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (100, "Excellent"),
        (80, "Excellent"),
        (79.9, "Strong"),
        (65, "Strong"),
        (50, "Moderate"),
        (35, "Weak"),
        (34.9, "Very weak"),
        (0, "Very weak"),
    ],
)
def test_recommendation_strength_bands(score, label):
    assert recommendation_strength(score) == label
