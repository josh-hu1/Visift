"""Significance gates on the box-plot scorer's grouped-distribution signals."""

import numpy as np
import pandas as pd
import pytest

import visift.scoring.box as box
from evaluation.box_calibration import null_frame, planted_spread
from visift.profiler import profile_dataset


def _box(df):
    return box.score_box_chart(
        df, {"chart": "box", "x": "segment", "y": "value"}, profile_dataset(df)
    )


@pytest.mark.parametrize(
    ("p_value", "expected"),
    [(0.0005, 1.0), (0.001, 1.0), (0.01, 0.5), (0.10, 0.0), (0.5, 0.0), (np.nan, 0.0)],
)
def test_reliability_ramp(p_value, expected):
    assert box.significance_reliability(p_value) == pytest.approx(expected)


def test_small_group_noise_is_no_longer_flagged():
    # Before the gates, ~81% of these scored >= 50.
    scores = [_box(null_frame(8, 20, seed))["score"] for seed in range(40)]
    assert max(scores) < 65
    assert np.mean(np.array(scores) >= 50) <= 0.05


def test_clear_spread_difference_is_still_detected():
    detected = [
        _box(planted_spread(4, 100, 3.0, seed))["score"] >= 50 for seed in range(20)
    ]
    assert all(detected)
    stats = _box(planted_spread(4, 100, 3.0, 0))["statistics"]
    assert stats["dispersion_score"] > stats["location_score"]


def test_null_p_values_are_roughly_uniform():
    for test in (box.location_p_value, box.dispersion_p_value, box.outlier_p_value):
        p_values = np.array(
            [test(null_frame(5, 20, seed), "segment", "value") for seed in range(200)]
        )
        assert (p_values < 0.05).mean() <= 0.10, test.__name__


def test_outlier_test_is_valid_permutation_of_values():
    # Same distribution, very different group sizes: outlier rates differ
    # only by chance, so the test should not reject.
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {"segment": ["a"] * 30 + ["b"] * 300, "value": rng.standard_t(3, size=330)}
    )
    assert box.outlier_p_value(df, "segment", "value") > 0.05


def test_outlier_test_is_skipped_when_raw_outlier_signal_is_zero():
    df = null_frame(3, 100, 0)
    assert box.grouped_outlier_statistics(df, "segment", "value")["score"] == 0
    stats = box.distribution_signal_score(df, "segment", "value")
    assert np.isnan(stats["outlier_p"])
    assert stats["outlier_score"] == 0


def test_gates_are_deterministic():
    df = planted_spread(4, 50, 2.0, 7)
    assert box.outlier_p_value(df, "segment", "value") == box.outlier_p_value(
        df, "segment", "value"
    )
