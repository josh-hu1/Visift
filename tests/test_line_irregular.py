"""Seasonality on unevenly spaced time steps (evaluation/irregular_seasonality.py)."""

import numpy as np
import pandas as pd

from evaluation.temporal_calibration import (
    irregular_seasonality_scenario,
    score_dataframe,
)
from visift.scoring.line import evenly_spaced_residuals


def test_seasonality_is_found_on_uneven_time_steps():
    scenario = irregular_seasonality_scenario(seed=0, jitter=0.30)
    metrics = score_dataframe(scenario.dataframe, scenario.x, scenario.y)
    assert not metrics["seasonality_regular_sampling"]
    assert metrics["dominant_pattern"] == "seasonality"
    assert metrics["raw_seasonality_signal"] >= 50
    assert 20 <= metrics["seasonal_period"] <= 30  # planted period: 25


def test_evenly_spaced_series_take_the_old_path():
    scenario = irregular_seasonality_scenario(seed=0, jitter=0.0)
    metrics = score_dataframe(scenario.dataframe, scenario.x, scenario.y)
    assert metrics["seasonality_regular_sampling"]
    assert metrics["raw_seasonality_signal"] > 0


def test_grid_interpolates_between_observations():
    trend_data = pd.DataFrame({"_time_numeric": [0.0, 1.0, 3.0, 4.0]})
    grid = evenly_spaced_residuals(trend_data, np.array([0.0, 1.0, 3.0, 2.0]), 1.0, 4)
    np.testing.assert_allclose(grid, [0.0, 1.0, 2.0, 3.0, 2.0])


def test_grid_that_would_be_mostly_gaps_is_refused():
    # 10 observations spanning 39 median steps: 40 grid points > 2 x 10.
    times = np.concatenate([np.arange(5.0), 35.0 + np.arange(5.0)])
    trend_data = pd.DataFrame({"_time_numeric": times})
    assert evenly_spaced_residuals(trend_data, np.zeros(10), 1.0, 10) is None
