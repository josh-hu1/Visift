from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visift.profiler import profile_dataset
from visift.scoring.engine import score_candidate

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def score():
    """Score one candidate dict against a DataFrame, profiling it first."""

    def _score(df: pd.DataFrame, chart: str, x: str, y=None, aggregation=None):
        candidate = {"chart": chart, "x": x}
        if y is not None:
            candidate["y"] = y
        if aggregation is not None:
            candidate["aggregation"] = aggregation
        return score_candidate(df, candidate, profile_dataset(df))

    return _score
