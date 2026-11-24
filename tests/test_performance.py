"""Speed-ups must not change results."""

import json

import numpy as np
import pandas as pd
import pytest

import visift.scoring.scatter as scatter
from benchmarks.perf import wide_dataset
from visift.pipeline import analyze
from visift.scoring.engine import default_n_jobs


def _reference_nonlinear_signal(clean, x, y):
    """The pre-shortcut algorithm: always run every permutation."""

    mi_data = (
        clean.sample(n=scatter.MI_MAX_SAMPLE_SIZE, random_state=scatter.MI_RANDOM_STATE)
        if len(clean) > scatter.MI_MAX_SAMPLE_SIZE
        else clean.copy()
    )
    xv = mi_data[x].to_numpy(dtype=float)
    yv = mi_data[y].to_numpy(dtype=float)

    observed = scatter.symmetric_mutual_information(xv, yv, scatter.MI_RANDOM_STATE)
    rng = np.random.default_rng(scatter.MI_RANDOM_STATE)
    null = np.asarray(
        [
            scatter.symmetric_mutual_information(
                xv, rng.permutation(yv), scatter.MI_RANDOM_STATE + 100 + 2 * i
            )
            for i in range(scatter.MI_PERMUTATIONS)
        ]
    )
    threshold = max(
        scatter.MI_MINIMUM_EFFECT,
        null.mean() + scatter.MI_NULL_STD_MULTIPLIER * null.std(ddof=0),
    )
    excess = max(0, observed - threshold)
    signal = 0 if excess <= 0 else np.sqrt(1 - np.exp(-2 * excess)) * 100
    return float(scatter.clamp(signal)), float(observed), float(excess)


@pytest.mark.parametrize(
    "relationship", ["independent", "weak", "quadratic", "sinusoidal", "linear"]
)
def test_mi_shortcut_matches_full_permutation_test(rng, relationship):
    n = 400
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = {
        "independent": noise,
        "weak": 0.2 * x + noise,
        "quadratic": x**2 + 0.3 * noise,
        "sinusoidal": np.sin(3 * x) + 0.3 * noise,
        "linear": x + 0.5 * noise,
    }[relationship]
    clean = pd.DataFrame({"x": x, "y": y})

    fast = scatter.nonlinear_signal_score(clean, "x", "y")
    signal, observed, excess = _reference_nonlinear_signal(clean, "x", "y")

    assert fast["signal"] == signal
    assert fast["observed_mi"] == observed
    assert fast["excess_mi"] == excess

    if observed <= scatter.MI_MINIMUM_EFFECT:
        # Skipped null statistics are reported as not computed.
        assert np.isnan(fast["null_mean"]) and np.isnan(fast["null_std"])


def _canonical(analysis):
    # NaN-safe structural comparison (NaN != NaN in Python).
    keys = ("scored", "recommendations", "notes", "noted_charts")
    return json.dumps({k: analysis[k] for k in keys}, sort_keys=True, default=str)


@pytest.mark.slow
def test_parallel_scoring_is_identical_to_serial():
    df = wide_dataset(12, 600)
    serial = analyze(df, n_jobs=1)
    parallel = analyze(df, n_jobs=2)

    assert _canonical(parallel) == _canonical(serial)
    assert [c["score"] for c in parallel["scored"]] == [
        c["score"] for c in serial["scored"]
    ]


def test_progress_is_monotonic_and_completes():
    updates = []
    analyze(wide_dataset(10, 300), progress=lambda stage, f: updates.append((stage, f)))

    fractions = [f for _, f in updates]
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert any(stage.startswith("Scoring") for stage, _ in updates)


def test_default_is_serial_for_typical_uploads(monkeypatch):
    monkeypatch.delenv("VISIFT_N_JOBS", raising=False)
    assert default_n_jobs(458) == 1


def test_n_jobs_environment_override(monkeypatch):
    monkeypatch.setenv("VISIFT_N_JOBS", "3")
    assert default_n_jobs(10) == 3
    monkeypatch.setenv("VISIFT_N_JOBS", "0")
    assert default_n_jobs(10_000) == 1
