"""
Seasonality on irregular time steps: pre-registered evaluation.

This file was written and committed before the change it evaluates was
implemented or run. Run it with ``python -m evaluation.irregular_seasonality``.

Problem
-------
The line scorer's seasonality detector (the strongest FFT frequency,
confirmed by autocorrelation at its period) needs evenly spaced points.
When the median absolute deviation of the time steps is more than 10% of
their median step, it returns 0. On the temporal calibration suite's
irregular grids, a strong sinusoid (amplitude 1.5, unit noise, period 25,
500 points) has a median seasonality signal of about 68 at jitter 0.08 or
less, but 0 at jitter 0.15 (40% of seeds still pass the regularity check)
and 0 at jitter 0.30 (none pass). The line chart then scores at its floor.

Change
------
When the time steps are irregular, the detrended residuals are linearly
interpolated onto an evenly spaced grid from the first to the last time
point, with a step equal to the median time step (the median the
regularity check already computes). The existing detector then runs on the
grid unchanged: the same thresholds, period limits and autocorrelation
check. Its sample-size reliability still counts real observations, not grid
points. If the grid would need more than twice as many points as there are
observations (so more than half of it would be drawn across gaps), the
detector returns 0 as it does now. That factor of 2 is the only new
constant. Evenly spaced series never take this path, so their scores cannot
change.

Acceptance criteria (all must hold)
-----------------------------------
1. Detection. On the calibration suite's irregular seasonal series (25
   seeds at each of jitter 0.15 and 0.30), the median raw seasonality
   signal is at least 50, and seasonality is the line chart's dominant
   pattern in at least 80% of series, at each jitter level.
2. No new false positives. On series without seasonality placed on
   irregular grids (time steps uniform on [0.5, 2], as in the suite's
   irregular null; 100 seeds at each of n = 50, 100, 200 and 500), the share
   with a nonzero raw seasonality signal is at most 1 percentage point above
   the share for the same values on an evenly spaced grid. This is checked
   for independent Gaussian noise and for a Gaussian random walk (red
   noise, a common source of spurious spectral peaks).
3. Gaps. The same comparison and tolerance hold for Gaussian noise on an
   irregular grid that also has three gaps of 30 median steps each (300
   points, 100 seeds).
4. Nothing else moves. Every score on the bundled samples and the wide
   synthetic tables is byte-identical, and the evenly spaced series in this
   evaluation score exactly as they did before.

If a criterion fails, the change is reported as failed and is not tuned to
pass.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from evaluation.temporal_calibration import (
    irregular_seasonality_scenario,
    score_dataframe,
)

SEASONAL_JITTERS = (0.15, 0.30)
SEASONAL_SEEDS = range(25)
NULL_SIZES = (50, 100, 200, 500)
NULL_SEEDS = range(100)
GAP_SIZE = 300
GAP_LENGTH = 30
GAP_COUNT = 3
TOLERANCE = 0.01


def _frame(time_values: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"time": time_values, "value": values})


def _raw_seasonality(time_values: np.ndarray, values: np.ndarray) -> float:
    return float(
        score_dataframe(_frame(time_values, values), "time", "value")[
            "raw_seasonality_signal"
        ]
    )


def _null_values(kind: str, rng: np.random.Generator, n: int) -> np.ndarray:
    noise = rng.normal(0, 1, n)
    return np.cumsum(noise) if kind == "random_walk" else noise


def detection() -> pd.DataFrame:
    rows = []
    for jitter in SEASONAL_JITTERS:
        for seed in SEASONAL_SEEDS:
            scenario = irregular_seasonality_scenario(seed=seed, jitter=jitter)
            metrics = score_dataframe(scenario.dataframe, scenario.x, scenario.y)
            rows.append(
                {
                    "jitter": jitter,
                    "seed": seed,
                    "raw_seasonality": metrics["raw_seasonality_signal"],
                    "dominant": metrics["dominant_pattern"],
                    "score": metrics["score"],
                }
            )
    return pd.DataFrame(rows)


def false_positives() -> pd.DataFrame:
    rows = []
    for kind in ("white_noise", "random_walk"):
        for n in NULL_SIZES:
            for seed in NULL_SEEDS:
                rng = np.random.default_rng([seed, n, int(kind == "random_walk")])
                values = _null_values(kind, rng, n)
                irregular = np.cumsum(rng.uniform(0.5, 2.0, n))
                rows.append(
                    {
                        "family": kind,
                        "n": n,
                        "seed": seed,
                        "irregular": _raw_seasonality(irregular, values),
                        "regular": _raw_seasonality(np.arange(n, dtype=float), values),
                    }
                )
    for seed in NULL_SEEDS:
        rng = np.random.default_rng([seed, GAP_SIZE, 7])
        values = rng.normal(0, 1, GAP_SIZE)
        steps = rng.uniform(0.5, 2.0, GAP_SIZE)
        median_step = float(np.median(steps[1:]))
        for position in rng.choice(np.arange(10, GAP_SIZE - 10), GAP_COUNT, False):
            steps[position] += GAP_LENGTH * median_step
        rows.append(
            {
                "family": "gaps",
                "n": GAP_SIZE,
                "seed": seed,
                "irregular": _raw_seasonality(np.cumsum(steps), values),
                "regular": _raw_seasonality(np.arange(GAP_SIZE, dtype=float), values),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)

    passed = True

    found = detection()
    print("1. Detection on irregular seasonal series")
    for jitter, group in found.groupby("jitter"):
        median = group["raw_seasonality"].median()
        dominant = (group["dominant"] == "seasonality").mean()
        ok = median >= 50 and dominant >= 0.80
        passed &= ok
        print(
            f"   jitter {jitter:.2f}: median raw seasonality {median:.1f} "
            f"(need >= 50), seasonality dominant in {dominant:.0%} "
            f"(need >= 80%), median line score {group['score'].median():.1f}"
            f"  {'PASS' if ok else 'FAIL'}"
        )

    nulls = false_positives()
    print("2-3. Nonzero seasonality without seasonality: irregular vs even grid")
    for (family, n), group in nulls.groupby(["family", "n"], sort=False):
        irregular = (group["irregular"] > 0).mean()
        regular = (group["regular"] > 0).mean()
        ok = irregular <= regular + TOLERANCE
        passed &= ok
        print(
            f"   {family:12s} n={n:<4d} irregular {irregular:5.1%}  "
            f"even {regular:5.1%}  {'PASS' if ok else 'FAIL'}"
        )

    print("4. Checked separately: score dump and app output against main (see the PR).")
    print("ALL PASS" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
