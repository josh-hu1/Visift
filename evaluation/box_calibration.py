"""
Box-plot scorer calibration: false positives on pure noise and detection of
planted spread differences.

    python -m evaluation.box_calibration

Acceptance criteria for the Phase 2.6 significance gates (pre-registered
with the gates, before running them on this validation data):

1. Null false-positive rate (score >= 50) at most 1% in the test-suite case
   and in every group-count/group-size cell (500 seeds each), and no null
   score >= 65 anywhere.
2. Planted spread differences (IQR ratio 1.5 / 2 / 3, small and large
   groups) still detected; detection may drop only where Brown-Forsythe
   genuinely lacks power. Full table reported.
3. No regressions: hardened benchmark top-1 100% with nulls >= 50 at 0%;
   cross-chart calibration: stronger planted pattern still wins 100%;
   Simpson's paradox calibration unchanged.

Nulls (nothing to find):
- test_frame: the null case from tests/test_scorers.py (segment vs noise in
  _base_frame(seed), 4 groups x ~100 rows), 500 seeds
- grid: k groups x n rows per group of pure noise, k in (2, 3, 5, 8),
  n in (20, 100), 500 seeds per cell

Planted spread differences (same median, one group's IQR multiplied by r),
r in (1.5, 2, 3), k in (2, 4), n in (20, 100), 100 seeds per cell.

For every score >= 50 the script records which component of
distribution_signal_score drove the max(): location, dispersion or outlier.
"""

import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from visift.profiler import profile_dataset
from visift.scoring.box import score_box_chart

RESULTS_DIR = Path("evaluation/results/box_calibration_v1")

NULL_SEEDS = 500
TEST_FRAME_SEEDS = 500
PLANTED_SEEDS = 100
GROUP_COUNTS = (2, 3, 5, 8)
GROUP_SIZES = (20, 100)
SPREAD_RATIOS = (1.5, 2.0, 3.0)
PLANTED_GROUP_COUNTS = (2, 4)

COMPONENTS = {
    "location": "location_score",
    "dispersion": "dispersion_score",
    "outlier": "outlier_score",
}


def score(df: pd.DataFrame, x: str, y: str) -> dict:
    result = score_box_chart(df, {"chart": "box", "x": x, "y": y}, profile_dataset(df))
    stats = result["statistics"]
    driver = max(COMPONENTS, key=lambda name: stats[COMPONENTS[name]])
    return {"score": result["score"], "driver": driver, **stats}


def null_frame(k: int, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{j}" for j in range(k)], n)
    return pd.DataFrame({"segment": groups, "value": rng.normal(size=k * n)})


def planted_spread(k: int, n: int, ratio: float, seed: int) -> pd.DataFrame:
    """Same center everywhere; the last group's spread is `ratio` times wider."""

    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{j}" for j in range(k)], n)
    scale = np.where(groups == f"g{k - 1}", ratio, 1.0)
    return pd.DataFrame({"segment": groups, "value": rng.normal(size=k * n) * scale})


def run() -> dict[str, pd.DataFrame]:
    from tests.test_scorers import _base_frame

    test_frame = pd.DataFrame(
        [
            {"seed": seed, **score(_base_frame(seed), "segment", "noise")}
            for seed in range(TEST_FRAME_SEEDS)
        ]
    )

    grid = pd.DataFrame(
        [
            {
                "groups": k,
                "rows_per_group": n,
                "seed": seed,
                **score(null_frame(k, n, seed), "segment", "value"),
            }
            for k, n in itertools.product(GROUP_COUNTS, GROUP_SIZES)
            for seed in range(NULL_SEEDS)
        ]
    )

    planted = pd.DataFrame(
        [
            {
                "groups": k,
                "rows_per_group": n,
                "ratio": ratio,
                "seed": seed,
                **score(planted_spread(k, n, ratio, seed), "segment", "value"),
            }
            for k, n, ratio in itertools.product(
                PLANTED_GROUP_COUNTS, GROUP_SIZES, SPREAD_RATIOS
            )
            for seed in range(PLANTED_SEEDS)
        ]
    )

    return {"test_frame": test_frame, "null_grid": grid, "planted": planted}


def describe_nulls(frame: pd.DataFrame) -> str:
    high = frame[frame["score"] >= 50]
    drivers = dict(Counter(high["driver"]))
    return (
        f">=50: {(frame['score'] >= 50).mean():.1%}  >=65: "
        f"{(frame['score'] >= 65).mean():.1%}  max: {frame['score'].max():.1f}  "
        f"drivers of >=50: {drivers}"
    )


def main() -> None:
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False)

    print("NULL: tests/test_scorers.py box case (500 seeds)")
    print("  " + describe_nulls(results["test_frame"]))

    print(f"\nNULL GRID (pure noise; {NULL_SEEDS} seeds per cell)")
    for (k, n), part in results["null_grid"].groupby(["groups", "rows_per_group"]):
        print(f"  k={k} n/group={n:>3}: " + describe_nulls(part))

    planted = results["planted"]
    print("\nPLANTED SPREAD: share scoring >= 50 (100 seeds per cell)")
    table = planted.assign(detected=planted["score"] >= 50).pivot_table(
        index=["groups", "rows_per_group"],
        columns="ratio",
        values="detected",
        aggfunc="mean",
    )
    print(table.to_string(float_format=lambda v: f"{v:.0%}"))


if __name__ == "__main__":
    main()
