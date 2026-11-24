"""
Synthetic validation of Simpson's paradox detection (visift.paradox).

    python -m evaluation.simpson_calibration

Planted reversals: k groups whose means move along a negative line in
(x, y) while x and y are positively correlated inside every group, across
group counts, within-group correlation and between-group separation.

Nulls (nothing should be flagged):
- independent_groups: x and y correlated, grouping unrelated to both
- shift_same_sign: groups shift x and y along the same direction as the
  within-group relationship (overall and within agree)
- shift_x_only: groups shift x only, y unrelated
- pure_noise: no relationships, random grouping
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from visift.paradox import REVERSAL_MIN_ABS, _within_rank_correlation, find_paradoxes
from visift.profiler import profile_dataset

RESULTS_DIR = Path("evaluation/results/simpson_calibration_v1")

SEEDS = range(20)
ROWS_PER_GROUP = 60
GROUP_COUNTS = (2, 3, 5, 8)
WITHIN_RHO = (0.3, 0.6)
SEPARATION = (1.0, 2.0, 3.0)


def _groups(k: int, n: int) -> np.ndarray:
    return np.repeat([f"g{j}" for j in range(k)], n)


def planted_reversal(k: int, rho: float, separation: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = _groups(k, ROWS_PER_GROUP)
    index = np.repeat(np.arange(k), ROWS_PER_GROUP)
    mean_x = index * separation
    mean_y = -index * separation
    dx = rng.normal(size=len(index))
    dy = rho * dx + np.sqrt(1 - rho**2) * rng.normal(size=len(index))
    return pd.DataFrame({"x": mean_x + dx, "y": mean_y + dy, "group": labels})


def null_dataset(kind: str, k: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = k * ROWS_PER_GROUP
    labels = _groups(k, ROWS_PER_GROUP)
    index = np.repeat(np.arange(k), ROWS_PER_GROUP)
    e1, e2 = rng.normal(size=n), rng.normal(size=n)

    if kind == "independent_groups":
        x, y = e1, 0.5 * e1 + np.sqrt(0.75) * e2
        labels = rng.permutation(labels)
    elif kind == "shift_same_sign":
        x = index * 2.0 + e1
        y = index * 2.0 + 0.5 * e1 + np.sqrt(0.75) * e2
    elif kind == "shift_x_only":
        x, y = index * 2.0 + e1, e2
    elif kind == "pure_noise":
        x, y = e1, e2
        labels = rng.permutation(labels)
    else:
        raise ValueError(kind)

    return pd.DataFrame({"x": x, "y": y, "group": labels})


NULL_KINDS = ("independent_groups", "shift_same_sign", "shift_x_only", "pure_noise")


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    planted_rows = []

    for k, rho, separation in itertools.product(GROUP_COUNTS, WITHIN_RHO, SEPARATION):
        for seed in SEEDS:
            df = planted_reversal(k, rho, separation, seed)
            flags = find_paradoxes(df, profile_dataset(df))
            overall = float(stats.spearmanr(df["x"], df["y"])[0])
            within = _within_rank_correlation(df, "x", "y", "group")
            planted_rows.append(
                {
                    "groups": k,
                    "within_rho": rho,
                    "separation": separation,
                    "seed": seed,
                    "sample_overall": overall,
                    "sample_within": within,
                    # The sample itself shows a reversal of the size the rule
                    # is meant to catch (significance aside).
                    "reversal_in_sample": overall <= -REVERSAL_MIN_ABS
                    and within >= REVERSAL_MIN_ABS,
                    "detected": any(f["kind"] == "reversal" for f in flags),
                    "masked": any(f["kind"] == "masked" for f in flags),
                }
            )

    null_rows = []

    for kind, k in itertools.product(NULL_KINDS, GROUP_COUNTS):
        for seed in SEEDS:
            df = null_dataset(kind, k, seed)
            flags = find_paradoxes(df, profile_dataset(df))
            null_rows.append(
                {
                    "null": kind,
                    "groups": k,
                    "seed": seed,
                    "flagged": bool(flags),
                    "kinds": ",".join(f["kind"] for f in flags),
                }
            )

    return pd.DataFrame(planted_rows), pd.DataFrame(null_rows)


def main() -> None:
    planted, nulls = run()

    detection = (
        planted.groupby(["groups", "within_rho", "separation"])["detected"]
        .mean()
        .unstack("separation")
    )
    false_positives = nulls.groupby(["null", "groups"])["flagged"].mean().unstack()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    planted.to_csv(RESULTS_DIR / "planted.csv", index=False)
    nulls.to_csv(RESULTS_DIR / "nulls.csv", index=False)

    print("DETECTION RATE (rows: groups x within-group rho; columns: separation)")
    print(detection.to_string(float_format=lambda v: f"{v:.0%}"))
    print()
    print("FALSE-POSITIVE RATE (rows: null kind; columns: groups)")
    print(false_positives.to_string(float_format=lambda v: f"{v:.0%}"))
    print()
    in_sample = planted[planted["reversal_in_sample"]]
    print(
        f"Planted datasets whose sample shows a reversal of at least "
        f"{REVERSAL_MIN_ABS}: {len(in_sample)}; detected {in_sample['detected'].mean():.1%}"
    )
    print(
        f"Planted datasets without such a sample reversal: "
        f"{len(planted) - len(in_sample)}; flagged as reversal "
        f"{planted[~planted['reversal_in_sample']]['detected'].mean():.1%}, "
        f"as masked {planted[~planted['reversal_in_sample']]['masked'].mean():.1%}"
    )
    print(
        f"Overall: detection {planted['detected'].mean():.1%} of "
        f"{len(planted)} planted; false positives {nulls['flagged'].mean():.1%} "
        f"of {len(nulls)} nulls"
    )


if __name__ == "__main__":
    main()
