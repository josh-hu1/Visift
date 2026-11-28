"""
Binned-rate bar calibration: false positives on pure noise and detection of
planted U-shaped and monotonic effects.

    python -m evaluation.binned_rate_calibration

Pre-registered in evaluation/binned_rate_preregistration.md (criteria 1-2):

1. Null: numeric x independent of a binary y, base rate in (10%, 30%, 50%),
   n in (100, 300, 1000, 5000), 500 seeds per cell. Pass if score >= 50 in
   at most 1% of every cell and never >= 65.
2. Power (reported, not gated): x ~ Uniform(0, 1); P(y=1) = base + lift *
   shape(x) with shape U(x) = (2x - 1)^2 or linear(x) = x, lift in
   (0.10, 0.20, 0.30), base 0.10, n in (300, 1000, 5000), 100 seeds per
   cell. Reports the share scoring >= 50.

Outputs evaluation/results/binned_rate_calibration_v1/{null,power}.csv.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from visift.scoring.bar import score_bar_chart
from visift.types import DatasetProfile

OUT = Path(__file__).resolve().parent / "results" / "binned_rate_calibration_v1"
CANDIDATE = {"chart": "bar", "x": "x", "y": "y", "aggregation": "binned_rate"}

# The binned-rate branch fixes its own semantic fit; the profile is only
# consulted for the x column's type, which this branch does not use.
PROFILE: DatasetProfile = {"rows": 0, "columns": 2, "column_profiles": []}  # type: ignore[typeddict-item]


def _score(x: np.ndarray, y: np.ndarray) -> float:
    df = pd.DataFrame({"x": x, "y": y.astype(int)})
    return float(score_bar_chart(df, CANDIDATE, PROFILE)["score"])  # type: ignore[arg-type]


def null_grid(seeds: int = 500) -> pd.DataFrame:
    rows = []
    for base, n in itertools.product((0.10, 0.30, 0.50), (100, 300, 1000, 5000)):
        scores = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            x = rng.normal(40, 10, n).round(1)
            y = rng.random(n) < base
            scores.append(_score(x, y))
        s = np.array(scores)
        rows.append(
            {
                "base_rate": base,
                "n": n,
                "seeds": seeds,
                "share_ge_50": float((s >= 50).mean()),
                "share_ge_65": float((s >= 65).mean()),
                "max": float(s.max()),
                "p95": float(np.percentile(s, 95)),
            }
        )
    return pd.DataFrame(rows)


def power_grid(seeds: int = 100) -> pd.DataFrame:
    shapes = {"u_shape": lambda x: (2 * x - 1) ** 2, "linear": lambda x: x}
    rows = []
    for (shape, f), lift, n in itertools.product(
        shapes.items(), (0.10, 0.20, 0.30), (300, 1000, 5000)
    ):
        scores = []
        for seed in range(seeds):
            rng = np.random.default_rng(10_000 + seed)
            x = rng.random(n)
            y = rng.random(n) < 0.10 + lift * f(x)
            scores.append(_score((x * 100).round(2), y))
        s = np.array(scores)
        rows.append(
            {
                "shape": shape,
                "lift": lift,
                "n": n,
                "seeds": seeds,
                "share_ge_50": float((s >= 50).mean()),
                "median": float(np.median(s)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    null = null_grid()
    null.to_csv(OUT / "null.csv", index=False)
    print("Null (pure noise):")
    print(null.to_string(index=False))
    passed = (null["share_ge_50"] <= 0.01).all() and (null["share_ge_65"] == 0).all()
    print(
        f"Criterion 1 (<= 1% >= 50 per cell, none >= 65): {'PASS' if passed else 'FAIL'}"
    )

    power = power_grid()
    power.to_csv(OUT / "power.csv", index=False)
    print("\nPower (share scoring >= 50):")
    print(power.to_string(index=False))


if __name__ == "__main__":
    main()
