"""
A deliberately simple baseline ranker for the real-world benchmark.

Pre-registered: this definition was committed before the baseline was run
on any dataset, and is not tuned afterwards.

Question it answers: what does Visift's scoring add over sorting charts by
a plain effect size? It sees exactly the same profile and candidate charts
as Visift (same semantic types and candidate generation, including the
target-oriented charts in target mode) and only changes the ordering. It
has no significance testing, no calibration, no redundancy handling and no
diversity step (two charts of the same variables are both listed).

Effect size per chart, one standard untuned measure each, all on a
correlation scale in [0, 1] so different chart types are comparable:

- scatter: |Spearman rho| of x and y
- line: |Spearman rho| of the value against time order
- bar (mean) and box, numeric value: correlation ratio eta = sqrt(eta^2).
  eta rather than eta^2: eta^2 is a variance-explained measure (the analog
  of rho^2), so it would rank grouped charts systematically below scatters
  of equal strength. eta is the correlation-scale analog of |rho|.
- binary-rate bar (mean of a boolean outcome by a category): bias-corrected
  Cramer's V of category vs outcome
- single-variable charts (histograms, count bars): 0, ranked after every
  two-variable chart. A correlation-sorting analyst has no measure of an
  "interesting distribution", and inventing one (e.g. skewness) would be a
  scoring model rather than a baseline.

Ties (including all single-variable charts) keep candidate-generation
order, so the ranking is deterministic.

Target mode ("an analyst sorts by correlation with the outcome"): the
target's two-variable charts come first, ranked by the same effect size,
followed by every remaining chart in the no-target order. This mirrors how
Visift's target mode is scored (drivers first, then everything else).

Amendment (decided after seeing results; the pre-registered original is
commit e392ea7): target mode originally kept ONLY the charts involving the
target. Three of the 20 expected insights do not involve the target, so
they were unreachable for the baseline by construction while Visift could
still find them further down its combined list. Listing the remaining
charts after the drivers can only raise the baseline's numbers. Nothing
else about the baseline changed.
"""

from typing import Any

import numpy as np
import pandas as pd

from visift.candidate_generator import generate_candidates
from visift.profiler import profile_dataset
from visift.redundancy import bias_corrected_cramers_v
from visift.types import Candidate, DatasetProfile
from visift.value_normalization import coerce_numeric_or_boolean


def _as_number(series: pd.Series) -> pd.Series:
    """Numbers, booleans as 0/1, and dates as nanoseconds since the epoch."""

    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64").astype(float).where(series.notna())

    converted = pd.to_numeric(coerce_numeric_or_boolean(series), errors="coerce")

    if converted.notna().any():
        return converted

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.astype("int64").astype(float).where(parsed.notna())


def abs_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": _as_number(x), "y": _as_number(y)}).dropna()

    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return 0.0

    rho = pair["x"].rank().corr(pair["y"].rank())
    return float(abs(rho)) if np.isfinite(rho) else 0.0


def correlation_ratio(category: pd.Series, value: pd.Series) -> float:
    frame = pd.DataFrame({"g": category, "v": _as_number(value)}).dropna()

    if len(frame) < 3 or frame["g"].nunique() < 2:
        return 0.0

    total = float(((frame["v"] - frame["v"].mean()) ** 2).sum())

    if total == 0:
        return 0.0

    means = frame.groupby("g")["v"].transform("mean")
    between = float(((means - frame["v"].mean()) ** 2).sum())

    return float(np.sqrt(between / total))


def effect_size(df: pd.DataFrame, candidate: Candidate, types: dict[str, str]) -> float:
    chart = candidate["chart"]
    x = candidate["x"]
    y = candidate.get("y")

    if y is None:
        return 0.0

    if chart in ("scatter", "line"):
        return abs_spearman(df[x], df[y])

    if chart == "bar" and types.get(y) == "boolean":
        pair = df[[x, y]].dropna()
        return bias_corrected_cramers_v(pair[x], pair[y]) if len(pair) else 0.0

    if chart in ("bar", "box"):
        return correlation_ratio(df[x], df[y])

    return 0.0


def baseline_ranking(
    df: pd.DataFrame, profile: DatasetProfile, target: str | None = None
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """(candidates, ranked charts with a 0-100 'score' = 100 x effect size)."""

    types = {c["name"]: c["semantic_type"] for c in profile["column_profiles"]}
    candidates = generate_candidates(df, profile, target=target)

    ranked: list[dict[str, Any]] = []

    for candidate in candidates:
        score = 100 * effect_size(df, candidate, types)
        ranked.append(
            {
                **candidate,
                "score": score,
                "components": {"signal": score},
                "reasons": [],
                "alternative_charts": [],
            }
        )

    # Stable sort: ties keep candidate-generation order.
    ranked.sort(key=lambda item: -item["score"])

    if target is not None:
        # Drivers of the target first (same effect-size order), then the rest.
        def is_driver(chart: dict[str, Any]) -> bool:
            y = chart.get("y")
            return y is not None and target in (chart["x"], y)

        ranked = [c for c in ranked if is_driver(c)] + [
            c for c in ranked if not is_driver(c)
        ]

    return candidates, ranked


def baseline_analysis(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """The keys the real-world benchmark reads from visift.pipeline.analyze."""

    profile = profile_dataset(df)
    candidates, ranked = baseline_ranking(df, profile, target)

    return {
        "profile": profile,
        "candidates": candidates,
        "recommendations": ranked,
        "noted_charts": [],
    }
