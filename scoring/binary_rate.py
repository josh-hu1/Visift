import numpy as np
from scipy.stats import chi2

from value_normalization import coerce_numeric_or_boolean


ABSOLUTE_SCALE_PP = 60.0
RELATIVE_SCALE = 2.5
MINORITY_EVENTS_PER_GROUP_FULL = 15.0


def clamp(value, minimum=0.0, maximum=100.0):
    return max(
        minimum,
        min(
            float(value),
            maximum
        )
    )


def significance_reliability(p_value):
    """
    Convert a chi-square p-value into a smooth 0-1 reliability.

    p >= 0.10  -> 0
    p <= 0.001 -> 1
    """

    if not np.isfinite(p_value):
        return 0.0

    if p_value >= 0.10:
        return 0.0

    if p_value <= 0.001:
        return 1.0

    value = (
        (
            np.log10(0.10)
            - np.log10(p_value)
        )
        / (
            np.log10(0.10)
            - np.log10(0.001)
        )
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0
        )
    )


def binary_rate_signal_score(
    df,
    x,
    y
):
    """
    Score a categorical predictor against a Boolean outcome.

    The calibrated conservative signal combines:
    - absolute positive-rate separation,
    - relative lift,
    - minority-class event evidence,
    - Pearson chi-square statistical reliability.

    Returns a dictionary so the scorer can expose diagnostics
    without recomputing the relationship.
    """

    clean = df[[x, y]].copy()

    clean[y] = coerce_numeric_or_boolean(
        clean[y]
    )

    clean = clean.dropna(
        subset=[
            x,
            y
        ]
    )

    default = {
        "signal": 0.0,
        "effect_signal": 0.0,
        "absolute_signal": 0.0,
        "relative_signal": 0.0,
        "reliability": 0.0,
        "event_reliability": 0.0,
        "statistical_reliability": 0.0,
        "rate_range_pp": 0.0,
        "max_lift": 1.0,
        "p_value": 1.0,
        "positive_count": 0,
        "negative_count": 0
    }

    if clean.empty:
        return default

    unique_outcomes = set(
        clean[y].unique()
    )

    if not unique_outcomes.issubset(
        {
            0,
            1,
            0.0,
            1.0
        }
    ):
        return default

    grouped = (
        clean
        .groupby(x)[y]
        .agg(
            [
                "count",
                "sum",
                "mean"
            ]
        )
    )

    group_count = len(grouped)

    if group_count <= 1:
        result = dict(default)

        result["positive_count"] = int(
            clean[y].sum()
        )

        result["negative_count"] = int(
            len(clean)
            - clean[y].sum()
        )

        return result

    overall_rate = float(
        clean[y].mean()
    )

    minimum_rate = float(
        grouped["mean"].min()
    )

    maximum_rate = float(
        grouped["mean"].max()
    )

    rate_range_pp = (
        maximum_rate
        - minimum_rate
    ) * 100.0

    absolute_signal = (
        100.0
        * (
            1.0
            - np.exp(
                -rate_range_pp
                / ABSOLUTE_SCALE_PP
            )
        )
    )

    if overall_rate > 0:
        max_lift = (
            maximum_rate
            / overall_rate
        )
    else:
        max_lift = 1.0

    if max_lift > 1.0:
        log2_lift = np.log2(
            max_lift
        )
    else:
        log2_lift = 0.0

    relative_signal = (
        100.0
        * (
            1.0
            - np.exp(
                -log2_lift
                / RELATIVE_SCALE
            )
        )
    )

    effect_signal = max(
        absolute_signal,
        relative_signal
    )

    observations = len(clean)

    positive_count = float(
        clean[y].sum()
    )

    negative_count = (
        observations
        - positive_count
    )

    minority_events = min(
        positive_count,
        negative_count
    )

    minority_events_per_group = (
        minority_events
        / group_count
    )

    event_reliability = np.sqrt(
        min(
            1.0,
            minority_events_per_group
            / MINORITY_EVENTS_PER_GROUP_FULL
        )
    )

    observed = np.column_stack([
        grouped["sum"].to_numpy(
            dtype=float
        ),
        (
            grouped["count"]
            - grouped["sum"]
        ).to_numpy(
            dtype=float
        )
    ])

    row_totals = observed.sum(
        axis=1,
        keepdims=True
    )

    column_totals = observed.sum(
        axis=0,
        keepdims=True
    )

    expected = (
        row_totals
        @ column_totals
        / observations
    )

    valid = expected > 0

    chi_square = (
        (
            (
                observed
                - expected
            ) ** 2
            / np.where(
                valid,
                expected,
                1
            )
        )[valid]
        .sum()
    )

    degrees_of_freedom = (
        group_count
        - 1
    )

    p_value = float(
        chi2.sf(
            chi_square,
            degrees_of_freedom
        )
    )

    p_reliability = (
        significance_reliability(
            p_value
        )
    )

    statistical_reliability = np.sqrt(
        p_reliability
    )

    reliability = (
        event_reliability
        * statistical_reliability
    )

    signal = (
        effect_signal
        * reliability
    )

    return {
        "signal": clamp(signal),
        "effect_signal": clamp(effect_signal),
        "absolute_signal": clamp(absolute_signal),
        "relative_signal": clamp(relative_signal),
        "reliability": float(reliability),
        "event_reliability": float(event_reliability),
        "statistical_reliability": float(
            statistical_reliability
        ),
        "rate_range_pp": float(rate_range_pp),
        "max_lift": float(max_lift),
        "p_value": float(p_value),
        "positive_count": int(
            round(
                positive_count
            )
        ),
        "negative_count": int(
            round(
                negative_count
            )
        )
    }
