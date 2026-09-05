import math
import pandas as pd
from value_normalization import coerce_numeric_or_boolean

from scipy.stats import chisquare


def clamp(value, minimum=0, maximum=100):
    """
    Keep a score between 0 and 100.
    """
    return max(minimum, min(value, maximum))


def get_semantic_type(profile, column_name):
    """
    Get the semantic type of a column from
    the dataset profile.
    """

    for column in profile["column_profiles"]:
        if column["name"] == column_name:
            return column["semantic_type"]

    return None


def semantic_fit_score(semantic_type):
    """
    Score how naturally a semantic type fits
    a bar chart.
    """

    scores = {
        "categorical": 100,
        "ordinal": 95,
        "boolean": 90,
        "geographic": 65
    }

    return scores.get(semantic_type, 50)


def category_readability_score(unique_count):
    """
    Score how readable the number of categories would be
    in a bar chart.
    """

    if unique_count <= 1:
        return 20

    if unique_count <= 8:
        return 100

    if unique_count <= 12:
        return 85

    if unique_count <= 20:
        return 60

    if unique_count <= 30:
        return 35

    return 10


def adjust_signal_for_support(signal, support):
    """
    Reduce confidence in an apparent statistical
    signal when group sample sizes are small.

    Uses the square root of the support proportion
    so weak samples are penalized without completely
    eliminating potentially interesting effects.
    """

    reliability = (support / 100) ** 0.5

    return signal * reliability


def sample_support_score(counts):
    """
    Score whether each category has enough observations
    to support comparison.
    """

    if counts.empty:
        return 0

    average_group_size = counts.mean()
    minimum_group_size = counts.min()

    # Average group size gets full credit around 10+
    average_score = clamp(
        (average_group_size / 10) * 100
    )

    # Smallest group gets full credit around 5+
    minimum_score = clamp(
        (minimum_group_size / 5) * 100
    )

    return (
        0.60 * average_score
        + 0.40 * minimum_score
    )


def data_quality_score(df, columns):
    """
    Penalize missing values in the columns required
    for the visualization.
    """

    if len(df) == 0:
        return 0

    complete_rows = df[columns].dropna()

    completeness = len(complete_rows) / len(df)

    return completeness * 100


COUNT_STRONG_P = 0.001
COUNT_WEAK_P = 0.10
COUNT_MIN_RELIABILITY = 0.20
COUNT_IMBALANCE_EXPONENT = 0.50


def count_reliability_score(p_value):
    """
    Convert a goodness-of-fit p-value into a
    0-1 reliability multiplier.

    Category imbalance is measured against an
    equal-frequency reference distribution. Small
    samples can look strongly imbalanced by chance,
    so weak statistical evidence receives only
    partial signal credit.
    """

    if (
        p_value is None
        or not math.isfinite(
            p_value
        )
    ):
        return COUNT_MIN_RELIABILITY

    if p_value <= COUNT_STRONG_P:
        return 1.0

    if p_value >= COUNT_WEAK_P:
        return COUNT_MIN_RELIABILITY

    log_strong = math.log10(
        COUNT_STRONG_P
    )

    log_weak = math.log10(
        COUNT_WEAK_P
    )

    log_p = math.log10(
        p_value
    )

    position = (
        (
            log_weak
            - log_p
        )
        / (
            log_weak
            - log_strong
        )
    )

    position = clamp(
        position,
        0,
        1
    )

    return (
        COUNT_MIN_RELIABILITY
        + (
            1
            - COUNT_MIN_RELIABILITY
        )
        * position
    )


def count_signal_details(counts):
    """
    Estimate category-frequency imbalance and its
    statistical reliability.

    The raw effect is normalized entropy deficit.
    Entropy deficit is bounded and category-count
    aware, but its numeric scale is compressed: a
    visibly meaningful imbalance can still produce
    a relatively small raw value.

    A square-root calibration expands the lower and
    middle part of the scale while preserving 0-100
    bounds. The transformed effect is then adjusted
    by a chi-square goodness-of-fit reliability
    multiplier.
    """

    default_result = {
        "signal": 0,
        "entropy_imbalance": 0,
        "effect_signal": 0,
        "p_value": 1,
        "reliability": 0
    }

    if len(counts) <= 1:
        return default_result

    total = counts.sum()

    if total <= 0:
        return default_result

    probabilities = (
        counts
        / total
    )

    entropy = -sum(
        p * math.log(p)
        for p in probabilities
        if p > 0
    )

    max_entropy = math.log(
        len(counts)
    )

    normalized_entropy = (
        entropy
        / max_entropy
        if max_entropy > 0
        else 1
    )

    entropy_imbalance = clamp(
        1
        - normalized_entropy,
        0,
        1
    )

    effect_signal = (
        entropy_imbalance
        ** COUNT_IMBALANCE_EXPONENT
        * 100
    )

    expected_count = (
        total
        / len(counts)
    )

    try:
        result = chisquare(
            counts.to_numpy(
                dtype=float
            ),
            f_exp=[
                expected_count
            ]
            * len(counts)
        )

        p_value = float(
            result.pvalue
        )

    except Exception:
        p_value = 1

    reliability = (
        count_reliability_score(
            p_value
        )
    )

    signal = clamp(
        effect_signal
        * reliability
    )

    return {
        "signal": float(
            signal
        ),
        "entropy_imbalance": float(
            entropy_imbalance
        ),
        "effect_signal": float(
            effect_signal
        ),
        "p_value": float(
            p_value
        ),
        "reliability": float(
            reliability
        )
    }


def count_signal_score(counts):
    """
    Return the calibrated count-bar signal.

    This wrapper preserves the existing public
    function name used by evaluation code.
    """

    return count_signal_details(
        counts
    )[
        "signal"
    ]


def group_separation_score(df, x, y):
    """
    Measure how strongly group membership explains
    variation in a numeric variable.

    Uses eta-squared:

        between-group variation / total variation

    Returns a score from 0 to 100.
    """

    clean = df[[x, y]].copy()

    clean[y] = coerce_numeric_or_boolean(
        clean[y]
    )

    clean = clean.dropna(
        subset=[x, y]
    )

    if clean.empty:
        return 0

    if clean[x].nunique() <= 1:
        return 0

    overall_mean = clean[y].mean()

    total_variation = (
        (clean[y] - overall_mean) ** 2
    ).sum()

    if total_variation == 0:
        return 0

    between_group_variation = 0

    for _, group in clean.groupby(x):

        group_mean = group[y].mean()

        between_group_variation += (
            len(group)
            * (group_mean - overall_mean) ** 2
        )

    eta_squared = (
        between_group_variation
        / total_variation
    )

    return clamp(eta_squared * 100)


def score_bar_chart(df, candidate, profile):
    """
    Score one bar-chart candidate from 0 to 100.

    The score has two stages:

    1. Visualization quality:
        semantic fit:   30%
        readability:    20%
        data quality:   20%
        sample support: 30%

    2. Insight strength:
        signal modifies the visualization-quality score
        so technically valid but uninformative charts
        cannot receive very high recommendations.
    """

    x = candidate["x"]

    aggregation = candidate.get(
        "aggregation",
        "count"
    )

    semantic_type = get_semantic_type(
        profile,
        x
    )

    semantic_fit = semantic_fit_score(
        semantic_type
    )

    # -----------------------------------
    # Count bar chart
    # -----------------------------------

    if aggregation == "count":

        clean = df[[x]].dropna()

        counts = clean[x].value_counts()

        unique_count = clean[x].nunique()

        readability = category_readability_score(
            unique_count
        )

        support = sample_support_score(
            counts
        )

        quality = data_quality_score(
            df,
            [x]
        )

        count_signal = (
            count_signal_details(
                counts
            )
        )

        signal = count_signal[
            "signal"
        ]

        final_score, visualization_quality = (
            combine_quality_and_signal(
                semantic_fit,
                readability,
                quality,
                support,
                signal
            )
        )

        reasons = [
            f"Semantic type: {semantic_type}.",
            f"Semantic fit for bar chart: {semantic_fit:.1f}/100.",
            f"{unique_count} categories detected.",
            f"Average category size: {counts.mean():.1f} observations.",
            f"Smallest category: {counts.min()} observations.",
            f"Data completeness: {quality:.1f}%.",
            (
                "Normalized entropy imbalance: "
                f"{count_signal['entropy_imbalance']:.3f}."
            ),
            (
                "Frequency-imbalance effect: "
                f"{count_signal['effect_signal']:.1f}/100."
            ),
            (
                "Imbalance reliability: "
                f"{count_signal['reliability']:.2f} "
                f"(p={count_signal['p_value']:.3g})."
            ),
            f"Category frequency signal: {signal:.1f}/100."
        ]

    # -----------------------------------
    # Category + numeric bar chart
    # -----------------------------------

    else:

        y = candidate["y"]

        clean = df[[x, y]].dropna()

        counts = clean[x].value_counts()

        unique_count = clean[x].nunique()

        readability = category_readability_score(
            unique_count
        )

        support = sample_support_score(
            counts
        )

        quality = data_quality_score(
            df,
            [x, y]
        )

        raw_signal = group_separation_score(
            df,
            x,
            y
        )

        signal = adjust_signal_for_support(
            raw_signal,
            support
        )

        final_score, visualization_quality = (
            combine_quality_and_signal(
                semantic_fit,
                readability,
                quality,
                support,
                signal
            )
        )

        reasons = [
            f"Semantic type: {semantic_type}.",
            f"Semantic fit for bar chart: {semantic_fit:.1f}/100.",
            f"{unique_count} categories detected.",
            f"Average group size: {counts.mean():.1f} observations.",
            f"Smallest group: {counts.min()} observations.",
            f"Data completeness: {quality:.1f}%.",
            f"Raw group separation: {raw_signal:.1f}/100.",
            f"Sample-adjusted signal: {signal:.1f}/100."
        ]

    return {
        "score": round(final_score, 2),

        "components": {
            "semantic_fit": round(
                semantic_fit,
                2
            ),
            "readability": round(
                readability,
                2
            ),
            "sample_support": round(
                support,
                2
            ),
            "data_quality": round(
                quality,
                2
            ),
            "signal": round(
                signal,
                2
            ), 
            "visualization_quality": round(
                visualization_quality,
                2
            )
        },

        "reasons": reasons
    }

def combine_quality_and_signal(
    semantic_fit,
    readability,
    quality,
    support,
    signal
):
    """
    Combine visualization suitability with insight strength.

    A chart cannot receive a very high recommendation score
    solely because it is clean and readable. It also needs
    to reveal a meaningful pattern.
    """

    visualization_quality = (
        0.30 * semantic_fit
        + 0.20 * readability
        + 0.20 * quality
        + 0.30 * support
    )

    signal_multiplier = (
        0.35
        + 0.65 * (signal / 100)
    )

    final_score = (
        visualization_quality
        * signal_multiplier
    )

    return final_score, visualization_quality