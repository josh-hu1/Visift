import math
import pandas as pd


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


def count_signal_score(counts):
    """
    Estimate whether category frequencies show an
    interesting difference.

    Balanced distributions receive a low signal score,
    while increasingly imbalanced distributions receive
    higher scores.
    """

    if len(counts) <= 1:
        return 0

    probabilities = counts / counts.sum()

    entropy = -sum(
        p * math.log(p)
        for p in probabilities
        if p > 0
    )

    max_entropy = math.log(len(counts))

    normalized_entropy = (
        entropy / max_entropy
        if max_entropy > 0
        else 1
    )

    imbalance = 1 - normalized_entropy

    # Greater imbalance produces a stronger frequency signal.
    return imbalance * 100


def group_separation_score(df, x, y):
    """
    Measure how strongly group membership explains
    variation in a numeric variable.

    Uses eta-squared:

        between-group variation / total variation

    Returns a score from 0 to 100.
    """

    clean = df[[x, y]].dropna()

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

        signal = count_signal_score(
            counts
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
            f"Average category size: {counts.mean():.1f} observations.",
            f"Smallest category: {counts.min()} observations.",
            f"Data completeness: {quality:.1f}%.",
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