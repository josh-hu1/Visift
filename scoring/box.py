import math

import pandas as pd


CATEGORY_TYPES = {
    "categorical",
    "ordinal",
    "boolean",
    "geographic"
}

NUMERIC_TYPES = {
    "numeric_continuous",
    "numeric_discrete"
}


def clamp(value, minimum=0, maximum=100):
    """
    Keep a score between 0 and 100.
    """
    return max(minimum, min(value, maximum))


def piecewise_score(
    value,
    points
):
    """
    Map a statistic to a calibrated score using
    linear interpolation between control points.
    """

    if value is None:
        return 0

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return 0

    if not math.isfinite(
        value
    ):
        return 0

    if value <= points[0][0]:
        return points[0][1]

    if value >= points[-1][0]:
        return points[-1][1]

    for (
        x0,
        y0
    ), (
        x1,
        y1
    ) in zip(
        points,
        points[1:]
    ):

        if (
            x0
            <= value
            <= x1
        ):

            if x1 == x0:
                return y1

            fraction = (
                value - x0
            ) / (
                x1 - x0
            )

            return (
                y0
                + fraction
                * (
                    y1 - y0
                )
            )

    return 0


def get_semantic_type(profile, column_name):
    for column in profile["column_profiles"]:
        if column["name"] == column_name:
            return column["semantic_type"]
    return None


def semantic_fit_score(x_type, y_type):
    category_scores = {
        "categorical": 100,
        "ordinal": 95,
        "boolean": 85,
        "geographic": 60
    }
    numeric_scores = {
        "numeric_continuous": 100,
        "numeric_discrete": 85
    }
    x_score = category_scores.get(x_type, 0)
    y_score = numeric_scores.get(y_type, 0)
    if x_score == 0 or y_score == 0:
        return 0
    return 0.60 * x_score + 0.40 * y_score


def category_readability_score(unique_count):
    if unique_count <= 1:
        return 0
    if unique_count <= 6:
        return 100
    if unique_count <= 10:
        return 90
    if unique_count <= 15:
        return 70
    if unique_count <= 25:
        return 45
    return 20


def data_quality_score(df, x, y):
    if len(df) == 0:
        return 0
    complete = df[[x, y]].dropna()
    return (len(complete) / len(df)) * 100


def sample_support_score(counts):
    if counts.empty:
        return 0
    average_group_size = counts.mean()
    minimum_group_size = counts.min()
    average_score = clamp(average_group_size / 20 * 100)
    minimum_score = clamp(minimum_group_size / 10 * 100)
    return 0.50 * average_score + 0.50 * minimum_score


def eta_squared_score(df, x, y):
    clean = df[[x, y]].dropna()
    if clean.empty or clean[x].nunique() <= 1:
        return 0
    overall_mean = clean[y].mean()
    total_variation = ((clean[y] - overall_mean) ** 2).sum()
    if total_variation == 0:
        return 0
    between_group_variation = 0
    for _, group in clean.groupby(x):
        group_mean = group[y].mean()
        between_group_variation += len(group) * (group_mean - overall_mean) ** 2
    eta_squared = between_group_variation / total_variation
    return clamp(eta_squared * 100)


def median_separation_score(df, x, y):
    clean = df[[x, y]].dropna()
    if clean.empty:
        return 0
    medians = clean.groupby(x)[y].median()
    if len(medians) <= 1:
        return 0
    median_range = medians.max() - medians.min()
    q1 = clean[y].quantile(0.25)
    q3 = clean[y].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    relative_separation = median_range / iqr
    return clamp(relative_separation / 2 * 100)


def grouped_outlier_statistics(df, x, y):
    clean = df[[x, y]].dropna()
    default = {
        "overall_rate": 0.0,
        "rate_disparity": 0.0,
        "score": 0.0,
        "minimum_rate": 0.0,
        "maximum_rate": 0.0
    }
    if clean.empty:
        return default
    rates = []
    weighted_outliers = 0
    weighted_observations = 0
    for _, group in clean.groupby(x):
        values = group[y]
        if len(values) < 8:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((values < lower) | (values > upper)).sum())
        rate = outlier_count / len(values)
        rates.append(rate)
        weighted_outliers += outlier_count
        weighted_observations += len(values)
    if len(rates) < 2 or weighted_observations == 0:
        return default
    rates_series = pd.Series(rates)
    median_rate = float(rates_series.median())
    max_rate = float(rates_series.max())
    min_rate = float(rates_series.min())
    disparity = max(0.0, max_rate - median_rate)
    score = piecewise_score(
        disparity,
        [
            (0.000, 0),
            (0.015, 0),
            (0.030, 10),
            (0.060, 25),
            (0.100, 45),
            (0.200, 65),
            (0.300, 78),
            (0.400, 88)
        ]
    )
    return {
        "overall_rate": weighted_outliers / weighted_observations,
        "rate_disparity": disparity,
        "score": clamp(score),
        "minimum_rate": min_rate,
        "maximum_rate": max_rate
    }


def grouped_outlier_score(df, x, y):
    stats = grouped_outlier_statistics(df, x, y)
    return stats["score"], stats["overall_rate"]


def dispersion_difference_score(df, x, y):
    clean = df[[x, y]].dropna()
    default = {
        "score": 0.0,
        "dispersion_ratio": 1.0,
        "minimum_iqr": 0.0,
        "maximum_iqr": 0.0
    }
    if clean.empty:
        return default
    iqrs = []
    for _, group in clean.groupby(x):
        values = group[y]
        if len(values) < 8:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = float(q3 - q1)
        if math.isfinite(iqr) and iqr > 0:
            iqrs.append(iqr)
    if len(iqrs) < 2:
        return default
    minimum_iqr = min(iqrs)
    maximum_iqr = max(iqrs)
    if minimum_iqr <= 0:
        return default
    ratio = maximum_iqr / minimum_iqr
    score = piecewise_score(
        ratio,
        [
            (1.00, 0),
            (1.15, 0),
            (1.30, 10),
            (1.50, 20),
            (2.00, 40),
            (3.00, 65),
            (4.00, 80),
            (6.00, 90),
            (8.00, 95)
        ]
    )
    return {
        "score": clamp(score),
        "dispersion_ratio": ratio,
        "minimum_iqr": minimum_iqr,
        "maximum_iqr": maximum_iqr
    }


def distribution_signal_score(df, x, y):
    eta_score = eta_squared_score(df, x, y)
    median_score = median_separation_score(df, x, y)
    location_signal = 0.55 * eta_score + 0.45 * median_score

    outlier_stats = grouped_outlier_statistics(df, x, y)
    dispersion_stats = dispersion_difference_score(df, x, y)

    group_count = int(df[[x, y]].dropna()[x].nunique())

    if group_count <= 4:
        comparison_reliability = 1.0
    else:
        comparison_reliability = max(
            0.40,
            math.sqrt(4 / group_count)
        )

    outlier_difference_signal = (
        outlier_stats["score"]
        * comparison_reliability
    )

    dispersion_difference_signal = (
        dispersion_stats["score"]
        * comparison_reliability
    )

    signal = max(
        location_signal,
        outlier_difference_signal,
        dispersion_difference_signal
    )

    return {
        "signal": signal,
        "location_score": location_signal,
        "eta_squared_score": eta_score,
        "median_separation_score": median_score,
        "outlier_score": outlier_difference_signal,
        "outlier_rate": outlier_stats["overall_rate"],
        "outlier_rate_disparity": outlier_stats["rate_disparity"],
        "minimum_group_outlier_rate": outlier_stats["minimum_rate"],
        "maximum_group_outlier_rate": outlier_stats["maximum_rate"],
        "dispersion_score": dispersion_difference_signal,
        "dispersion_ratio": dispersion_stats["dispersion_ratio"]
    }


def adjust_signal_for_support(signal, support):
    reliability = (support / 100) ** 0.5
    return signal * reliability


def combine_quality_and_signal(semantic_fit, readability, quality, support, signal):
    visualization_quality = (
        0.30 * semantic_fit
        + 0.20 * readability
        + 0.20 * quality
        + 0.30 * support
    )
    signal_multiplier = 0.35 + 0.65 * (signal / 100)
    final_score = visualization_quality * signal_multiplier
    return final_score, visualization_quality


def score_box_chart(df, candidate, profile):
    x = candidate["x"]
    y = candidate["y"]
    x_type = get_semantic_type(profile, x)
    y_type = get_semantic_type(profile, y)
    semantic_fit = semantic_fit_score(x_type, y_type)
    clean = df[[x, y]].dropna()
    counts = clean[x].value_counts()
    unique_count = clean[x].nunique()
    readability = category_readability_score(unique_count)
    support = sample_support_score(counts)
    quality = data_quality_score(df, x, y)
    distribution = distribution_signal_score(df, x, y)
    raw_signal = distribution["signal"]
    signal = adjust_signal_for_support(raw_signal, support)
    final_score, visualization_quality = combine_quality_and_signal(
        semantic_fit,
        readability,
        quality,
        support,
        signal
    )
    reasons = [
        f"X semantic type: {x_type}.",
        f"Y semantic type: {y_type}.",
        f"Semantic fit for box plot: {semantic_fit:.1f}/100.",
        f"{unique_count} groups detected.",
        f"{len(clean)} complete observations.",
        f"Average group size: {counts.mean():.1f}.",
        f"Smallest group size: {counts.min()}.",
        f"Data completeness: {quality:.1f}%.",
        f"Location difference signal: {distribution['location_score']:.1f}/100.",
        f"Dispersion difference signal: {distribution['dispersion_score']:.1f}/100.",
        f"Outlier-pattern difference signal: {distribution['outlier_score']:.1f}/100.",
        f"Raw distribution signal: {raw_signal:.1f}/100.",
        f"Sample-adjusted signal: {signal:.1f}/100."
    ]
    return {
        "score": round(final_score, 2),
        "components": {
            "semantic_fit": round(semantic_fit, 2),
            "readability": round(readability, 2),
            "sample_support": round(support, 2),
            "data_quality": round(quality, 2),
            "signal": round(signal, 2),
            "visualization_quality": round(visualization_quality, 2)
        },
        "statistics": {
            "groups": int(unique_count),
            "observations": int(len(clean)),
            "average_group_size": round(float(counts.mean()), 2),
            "minimum_group_size": int(counts.min()),
            "location_score": round(float(distribution["location_score"]), 2),
            "eta_squared_score": round(float(distribution["eta_squared_score"]), 2),
            "median_separation_score": round(float(distribution["median_separation_score"]), 2),
            "dispersion_score": round(float(distribution["dispersion_score"]), 2),
            "dispersion_ratio": round(float(distribution["dispersion_ratio"]), 4),
            "outlier_score": round(float(distribution["outlier_score"]), 2),
            "outlier_rate": round(float(distribution["outlier_rate"]), 4),
            "outlier_rate_disparity": round(float(distribution["outlier_rate_disparity"]), 4),
            "maximum_group_outlier_rate": round(float(distribution["maximum_group_outlier_rate"]), 4),
            "raw_signal": round(float(raw_signal), 2)
        },
        "reasons": reasons
    }
