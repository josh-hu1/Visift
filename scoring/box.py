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


def get_semantic_type(profile, column_name):
    """
    Get a column's semantic type from the
    dataset profile.
    """

    for column in profile["column_profiles"]:

        if column["name"] == column_name:
            return column["semantic_type"]

    return None


def semantic_fit_score(
    x_type,
    y_type
):
    """
    Score how naturally the variable types fit
    a box plot.
    """

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

    x_score = category_scores.get(
        x_type,
        0
    )

    y_score = numeric_scores.get(
        y_type,
        0
    )

    if x_score == 0 or y_score == 0:
        return 0

    return (
        0.60 * x_score
        + 0.40 * y_score
    )


def category_readability_score(
    unique_count
):
    """
    Score whether the number of groups is
    reasonable for a box plot.
    """

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


def data_quality_score(
    df,
    x,
    y
):
    """
    Score the percentage of rows containing
    complete x/y observations.
    """

    if len(df) == 0:
        return 0

    complete = (
        df[[x, y]]
        .dropna()
    )

    return (
        len(complete) / len(df)
    ) * 100


def sample_support_score(counts):
    """
    Score whether groups contain enough
    observations to support distribution
    comparisons.
    """

    if counts.empty:
        return 0

    average_group_size = (
        counts.mean()
    )

    minimum_group_size = (
        counts.min()
    )

    # Box plots need several observations
    # within each group to show a distribution.
    average_score = clamp(
        average_group_size
        / 20
        * 100
    )

    minimum_score = clamp(
        minimum_group_size
        / 10
        * 100
    )

    return (
        0.50 * average_score
        + 0.50 * minimum_score
    )


def eta_squared_score(
    df,
    x,
    y
):
    """
    Measure how much total numeric variation
    is associated with group membership.

    Returns eta-squared as a 0-100 score.
    """

    clean = (
        df[[x, y]]
        .dropna()
    )

    if clean.empty:
        return 0

    if clean[x].nunique() <= 1:
        return 0

    overall_mean = (
        clean[y].mean()
    )

    total_variation = (
        (clean[y] - overall_mean) ** 2
    ).sum()

    if total_variation == 0:
        return 0

    between_group_variation = 0

    for _, group in clean.groupby(x):

        group_mean = (
            group[y].mean()
        )

        between_group_variation += (
            len(group)
            * (
                group_mean
                - overall_mean
            ) ** 2
        )

    eta_squared = (
        between_group_variation
        / total_variation
    )

    return clamp(
        eta_squared * 100
    )


def median_separation_score(
    df,
    x,
    y
):
    """
    Measure how different group medians are
    relative to the overall IQR.

    Large median differences make a box plot
    more informative.
    """

    clean = (
        df[[x, y]]
        .dropna()
    )

    if clean.empty:
        return 0

    medians = (
        clean.groupby(x)[y]
        .median()
    )

    if len(medians) <= 1:
        return 0

    median_range = (
        medians.max()
        - medians.min()
    )

    q1 = clean[y].quantile(0.25)
    q3 = clean[y].quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return 0

    relative_separation = (
        median_range / iqr
    )

    # Median differences equal to roughly
    # two overall IQRs receive full credit.
    score = (
        relative_separation
        / 2
        * 100
    )

    return clamp(score)


def grouped_outlier_score(
    df,
    x,
    y
):
    """
    Measure the prevalence of potential outliers
    within individual groups using the IQR rule.
    """

    clean = (
        df[[x, y]]
        .dropna()
    )

    if clean.empty:
        return 0, 0

    total_outliers = 0
    total_observations = 0

    for _, group in clean.groupby(x):

        values = group[y]

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outlier_count = (
            (
                (values < lower_bound)
                | (values > upper_bound)
            )
            .sum()
        )

        total_outliers += (
            outlier_count
        )

        total_observations += (
            len(values)
        )

    if total_observations == 0:
        return 0, 0

    outlier_rate = (
        total_outliers
        / total_observations
    )

    # Around 10% potential outliers
    # receives full signal credit.
    score = clamp(
        outlier_rate
        / 0.10
        * 100
    )

    return (
        score,
        outlier_rate
    )


def distribution_signal_score(
    df,
    x,
    y
):
    """
    Combine signals that make grouped
    distributions worth comparing.

    Components:
        group mean separation
        median separation
        within-group outliers
    """

    eta_score = eta_squared_score(
        df,
        x,
        y
    )

    median_score = (
        median_separation_score(
            df,
            x,
            y
        )
    )

    (
        outlier_signal,
        outlier_rate
    ) = grouped_outlier_score(
        df,
        x,
        y
    )

    signal = (
        0.45 * eta_score
        + 0.35 * median_score
        + 0.20 * outlier_signal
    )

    return {
        "signal": signal,
        "eta_squared_score": eta_score,
        "median_separation_score": median_score,
        "outlier_score": outlier_signal,
        "outlier_rate": outlier_rate
    }


def adjust_signal_for_support(
    signal,
    support
):
    """
    Reduce confidence in distribution differences
    when groups contain few observations.
    """

    reliability = (
        support / 100
    ) ** 0.5

    return (
        signal * reliability
    )


def combine_quality_and_signal(
    semantic_fit,
    readability,
    quality,
    support,
    signal
):
    """
    Combine box-plot suitability with
    distribution insight strength.
    """

    visualization_quality = (
        0.30 * semantic_fit
        + 0.20 * readability
        + 0.20 * quality
        + 0.30 * support
    )

    signal_multiplier = (
        0.35
        + 0.65 * (
            signal / 100
        )
    )

    final_score = (
        visualization_quality
        * signal_multiplier
    )

    return (
        final_score,
        visualization_quality
    )


def score_box_chart(
    df,
    candidate,
    profile
):
    """
    Score one box-plot candidate from
    0 to 100.
    """

    x = candidate["x"]
    y = candidate["y"]

    x_type = get_semantic_type(
        profile,
        x
    )

    y_type = get_semantic_type(
        profile,
        y
    )

    semantic_fit = (
        semantic_fit_score(
            x_type,
            y_type
        )
    )

    clean = (
        df[[x, y]]
        .dropna()
    )

    counts = (
        clean[x]
        .value_counts()
    )

    unique_count = (
        clean[x]
        .nunique()
    )

    readability = (
        category_readability_score(
            unique_count
        )
    )

    support = (
        sample_support_score(
            counts
        )
    )

    quality = (
        data_quality_score(
            df,
            x,
            y
        )
    )

    distribution = (
        distribution_signal_score(
            df,
            x,
            y
        )
    )

    raw_signal = (
        distribution["signal"]
    )

    signal = (
        adjust_signal_for_support(
            raw_signal,
            support
        )
    )

    (
        final_score,
        visualization_quality
    ) = combine_quality_and_signal(
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
        f"Group separation score: {distribution['eta_squared_score']:.1f}/100.",
        f"Median separation score: {distribution['median_separation_score']:.1f}/100.",
        f"Within-group outlier rate: {distribution['outlier_rate'] * 100:.1f}%.",
        f"Raw distribution signal: {raw_signal:.1f}/100.",
        f"Sample-adjusted signal: {signal:.1f}/100."
    ]

    return {
        "score": round(
            final_score,
            2
        ),

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

        "statistics": {
            "groups": int(
                unique_count
            ),
            "observations": int(
                len(clean)
            ),
            "average_group_size": round(
                float(counts.mean()),
                2
            ),
            "minimum_group_size": int(
                counts.min()
            ),
            "eta_squared_score": round(
                float(
                    distribution[
                        "eta_squared_score"
                    ]
                ),
                2
            ),
            "median_separation_score": round(
                float(
                    distribution[
                        "median_separation_score"
                    ]
                ),
                2
            ),
            "outlier_rate": round(
                float(
                    distribution[
                        "outlier_rate"
                    ]
                ),
                4
            ),
            "raw_signal": round(
                float(raw_signal),
                2
            )
        },

        "reasons": reasons
    }