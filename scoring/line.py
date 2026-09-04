import pandas as pd


NUMERIC_TYPES = {
    "numeric_continuous",
    "numeric_discrete"
}

TIME_TYPES = {
    "datetime",
    "temporal"
}


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


def semantic_fit_score(x_type, y_type):
    """
    Score how naturally the semantic types fit
    a line chart.
    """

    if (
        x_type in TIME_TYPES
        and y_type == "numeric_continuous"
    ):
        return 100

    if (
        x_type in TIME_TYPES
        and y_type == "numeric_discrete"
    ):
        return 90

    return 0


def prepare_temporal_data(df, x, y, x_type):
    """
    Prepare a temporal/numeric pair for line-chart
    analysis.

    Datetimes are converted into elapsed days.
    Numeric temporal fields such as years are kept
    as numeric values.

    Duplicate time points are aggregated by mean
    when calculating the trend.
    """

    clean = df[[x, y]].copy()

    clean[y] = pd.to_numeric(
        clean[y],
        errors="coerce"
    )

    if x_type == "datetime":

        clean[x] = pd.to_datetime(
            clean[x],
            errors="coerce"
        )

    elif x_type == "temporal":

        clean[x] = pd.to_numeric(
            clean[x],
            errors="coerce"
        )

    else:
        return pd.DataFrame(), pd.DataFrame()

    clean = clean.dropna(
        subset=[x, y]
    )

    if clean.empty:
        return clean, pd.DataFrame()

    clean = clean.sort_values(x)

    # -----------------------------------
    # Convert temporal variable to a
    # numeric axis for correlation.
    # -----------------------------------

    if x_type == "datetime":

        first_time = clean[x].min()

        clean["_time_numeric"] = (
            clean[x] - first_time
        ).dt.total_seconds() / 86400

    else:

        clean["_time_numeric"] = (
            clean[x].astype(float)
        )

    # -----------------------------------
    # Aggregate duplicate time points
    # -----------------------------------

    trend_data = (
        clean
        .groupby(
            "_time_numeric",
            as_index=False
        )[y]
        .mean()
        .sort_values("_time_numeric")
    )

    return clean, trend_data


def data_quality_score(
    total_rows,
    complete_rows
):
    """
    Score the percentage of usable temporal/numeric
    observations.
    """

    if total_rows == 0:
        return 0

    completeness = (
        complete_rows / total_rows
    )

    return completeness * 100


def observation_support_score(n):
    """
    Score the number of complete observations.
    """

    if n < 3:
        return 0

    if n < 10:
        return 20

    if n < 20:
        return 40

    if n < 30:
        return 60

    if n < 50:
        return 75

    if n < 100:
        return 90

    return 100


def time_point_support_score(unique_time_points):
    """
    Score how many distinct temporal positions are
    available for detecting a trend.
    """

    if unique_time_points <= 1:
        return 0

    if unique_time_points < 3:
        return 20

    if unique_time_points < 5:
        return 40

    if unique_time_points < 10:
        return 60

    if unique_time_points < 20:
        return 75

    if unique_time_points < 50:
        return 90

    return 100


def sample_support_score(
    observation_count,
    unique_time_points
):
    """
    Combine raw sample size with the number of
    distinct time points.

    Distinct temporal positions matter more than
    having many repeated observations at only a
    few dates.
    """

    observation_score = (
        observation_support_score(
            observation_count
        )
    )

    time_score = (
        time_point_support_score(
            unique_time_points
        )
    )

    return (
        0.40 * observation_score
        + 0.60 * time_score
    )


def readability_score(unique_time_points):
    """
    Estimate how readable a line chart would be
    based on the number of distinct time points.
    """

    if unique_time_points <= 1:
        return 0

    if unique_time_points == 2:
        return 30

    if unique_time_points <= 5:
        return 60

    if unique_time_points <= 200:
        return 100

    if unique_time_points <= 500:
        return 95

    if unique_time_points <= 1000:
        return 85

    if unique_time_points <= 2500:
        return 70

    return 55


def trend_signal_score(
    trend_data,
    y
):
    """
    Estimate temporal trend strength.

    Pearson correlation captures linear trends.
    Spearman correlation captures monotonic trends.

    The stronger absolute relationship is used
    as the raw signal.
    """

    if len(trend_data) < 3:
        return 0, 0, 0, "insufficient"

    time_values = (
        trend_data["_time_numeric"]
    )

    y_values = trend_data[y]

    if (
        time_values.nunique() <= 1
        or y_values.nunique() <= 1
    ):
        return 0, 0, 0, "none"

    pearson = time_values.corr(
        y_values,
        method="pearson"
    )

    spearman = time_values.corr(
        y_values,
        method="spearman"
    )

    if pd.isna(pearson):
        pearson = 0

    if pd.isna(spearman):
        spearman = 0

    if abs(pearson) >= abs(spearman):
        strongest = pearson
    else:
        strongest = spearman

    signal = abs(strongest) * 100

    if strongest > 0.05:
        direction = "upward"

    elif strongest < -0.05:
        direction = "downward"

    else:
        direction = "flat"

    return (
        signal,
        pearson,
        spearman,
        direction
    )


def adjust_signal_for_support(
    signal,
    support
):
    """
    Reduce confidence in apparent temporal trends
    when there are too few usable time points.

    The square-root adjustment prevents small
    samples from receiving very high confidence
    solely because they happen to form a line.
    """

    reliability = (
        support / 100
    ) ** 0.5

    return signal * reliability


def combine_quality_and_signal(
    semantic_fit,
    readability,
    quality,
    support,
    signal
):
    """
    Combine visualization suitability with
    temporal insight strength.
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

    return (
        final_score,
        visualization_quality
    )


def score_line_chart(
    df,
    candidate,
    profile
):
    """
    Score one line-chart candidate from 0 to 100.
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

    semantic_fit = semantic_fit_score(
        x_type,
        y_type
    )

    clean, trend_data = (
        prepare_temporal_data(
            df,
            x,
            y,
            x_type
        )
    )

    complete_count = len(clean)

    unique_time_points = (
        len(trend_data)
    )

    quality = data_quality_score(
        len(df),
        complete_count
    )

    support = sample_support_score(
        complete_count,
        unique_time_points
    )

    readability = readability_score(
        unique_time_points
    )

    (
        raw_signal,
        pearson,
        spearman,
        direction
    ) = trend_signal_score(
        trend_data,
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

    # -----------------------------------
    # Calculate temporal span
    # -----------------------------------

    if unique_time_points > 1:

        temporal_span = (
            trend_data["_time_numeric"].max()
            - trend_data["_time_numeric"].min()
        )

    else:

        temporal_span = 0

    if x_type == "datetime":
        span_description = (
            f"{temporal_span:.1f} days"
        )
    else:
        span_description = (
            f"{temporal_span:.1f} temporal units"
        )

    reasons = [
        f"X semantic type: {x_type}.",
        f"Y semantic type: {y_type}.",
        f"Semantic fit for line chart: {semantic_fit:.1f}/100.",
        f"{complete_count} complete observations.",
        f"{unique_time_points} distinct time points.",
        f"Temporal span: {span_description}.",
        f"Data completeness: {quality:.1f}%.",
        f"Pearson time correlation: {pearson:.3f}.",
        f"Spearman time correlation: {spearman:.3f}.",
        f"Raw trend signal: {raw_signal:.1f}/100.",
        f"Sample-adjusted trend signal: {signal:.1f}/100.",
        f"Detected trend direction: {direction}."
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
            "pearson": round(
                pearson,
                4
            ),
            "spearman": round(
                spearman,
                4
            ),
            "raw_signal": round(
                raw_signal,
                2
            ),
            "direction": direction,
            "time_points": unique_time_points,
            "temporal_span": round(
                temporal_span,
                2
            )
        },

        "reasons": reasons
    }