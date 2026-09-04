import pandas as pd


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
    a histogram.
    """

    if semantic_type == "numeric_continuous":
        return 100

    if semantic_type == "numeric_discrete":
        return 80

    return 0


def data_quality_score(df, column):
    """
    Score the percentage of non-missing values
    in the target column.
    """

    if len(df) == 0:
        return 0

    complete_count = df[column].notna().sum()

    completeness = (
        complete_count / len(df)
    )

    return completeness * 100


def sample_support_score(n):
    """
    Score whether there are enough observations
    for a meaningful histogram.
    """

    if n < 5:
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


def readability_score(unique_count):
    """
    Score whether the variable has enough distinct
    values to make a histogram informative.
    """

    if unique_count <= 1:
        return 0

    if unique_count <= 3:
        return 20

    if unique_count <= 5:
        return 40

    if unique_count <= 10:
        return 60

    if unique_count <= 20:
        return 80

    return 100


def skewness_score(series):
    """
    Convert absolute skewness into a 0-100 score.

    Rough interpretation:

        near 0:
            fairly symmetric

        around 1:
            noticeably skewed

        2 or greater:
            strongly skewed
    """

    clean = series.dropna()

    if len(clean) < 3:
        return 0, 0

    skewness = clean.skew()

    if pd.isna(skewness):
        return 0, 0

    score = clamp(
        abs(skewness) / 2 * 100
    )

    return score, skewness


def outlier_score(series):
    """
    Detect potential outliers using the IQR rule.

    Values below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR are treated as potential outliers.

    An outlier rate around 10% receives the
    maximum signal score.
    """

    clean = series.dropna()

    if clean.empty:
        return 0, 0

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return 0, 0

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outliers = clean[
        (clean < lower_bound)
        | (clean > upper_bound)
    ]

    outlier_rate = (
        len(outliers) / len(clean)
    )

    score = clamp(
        (outlier_rate / 0.10) * 100
    )

    return (
        score,
        outlier_rate
    )


def tail_asymmetry_score(series):
    """
    Measure robust asymmetry using the distance
    between the median and the 10th/90th percentiles.

    A symmetric distribution should have similar
    lower and upper tail distances.
    """

    clean = series.dropna()

    if len(clean) < 5:
        return 0, 0

    q10 = clean.quantile(0.10)
    median = clean.median()
    q90 = clean.quantile(0.90)

    lower_tail = (
        median - q10
    )

    upper_tail = (
        q90 - median
    )

    total_tail = (
        lower_tail + upper_tail
    )

    if total_tail == 0:
        return 0, 0

    asymmetry = abs(
        upper_tail - lower_tail
    ) / total_tail

    score = clamp(
        asymmetry * 100
    )

    return (
        score,
        asymmetry
    )


def distribution_signal_score(series):
    """
    Combine multiple distribution characteristics
    into one signal score.

    Current signals:

        skewness
        outlier prevalence
        tail asymmetry

    These measure whether viewing the distribution
    is likely to reveal something notable.
    """

    (
        skew_score,
        skewness
    ) = skewness_score(
        series
    )

    (
        outlier_signal,
        outlier_rate
    ) = outlier_score(
        series
    )

    (
        asymmetry_score,
        asymmetry
    ) = tail_asymmetry_score(
        series
    )

    signal = (
        0.45 * skew_score
        + 0.35 * outlier_signal
        + 0.20 * asymmetry_score
    )

    return {
        "signal": signal,
        "skewness": skewness,
        "skew_score": skew_score,
        "outlier_rate": outlier_rate,
        "outlier_score": outlier_signal,
        "tail_asymmetry": asymmetry,
        "tail_asymmetry_score": asymmetry_score
    }


def combine_quality_and_signal(
    semantic_fit,
    readability,
    quality,
    support,
    signal
):
    """
    Combine histogram suitability with
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


def score_histogram(
    df,
    candidate,
    profile
):
    """
    Score one histogram candidate from 0 to 100.
    """

    x = candidate["x"]

    semantic_type = get_semantic_type(
        profile,
        x
    )

    semantic_fit = semantic_fit_score(
        semantic_type
    )

    clean = pd.to_numeric(
        df[x],
        errors="coerce"
    ).dropna()

    observation_count = len(clean)

    unique_count = clean.nunique()

    quality = data_quality_score(
        df,
        x
    )

    support = sample_support_score(
        observation_count
    )

    readability = readability_score(
        unique_count
    )

    distribution = (
        distribution_signal_score(
            clean
        )
    )

    signal = distribution["signal"]

    final_score, visualization_quality = (
        combine_quality_and_signal(
            semantic_fit,
            readability,
            quality,
            support,
            signal
        )
    )

    if clean.empty:

        minimum = None
        maximum = None
        mean = None
        median = None

    else:

        minimum = clean.min()
        maximum = clean.max()
        mean = clean.mean()
        median = clean.median()

    reasons = [
        f"Semantic type: {semantic_type}.",
        f"Semantic fit for histogram: {semantic_fit:.1f}/100.",
        f"{observation_count} complete numeric observations.",
        f"{unique_count} distinct numeric values.",
        f"Data completeness: {quality:.1f}%.",
        f"Skewness: {distribution['skewness']:.3f}.",
        f"Potential outlier rate: {distribution['outlier_rate'] * 100:.1f}%.",
        f"Tail asymmetry: {distribution['tail_asymmetry']:.3f}.",
        f"Distribution signal: {signal:.1f}/100."
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
            "observations": observation_count,
            "unique_values": int(
                unique_count
            ),
            "min": (
                round(float(minimum), 4)
                if minimum is not None
                else None
            ),
            "max": (
                round(float(maximum), 4)
                if maximum is not None
                else None
            ),
            "mean": (
                round(float(mean), 4)
                if mean is not None
                else None
            ),
            "median": (
                round(float(median), 4)
                if median is not None
                else None
            ),
            "skewness": round(
                float(distribution["skewness"]),
                4
            ),
            "outlier_rate": round(
                float(
                    distribution["outlier_rate"]
                ),
                4
            ),
            "tail_asymmetry": round(
                float(
                    distribution["tail_asymmetry"]
                ),
                4
            )
        },

        "reasons": reasons
    }