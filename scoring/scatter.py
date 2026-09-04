import numpy as np
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


def semantic_fit_score(x_type, y_type):
    """
    Score how naturally the two semantic types
    fit a scatterplot.
    """

    continuous_types = {
        "numeric_continuous"
    }

    numeric_types = {
        "numeric_continuous",
        "numeric_discrete"
    }

    # Ideal scatterplot:
    # continuous numeric vs continuous numeric
    if (
        x_type in continuous_types
        and y_type in continuous_types
    ):
        return 100

    # Still valid if one variable is discrete numeric
    if (
        x_type in numeric_types
        and y_type in numeric_types
    ):
        return 80

    return 0


def data_quality_score(df, x, y):
    """
    Score the percentage of rows with complete
    x/y observations.
    """

    if len(df) == 0:
        return 0

    complete = df[[x, y]].dropna()

    return (
        len(complete) / len(df)
    ) * 100


def sample_support_score(n):
    """
    Score whether there are enough paired observations
    for a useful scatterplot.
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


def variance_score(series):
    """
    Penalize variables with little or no variation.
    """

    clean = series.dropna()

    if clean.empty:
        return 0

    unique_count = clean.nunique()

    if unique_count <= 1:
        return 0

    if unique_count <= 3:
        return 30

    if unique_count <= 5:
        return 50

    if unique_count <= 10:
        return 70

    return 100


def readability_score(df, x, y):
    """
    Estimate whether the variables provide enough
    distinct values for a useful scatterplot.
    """

    clean = df[[x, y]].dropna()

    if clean.empty:
        return 0

    x_score = variance_score(clean[x])
    y_score = variance_score(clean[y])

    return (
        x_score + y_score
    ) / 2


def relationship_signal_score(df, x, y):
    """
    Estimate the strength of the relationship between
    two numeric variables.

    Uses both Pearson and Spearman correlation.

    Pearson:
        linear relationship

    Spearman:
        monotonic relationship

    We use whichever detects the stronger pattern.
    """

    clean = df[[x, y]].dropna()

    if len(clean) < 3:
        return 0, 0, 0

    if (
        clean[x].nunique() <= 1
        or clean[y].nunique() <= 1
    ):
        return 0, 0, 0

    pearson = clean[x].corr(
        clean[y],
        method="pearson"
    )

    spearman = clean[x].corr(
        clean[y],
        method="spearman"
    )

    if pd.isna(pearson):
        pearson = 0

    if pd.isna(spearman):
        spearman = 0

    strongest_relationship = max(
        abs(pearson),
        abs(spearman)
    )

    signal = strongest_relationship * 100

    return (
        signal,
        pearson,
        spearman
    )


def combine_quality_and_signal(
    semantic_fit,
    readability,
    quality,
    support,
    signal
):
    """
    Combine scatterplot suitability with
    relationship strength.
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


def score_scatter_chart(
    df,
    candidate,
    profile
):
    """
    Score one scatterplot candidate from 0 to 100.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = df[[x, y]].dropna()

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

    quality = data_quality_score(
        df,
        x,
        y
    )

    support = sample_support_score(
        len(clean)
    )

    readability = readability_score(
        df,
        x,
        y
    )

    (
        signal,
        pearson,
        spearman
    ) = relationship_signal_score(
        df,
        x,
        y
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
        f"X semantic type: {x_type}.",
        f"Y semantic type: {y_type}.",
        f"Semantic fit for scatterplot: {semantic_type_display(semantic_fit)}.",
        f"{len(clean)} complete paired observations.",
        f"Data completeness: {quality:.1f}%.",
        f"Pearson correlation: {pearson:.3f}.",
        f"Spearman correlation: {spearman:.3f}.",
        f"Relationship signal: {signal:.1f}/100."
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
            )
        },

        "reasons": reasons
    }


def semantic_type_display(score):
    """
    Format semantic fit for output.
    """

    return f"{score:.1f}/100"