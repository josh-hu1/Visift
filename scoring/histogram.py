import math

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


NUMERIC_TYPES = {
    "numeric_continuous",
    "numeric_discrete"
}


# ==================================================
# CALIBRATION CONSTANTS
# ==================================================

MULTIMODALITY_MAX_SAMPLE = 2000
MULTIMODALITY_SCALE = 0.80


# ==================================================
# COMMON HELPERS
# ==================================================

def clamp(
    value,
    minimum=0,
    maximum=100
):
    """
    Keep a score between 0 and 100.
    """

    return max(
        minimum,
        min(
            value,
            maximum
        )
    )


def piecewise_score(
    value,
    points
):
    """
    Map a numeric statistic onto a calibrated
    0-100 score using linear interpolation between
    control points.

    Values below the first point receive the first
    score. Values above the last point receive the
    last score.
    """

    if value is None:
        return 0

    try:
        numeric_value = float(
            value
        )
    except (
        TypeError,
        ValueError
    ):
        return 0

    if not math.isfinite(
        numeric_value
    ):
        return 0

    x_values = [
        point[0]
        for point in points
    ]

    y_values = [
        point[1]
        for point in points
    ]

    return float(
        np.interp(
            numeric_value,
            x_values,
            y_values
        )
    )


def get_semantic_type(
    profile,
    column_name
):
    """
    Get the semantic type of a column from
    the dataset profile.
    """

    for column in profile[
        "column_profiles"
    ]:

        if (
            column["name"]
            == column_name
        ):
            return column[
                "semantic_type"
            ]

    return None


# ==================================================
# CHART SUITABILITY
# ==================================================

def semantic_fit_score(
    semantic_type
):
    """
    Score how naturally a semantic type fits
    a histogram.
    """

    if (
        semantic_type
        == "numeric_continuous"
    ):
        return 100

    if (
        semantic_type
        == "numeric_discrete"
    ):
        return 80

    return 0


def data_quality_score(
    df,
    column
):
    """
    Score the percentage of non-missing values
    in the target column.
    """

    if len(df) == 0:
        return 0

    complete_count = (
        df[column]
        .notna()
        .sum()
    )

    completeness = (
        complete_count
        / len(df)
    )

    return (
        completeness
        * 100
    )


def sample_support_score(
    n
):
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


def readability_score(
    unique_count
):
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


# ==================================================
# DISTRIBUTION SIGNALS
# ==================================================

def skewness_score(
    series
):
    """
    Measure conventional moment skewness.

    Important:
        Classical skewness is highly sensitive to
        extreme observations. It is therefore kept
        primarily as descriptive evidence rather
        than being allowed to dominate the final
        histogram signal.

    The returned score is a conservative diagnostic
    score. Robust tail asymmetry is the primary
    skew-related signal used for recommendation
    scoring.
    """

    clean = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if len(clean) < 3:
        return (
            0,
            0
        )

    skewness = (
        clean.skew()
    )

    if pd.isna(
        skewness
    ):
        return (
            0,
            0
        )

    absolute_skewness = abs(
        float(
            skewness
        )
    )

    score = piecewise_score(
        absolute_skewness,
        [
            (0.00, 0),
            (0.30, 0),
            (0.60, 10),
            (1.20, 22),
            (2.00, 35),
            (3.50, 50),
            (5.00, 60),
            (8.00, 70),
            (12.00, 80)
        ]
    )

    return (
        clamp(
            score
        ),
        float(
            skewness
        )
    )


def outlier_score(
    series
):
    """
    Detect potential outliers using the IQR rule.

    Values below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR are treated as potential outliers.

    The calibration intentionally treats roughly
    1% or fewer IQR outliers as ordinary behavior.
    This avoids assigning signal merely because a
    large approximately normal sample naturally
    contains some observations outside the IQR
    fences.
    """

    clean = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if clean.empty:
        return (
            0,
            0
        )

    q1 = clean.quantile(
        0.25
    )

    q3 = clean.quantile(
        0.75
    )

    iqr = (
        q3
        - q1
    )

    if (
        pd.isna(iqr)
        or iqr <= 0
    ):
        return (
            0,
            0
        )

    lower_bound = (
        q1
        - 1.5 * iqr
    )

    upper_bound = (
        q3
        + 1.5 * iqr
    )

    outlier_mask = (
        (
            clean
            < lower_bound
        )
        | (
            clean
            > upper_bound
        )
    )

    outlier_rate = (
        float(
            outlier_mask.mean()
        )
    )

    score = piecewise_score(
        outlier_rate,
        [
            (0.00, 0),
            (0.01, 0),
            (0.02, 15),
            (0.04, 30),
            (0.07, 50),
            (0.10, 65),
            (0.15, 78),
            (0.25, 92)
        ]
    )

    return (
        clamp(
            score
        ),
        outlier_rate
    )


def tail_asymmetry_score(
    series
):
    """
    Measure robust distribution asymmetry using
    the median and the 10th/90th percentiles.

    Unlike conventional moment skewness, this
    statistic is much less sensitive to a handful
    of extreme values.

    This is the primary skew-related signal used
    by the histogram recommendation model.
    """

    clean = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    if len(clean) < 5:
        return (
            0,
            0
        )

    q10 = clean.quantile(
        0.10
    )

    median = clean.median()

    q90 = clean.quantile(
        0.90
    )

    lower_tail = (
        median
        - q10
    )

    upper_tail = (
        q90
        - median
    )

    total_tail = (
        lower_tail
        + upper_tail
    )

    if (
        pd.isna(
            total_tail
        )
        or total_tail <= 0
    ):
        return (
            0,
            0
        )

    asymmetry = (
        abs(
            upper_tail
            - lower_tail
        )
        / total_tail
    )

    asymmetry = float(
        asymmetry
    )

    score = piecewise_score(
        asymmetry,
        [
            (0.00, 0),
            (0.05, 0),
            (0.12, 15),
            (0.24, 30),
            (0.39, 50),
            (0.52, 68),
            (0.70, 80),
            (0.85, 88),
            (1.00, 92)
        ]
    )

    return (
        clamp(
            score
        ),
        asymmetry
    )


def multimodality_score(
    series
):
    """
    Detect strong two-mode structure using a
    one-component versus two-component Gaussian
    mixture comparison.

    The detector considers three pieces of evidence:

        1. BIC improvement from one Gaussian
           to two Gaussians

        2. Separation between the two fitted
           component means relative to their
           within-component spread

        3. Whether both components contain a
           meaningful share of observations

    Requiring all three helps distinguish genuine
    bimodality from:

        - ordinary Gaussian variation
        - a small group of extreme outliers
        - a one-sided skewed tail

    The calculation is performed on at most
    MULTIMODALITY_MAX_SAMPLE observations for
    predictable runtime.
    """

    clean = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
    )

    default_statistics = {
        "score": 0.0,
        "bic_improvement": 0.0,
        "component_separation": 0.0,
        "smaller_component_weight": 0.0
    }

    if (
        len(clean) < 100
        or clean.nunique() < 20
    ):
        return default_statistics

    if (
        len(clean)
        > MULTIMODALITY_MAX_SAMPLE
    ):

        model_data = (
            clean.sample(
                n=MULTIMODALITY_MAX_SAMPLE,
                random_state=42
            )
        )

    else:

        model_data = clean

    values = (
        model_data
        .to_numpy(
            dtype=float
        )
        .reshape(
            -1,
            1
        )
    )

    standard_deviation = float(
        np.std(
            values
        )
    )

    if (
        not math.isfinite(
            standard_deviation
        )
        or standard_deviation <= 0
    ):
        return default_statistics

    standardized = (
        (
            values
            - np.mean(
                values
            )
        )
        / standard_deviation
    )

    try:

        one_component = (
            GaussianMixture(
                n_components=1,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=42
            )
            .fit(
                standardized
            )
        )

        two_components = (
            GaussianMixture(
                n_components=2,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=42,
                n_init=3
            )
            .fit(
                standardized
            )
        )

    except (
        ValueError,
        np.linalg.LinAlgError
    ):

        return default_statistics

    bic_improvement = (
        one_component.bic(
            standardized
        )
        - two_components.bic(
            standardized
        )
    )

    means = (
        two_components
        .means_
        .reshape(
            -1
        )
    )

    variances = (
        two_components
        .covariances_
        .reshape(
            -1
        )
    )

    weights = (
        two_components
        .weights_
        .reshape(
            -1
        )
    )

    order = np.argsort(
        means
    )

    means = means[
        order
    ]

    variances = variances[
        order
    ]

    weights = weights[
        order
    ]

    pooled_within_scale = math.sqrt(
        max(
            (
                float(
                    variances[0]
                )
                + float(
                    variances[1]
                )
            )
            / 2,
            0
        )
    )

    if (
        pooled_within_scale <= 0
        or not math.isfinite(
            pooled_within_scale
        )
    ):

        component_separation = (
            0.0
        )

    else:

        component_separation = (
            abs(
                float(
                    means[1]
                    - means[0]
                )
            )
            / pooled_within_scale
        )

    smaller_component_weight = (
        float(
            np.min(
                weights
            )
        )
    )

    bic_score = piecewise_score(
        bic_improvement,
        [
            (0, 0),
            (10, 10),
            (30, 30),
            (75, 55),
            (150, 75),
            (300, 90),
            (600, 97),
            (1000, 100)
        ]
    )

    separation_score = piecewise_score(
        component_separation,
        [
            (0.0, 0),
            (1.5, 0),
            (2.0, 15),
            (2.5, 35),
            (3.0, 55),
            (4.0, 75),
            (5.0, 88),
            (6.0, 95),
            (7.0, 100)
        ]
    )

    balance_score = piecewise_score(
        smaller_component_weight,
        [
            (0.00, 0),
            (0.10, 0),
            (0.15, 25),
            (0.20, 50),
            (0.30, 80),
            (0.40, 95),
            (0.48, 100)
        ]
    )

    if (
        bic_score <= 0
        or separation_score <= 0
        or balance_score <= 0
    ):

        raw_score = 0.0

    else:

        raw_score = (
            bic_score
            * separation_score
            * balance_score
        ) ** (
            1 / 3
        )

    score = clamp(
        raw_score
        * MULTIMODALITY_SCALE
    )

    return {
        "score": float(
            score
        ),
        "bic_improvement": float(
            bic_improvement
        ),
        "component_separation": float(
            component_separation
        ),
        "smaller_component_weight": float(
            smaller_component_weight
        )
    }


def distribution_signal_score(
    series
):
    """
    Combine histogram-specific distribution
    evidence into one calibrated signal.

    Previous versions used an additive weighted
    combination of:

        skewness
        outlier prevalence
        tail asymmetry

    Those measures are often different symptoms of
    the same long-tail phenomenon. Adding them can
    therefore count one underlying feature multiple
    times and cause scores to saturate too quickly.

    The calibrated model instead builds three
    distinct signal families:

        shape asymmetry
        outlier prevalence
        multimodality

    The final distribution signal is the strongest
    of those distinct families.

    Classical moment skewness is still reported as
    descriptive evidence, but robust tail asymmetry
    drives the asymmetry recommendation signal.
    """

    (
        moment_skew_score,
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
        asymmetry_signal,
        asymmetry
    ) = tail_asymmetry_score(
        series
    )

    multimodality = (
        multimodality_score(
            series
        )
    )

    # Robust quantile asymmetry is intentionally
    # used as the actual shape signal. Moment
    # skewness is retained as evidence because it
    # can be distorted by only a few extreme values.
    shape_signal = (
        asymmetry_signal
    )

    signal = max(
        shape_signal,
        outlier_signal,
        multimodality[
            "score"
        ]
    )

    return {
        "signal": float(
            clamp(
                signal
            )
        ),

        "skewness": float(
            skewness
        ),

        "skew_score": float(
            moment_skew_score
        ),

        "outlier_rate": float(
            outlier_rate
        ),

        "outlier_score": float(
            outlier_signal
        ),

        "tail_asymmetry": float(
            asymmetry
        ),

        "tail_asymmetry_score": float(
            asymmetry_signal
        ),

        "shape_score": float(
            shape_signal
        ),

        "multimodality_score": float(
            multimodality[
                "score"
            ]
        ),

        "multimodality_bic_improvement": float(
            multimodality[
                "bic_improvement"
            ]
        ),

        "multimodality_component_separation": float(
            multimodality[
                "component_separation"
            ]
        ),

        "multimodality_smaller_component_weight": float(
            multimodality[
                "smaller_component_weight"
            ]
        )
    }


# ==================================================
# FINAL SCORE
# ==================================================

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
        + 0.65
        * (
            signal
            / 100
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


# ==================================================
# HISTOGRAM SCORER
# ==================================================

def score_histogram(
    df,
    candidate,
    profile
):
    """
    Score one histogram candidate from 0 to 100.
    """

    x = candidate[
        "x"
    ]

    semantic_type = (
        get_semantic_type(
            profile,
            x
        )
    )

    semantic_fit = (
        semantic_fit_score(
            semantic_type
        )
    )

    clean = (
        pd.to_numeric(
            df[x],
            errors="coerce"
        )
        .dropna()
    )

    observation_count = (
        len(
            clean
        )
    )

    unique_count = (
        clean.nunique()
    )

    quality = (
        data_quality_score(
            df,
            x
        )
    )

    support = (
        sample_support_score(
            observation_count
        )
    )

    readability = (
        readability_score(
            unique_count
        )
    )

    distribution = (
        distribution_signal_score(
            clean
        )
    )

    signal = (
        distribution[
            "signal"
        ]
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

    if clean.empty:

        minimum = None
        maximum = None
        mean = None
        median = None

    else:

        minimum = (
            clean.min()
        )

        maximum = (
            clean.max()
        )

        mean = (
            clean.mean()
        )

        median = (
            clean.median()
        )

    reasons = [
        (
            f"Semantic type: "
            f"{semantic_type}."
        ),
        (
            "Semantic fit for histogram: "
            f"{semantic_fit:.1f}/100."
        ),
        (
            f"{observation_count} complete "
            "numeric observations."
        ),
        (
            f"{unique_count} distinct "
            "numeric values."
        ),
        (
            "Data completeness: "
            f"{quality:.1f}%."
        ),
        (
            "Moment skewness: "
            f"{distribution['skewness']:.3f}."
        ),
        (
            "Potential outlier rate: "
            f"{distribution['outlier_rate'] * 100:.1f}%."
        ),
        (
            "Robust tail asymmetry: "
            f"{distribution['tail_asymmetry']:.3f}."
        ),
        (
            "Multimodality signal: "
            f"{distribution['multimodality_score']:.1f}/100."
        ),
        (
            "Distribution signal: "
            f"{signal:.1f}/100."
        )
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
            "observations":
                observation_count,

            "unique_values": int(
                unique_count
            ),

            "min": (
                round(
                    float(
                        minimum
                    ),
                    4
                )
                if minimum
                is not None
                else None
            ),

            "max": (
                round(
                    float(
                        maximum
                    ),
                    4
                )
                if maximum
                is not None
                else None
            ),

            "mean": (
                round(
                    float(
                        mean
                    ),
                    4
                )
                if mean
                is not None
                else None
            ),

            "median": (
                round(
                    float(
                        median
                    ),
                    4
                )
                if median
                is not None
                else None
            ),

            "skewness": round(
                float(
                    distribution[
                        "skewness"
                    ]
                ),
                4
            ),

            "outlier_rate": round(
                float(
                    distribution[
                        "outlier_rate"
                    ]
                ),
                4
            ),

            "tail_asymmetry": round(
                float(
                    distribution[
                        "tail_asymmetry"
                    ]
                ),
                4
            ),

            "shape_score": round(
                float(
                    distribution[
                        "shape_score"
                    ]
                ),
                2
            ),

            "outlier_score": round(
                float(
                    distribution[
                        "outlier_score"
                    ]
                ),
                2
            ),

            "multimodality_score": round(
                float(
                    distribution[
                        "multimodality_score"
                    ]
                ),
                2
            ),

            "multimodality_bic_improvement": round(
                float(
                    distribution[
                        "multimodality_bic_improvement"
                    ]
                ),
                4
            ),

            "multimodality_component_separation": round(
                float(
                    distribution[
                        "multimodality_component_separation"
                    ]
                ),
                4
            ),

            "multimodality_smaller_component_weight": round(
                float(
                    distribution[
                        "multimodality_smaller_component_weight"
                    ]
                ),
                4
            )
        },

        "reasons":
            reasons
    }
