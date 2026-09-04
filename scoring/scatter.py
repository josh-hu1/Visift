import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr
)

from sklearn.feature_selection import (
    mutual_info_regression
)


# ==================================================
# CONFIGURATION
# ==================================================

MI_PERMUTATIONS = 8
MI_MAX_SAMPLE_SIZE = 1000
MI_MINIMUM_EFFECT = 0.10
MI_NULL_STD_MULTIPLIER = 3.0
MI_RANDOM_STATE = 42

# Correlation reliability calibration.
#
# Correlations with p <= 0.001 receive essentially
# full statistical reliability.
#
# Correlations with p >= 0.10 retain only 35% of
# their raw signal.
#
# Values between these thresholds are interpolated
# on a logarithmic p-value scale.

CORRELATION_STRONG_P = 0.001
CORRELATION_WEAK_P = 0.10
CORRELATION_MIN_RELIABILITY = 0.35


# ==================================================
# GENERAL HELPERS
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


def clean_numeric_pairs(
    df,
    x,
    y
):
    """
    Return finite numeric x/y observations.
    """

    clean = (
        df[[x, y]]
        .copy()
    )

    clean[x] = pd.to_numeric(
        clean[x],
        errors="coerce"
    )

    clean[y] = pd.to_numeric(
        clean[y],
        errors="coerce"
    )

    clean = clean.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    clean = clean.dropna()

    return clean


# ==================================================
# VISUALIZATION QUALITY
# ==================================================

def semantic_fit_score(
    x_type,
    y_type
):
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

    if (
        x_type in continuous_types
        and y_type in continuous_types
    ):

        return 100

    if (
        x_type in numeric_types
        and y_type in numeric_types
    ):

        return 80

    return 0


def data_quality_score(
    df,
    x,
    y
):
    """
    Score the percentage of rows containing
    finite x/y observations.
    """

    if len(df) == 0:

        return 0

    clean = clean_numeric_pairs(
        df,
        x,
        y
    )

    return (
        len(clean)
        / len(df)
    ) * 100


def sample_support_score(
    n
):
    """
    Score whether there are enough paired
    observations for a useful scatterplot.
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


def variance_score(
    series
):
    """
    Penalize variables with little or no variation.
    """

    clean = (
        series.dropna()
    )

    if clean.empty:

        return 0

    unique_count = (
        clean.nunique()
    )

    if unique_count <= 1:
        return 0

    if unique_count <= 3:
        return 30

    if unique_count <= 5:
        return 50

    if unique_count <= 10:
        return 70

    return 100


def readability_score(
    df,
    x,
    y
):
    """
    Estimate whether the variables provide enough
    distinct values for a useful scatterplot.
    """

    clean = clean_numeric_pairs(
        df,
        x,
        y
    )

    if clean.empty:

        return 0

    x_score = variance_score(
        clean[x]
    )

    y_score = variance_score(
        clean[y]
    )

    return (
        x_score
        + y_score
    ) / 2


# ==================================================
# CORRELATION RELIABILITY
# ==================================================

def correlation_reliability_score(
    p_value
):
    """
    Convert a correlation p-value into a
    0-1 reliability multiplier.

    Why logarithmic interpolation?

    P-values change by orders of magnitude.
    Treating 0.001, 0.01, and 0.10 as linearly
    spaced would not reflect that difference well.

    Approximate behavior:

        p <= .001  -> 1.00 reliability
        p =  .01   -> ~0.68 reliability
        p =  .05   -> ~0.45 reliability
        p >= .10   -> 0.35 reliability

    The minimum is deliberately not zero because
    statistical significance is not the same thing
    as effect size, and small exploratory datasets
    can still contain useful relationships.
    """

    if (
        p_value is None
        or not np.isfinite(
            p_value
        )
    ):

        return (
            CORRELATION_MIN_RELIABILITY
        )

    if (
        p_value
        <= CORRELATION_STRONG_P
    ):

        return 1.0

    if (
        p_value
        >= CORRELATION_WEAK_P
    ):

        return (
            CORRELATION_MIN_RELIABILITY
        )

    log_strong = (
        np.log10(
            CORRELATION_STRONG_P
        )
    )

    log_weak = (
        np.log10(
            CORRELATION_WEAK_P
        )
    )

    log_p = (
        np.log10(
            p_value
        )
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

    position = max(
        0,
        min(
            position,
            1
        )
    )

    reliability = (
        CORRELATION_MIN_RELIABILITY
        + (
            1
            - CORRELATION_MIN_RELIABILITY
        )
        * position
    )

    return float(
        reliability
    )


# ==================================================
# CORRELATION SIGNAL
# ==================================================

def correlation_signal_score(
    clean,
    x,
    y
):
    """
    Measure linear and monotonic relationships while
    accounting for statistical reliability.

    Pearson:
        linear association

    Spearman:
        monotonic association

    The stronger absolute correlation is selected,
    then calibrated using its corresponding p-value.

    This helps distinguish:

        strong relationship + small n
            from
        moderate chance correlation + small n
    """

    if len(clean) < 3:

        return {
            "signal": 0,
            "raw_signal": 0,
            "pearson": 0,
            "spearman": 0,
            "pearson_p": 1,
            "spearman_p": 1,
            "selected_p": 1,
            "reliability": 0,
            "selected_method": "none"
        }

    if (
        clean[x].nunique() <= 1
        or clean[y].nunique() <= 1
    ):

        return {
            "signal": 0,
            "raw_signal": 0,
            "pearson": 0,
            "spearman": 0,
            "pearson_p": 1,
            "spearman_p": 1,
            "selected_p": 1,
            "reliability": 0,
            "selected_method": "none"
        }

    x_values = (
        clean[x]
        .to_numpy(
            dtype=float
        )
    )

    y_values = (
        clean[y]
        .to_numpy(
            dtype=float
        )
    )

    # -----------------------------------
    # Pearson
    # -----------------------------------

    try:

        pearson_result = pearsonr(
            x_values,
            y_values
        )

        pearson = float(
            pearson_result.statistic
        )

        pearson_p = float(
            pearson_result.pvalue
        )

    except Exception:

        pearson = 0
        pearson_p = 1

    # -----------------------------------
    # Spearman
    # -----------------------------------

    try:

        spearman_result = spearmanr(
            x_values,
            y_values
        )

        spearman = float(
            spearman_result.statistic
        )

        spearman_p = float(
            spearman_result.pvalue
        )

    except Exception:

        spearman = 0
        spearman_p = 1

    if not np.isfinite(
        pearson
    ):

        pearson = 0

    if not np.isfinite(
        spearman
    ):

        spearman = 0

    if not np.isfinite(
        pearson_p
    ):

        pearson_p = 1

    if not np.isfinite(
        spearman_p
    ):

        spearman_p = 1

    # -----------------------------------
    # Select strongest relationship
    # -----------------------------------

    if (
        abs(
            pearson
        )
        >= abs(
            spearman
        )
    ):

        strongest = (
            pearson
        )

        selected_p = (
            pearson_p
        )

        selected_method = (
            "pearson"
        )

    else:

        strongest = (
            spearman
        )

        selected_p = (
            spearman_p
        )

        selected_method = (
            "spearman"
        )

    raw_signal = (
        abs(
            strongest
        )
        * 100
    )

    reliability = (
        correlation_reliability_score(
            selected_p
        )
    )

    calibrated_signal = (
        raw_signal
        * reliability
    )

    calibrated_signal = clamp(
        calibrated_signal
    )

    return {
        "signal": float(
            calibrated_signal
        ),

        "raw_signal": float(
            raw_signal
        ),

        "pearson": float(
            pearson
        ),

        "spearman": float(
            spearman
        ),

        "pearson_p": float(
            pearson_p
        ),

        "spearman_p": float(
            spearman_p
        ),

        "selected_p": float(
            selected_p
        ),

        "reliability": float(
            reliability
        ),

        "selected_method": (
            selected_method
        )
    }


# ==================================================
# MUTUAL INFORMATION
# ==================================================

def calculate_one_way_mutual_information(
    feature,
    target,
    random_state
):
    """
    Estimate continuous mutual information for:

        feature -> target
    """

    feature = np.asarray(
        feature,
        dtype=float
    )

    target = np.asarray(
        target,
        dtype=float
    )

    result = (
        mutual_info_regression(
            feature.reshape(
                -1,
                1
            ),
            target,
            discrete_features=False,
            random_state=random_state
        )
    )

    return float(
        result[0]
    )


def symmetric_mutual_information(
    x_values,
    y_values,
    random_state
):
    """
    Estimate MI in both directions and average
    because a scatterplot is symmetric.
    """

    x_to_y = (
        calculate_one_way_mutual_information(
            x_values,
            y_values,
            random_state
        )
    )

    y_to_x = (
        calculate_one_way_mutual_information(
            y_values,
            x_values,
            random_state + 1
        )
    )

    return (
        x_to_y
        + y_to_x
    ) / 2


def nonlinear_signal_score(
    clean,
    x,
    y
):
    """
    Estimate nonlinear dependence using
    permutation-calibrated mutual information.

    Raw MI is compared against a shuffled-data
    null distribution before being converted to
    a bounded nonlinear signal.
    """

    default_result = {
        "signal": 0,
        "observed_mi": 0,
        "null_mean": 0,
        "null_std": 0,
        "threshold": 0,
        "excess_mi": 0,
        "sample_size": len(
            clean
        )
    }

    if len(clean) < 20:

        return default_result

    if (
        clean[x].nunique() <= 5
        or clean[y].nunique() <= 5
    ):

        return default_result

    # -----------------------------------
    # Limit MI computation cost
    # -----------------------------------

    if (
        len(clean)
        > MI_MAX_SAMPLE_SIZE
    ):

        mi_data = clean.sample(
            n=MI_MAX_SAMPLE_SIZE,
            random_state=(
                MI_RANDOM_STATE
            )
        )

    else:

        mi_data = (
            clean.copy()
        )

    x_values = (
        mi_data[x]
        .to_numpy(
            dtype=float
        )
    )

    y_values = (
        mi_data[y]
        .to_numpy(
            dtype=float
        )
    )

    # -----------------------------------
    # Observed MI
    # -----------------------------------

    observed_mi = (
        symmetric_mutual_information(
            x_values,
            y_values,
            MI_RANDOM_STATE
        )
    )

    # -----------------------------------
    # Null distribution
    # -----------------------------------

    rng = (
        np.random.default_rng(
            MI_RANDOM_STATE
        )
    )

    null_values = []

    for permutation_index in range(
        MI_PERMUTATIONS
    ):

        shuffled_y = (
            rng.permutation(
                y_values
            )
        )

        null_mi = (
            symmetric_mutual_information(
                x_values,
                shuffled_y,
                (
                    MI_RANDOM_STATE
                    + 100
                    + (
                        permutation_index
                        * 2
                    )
                )
            )
        )

        null_values.append(
            null_mi
        )

    null_values = np.asarray(
        null_values,
        dtype=float
    )

    null_mean = float(
        null_values.mean()
    )

    null_std = float(
        null_values.std(
            ddof=0
        )
    )

    statistical_threshold = (
        null_mean
        + (
            MI_NULL_STD_MULTIPLIER
            * null_std
        )
    )

    threshold = max(
        MI_MINIMUM_EFFECT,
        statistical_threshold
    )

    excess_mi = max(
        0,
        observed_mi
        - threshold
    )

    # -----------------------------------
    # MI -> correlation-like scale
    # -----------------------------------

    if excess_mi <= 0:

        signal = 0

    else:

        equivalent_dependence = (
            np.sqrt(
                1
                - np.exp(
                    -2
                    * excess_mi
                )
            )
        )

        signal = (
            equivalent_dependence
            * 100
        )

    signal = clamp(
        signal
    )

    return {
        "signal": float(
            signal
        ),

        "observed_mi": float(
            observed_mi
        ),

        "null_mean": float(
            null_mean
        ),

        "null_std": float(
            null_std
        ),

        "threshold": float(
            threshold
        ),

        "excess_mi": float(
            excess_mi
        ),

        "sample_size": len(
            mi_data
        )
    }


# ==================================================
# COMBINED RELATIONSHIP SIGNAL
# ==================================================

def relationship_signal_score(
    df,
    x,
    y
):
    """
    Estimate overall relationship strength.

    Linear / monotonic relationships are calibrated
    using statistical reliability.

    Nonlinear relationships retain the existing
    permutation-calibrated MI detector.

    The strongest supported form of dependence wins.
    """

    clean = clean_numeric_pairs(
        df,
        x,
        y
    )

    if len(clean) < 3:

        return {
            "signal": 0,
            "linear_signal": 0,
            "raw_linear_signal": 0,
            "linear_reliability": 0,
            "pearson": 0,
            "spearman": 0,
            "pearson_p": 1,
            "spearman_p": 1,
            "selected_correlation_p": 1,
            "selected_correlation_method": "none",
            "nonlinear_signal": 0,
            "mutual_information": 0,
            "mi_null_mean": 0,
            "mi_null_std": 0,
            "mi_threshold": 0,
            "mi_excess": 0,
            "mi_sample_size": len(
                clean
            )
        }

    correlation = (
        correlation_signal_score(
            clean,
            x,
            y
        )
    )

    nonlinear = (
        nonlinear_signal_score(
            clean,
            x,
            y
        )
    )

    linear_signal = (
        correlation[
            "signal"
        ]
    )

    nonlinear_signal = (
        nonlinear[
            "signal"
        ]
    )

    signal = max(
        linear_signal,
        nonlinear_signal
    )

    signal = clamp(
        signal
    )

    return {
        "signal": float(
            signal
        ),

        "linear_signal": float(
            linear_signal
        ),

        "raw_linear_signal": float(
            correlation[
                "raw_signal"
            ]
        ),

        "linear_reliability": float(
            correlation[
                "reliability"
            ]
        ),

        "pearson": float(
            correlation[
                "pearson"
            ]
        ),

        "spearman": float(
            correlation[
                "spearman"
            ]
        ),

        "pearson_p": float(
            correlation[
                "pearson_p"
            ]
        ),

        "spearman_p": float(
            correlation[
                "spearman_p"
            ]
        ),

        "selected_correlation_p": float(
            correlation[
                "selected_p"
            ]
        ),

        "selected_correlation_method": (
            correlation[
                "selected_method"
            ]
        ),

        "nonlinear_signal": float(
            nonlinear_signal
        ),

        "mutual_information": float(
            nonlinear[
                "observed_mi"
            ]
        ),

        "mi_null_mean": float(
            nonlinear[
                "null_mean"
            ]
        ),

        "mi_null_std": float(
            nonlinear[
                "null_std"
            ]
        ),

        "mi_threshold": float(
            nonlinear[
                "threshold"
            ]
        ),

        "mi_excess": float(
            nonlinear[
                "excess_mi"
            ]
        ),

        "mi_sample_size": (
            nonlinear[
                "sample_size"
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
# SCATTER SCORER
# ==================================================

def score_scatter_chart(
    df,
    candidate,
    profile
):
    """
    Score one scatterplot candidate from 0 to 100.
    """

    x = (
        candidate["x"]
    )

    y = (
        candidate["y"]
    )

    clean = clean_numeric_pairs(
        df,
        x,
        y
    )

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

    quality = (
        data_quality_score(
            df,
            x,
            y
        )
    )

    support = (
        sample_support_score(
            len(
                clean
            )
        )
    )

    readability = (
        readability_score(
            df,
            x,
            y
        )
    )

    relationship = (
        relationship_signal_score(
            df,
            x,
            y
        )
    )

    signal = (
        relationship[
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

    reasons = [
        (
            f"X semantic type: "
            f"{x_type}."
        ),

        (
            f"Y semantic type: "
            f"{y_type}."
        ),

        (
            "Semantic fit for scatterplot: "
            f"{semantic_type_display(semantic_fit)}."
        ),

        (
            f"{len(clean)} complete paired "
            "observations."
        ),

        (
            f"Data completeness: "
            f"{quality:.1f}%."
        ),

        (
            f"Pearson correlation: "
            f"{relationship['pearson']:.3f}."
        ),

        (
            f"Pearson p-value: "
            f"{relationship['pearson_p']:.4g}."
        ),

        (
            f"Spearman correlation: "
            f"{relationship['spearman']:.3f}."
        ),

        (
            f"Spearman p-value: "
            f"{relationship['spearman_p']:.4g}."
        ),

        (
            "Raw correlation signal: "
            f"{relationship['raw_linear_signal']:.1f}/100."
        ),

        (
            "Correlation reliability: "
            f"{relationship['linear_reliability'] * 100:.1f}%."
        ),

        (
            "Reliability-adjusted correlation signal: "
            f"{relationship['linear_signal']:.1f}/100."
        ),

        (
            "Mutual information: "
            f"{relationship['mutual_information']:.3f}."
        ),

        (
            "Permutation-calibrated nonlinear "
            f"signal: "
            f"{relationship['nonlinear_signal']:.1f}/100."
        ),

        (
            f"Relationship signal: "
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
            "pearson": round(
                relationship[
                    "pearson"
                ],
                4
            ),

            "spearman": round(
                relationship[
                    "spearman"
                ],
                4
            ),

            "pearson_p": round(
                relationship[
                    "pearson_p"
                ],
                6
            ),

            "spearman_p": round(
                relationship[
                    "spearman_p"
                ],
                6
            ),

            "selected_correlation_method": (
                relationship[
                    "selected_correlation_method"
                ]
            ),

            "selected_correlation_p": round(
                relationship[
                    "selected_correlation_p"
                ],
                6
            ),

            "raw_linear_signal": round(
                relationship[
                    "raw_linear_signal"
                ],
                2
            ),

            "linear_reliability": round(
                relationship[
                    "linear_reliability"
                ],
                4
            ),

            "linear_signal": round(
                relationship[
                    "linear_signal"
                ],
                2
            ),

            "mutual_information": round(
                relationship[
                    "mutual_information"
                ],
                4
            ),

            "mi_null_mean": round(
                relationship[
                    "mi_null_mean"
                ],
                4
            ),

            "mi_null_std": round(
                relationship[
                    "mi_null_std"
                ],
                4
            ),

            "mi_threshold": round(
                relationship[
                    "mi_threshold"
                ],
                4
            ),

            "mi_excess": round(
                relationship[
                    "mi_excess"
                ],
                4
            ),

            "nonlinear_signal": round(
                relationship[
                    "nonlinear_signal"
                ],
                2
            ),

            "mi_sample_size": (
                relationship[
                    "mi_sample_size"
                ]
            )
        },

        "reasons": (
            reasons
        )
    }


# ==================================================
# DISPLAY
# ==================================================

def semantic_type_display(
    score
):
    """
    Format semantic fit for output.
    """

    return (
        f"{score:.1f}/100"
    )