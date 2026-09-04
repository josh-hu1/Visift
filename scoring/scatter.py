import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression


# ==================================================
# CONFIGURATION
# ==================================================

# Mutual information is only used as an additional
# nonlinear relationship detector.
#
# These settings intentionally make the detector
# conservative so random noise is unlikely to receive
# a high nonlinear signal.

MI_PERMUTATIONS = 8
MI_MAX_SAMPLE_SIZE = 1000
MI_MINIMUM_EFFECT = 0.10
MI_NULL_STD_MULTIPLIER = 3.0
MI_RANDOM_STATE = 42


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

    NaN and infinite values are removed before
    statistical calculations.
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

    # Ideal scatterplot:
    # continuous numeric vs continuous numeric

    if (
        x_type in continuous_types
        and y_type in continuous_types
    ):

        return 100

    # Still valid if one variable is
    # discrete numeric.

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

    clean = series.dropna()

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
# CORRELATION SIGNAL
# ==================================================

def correlation_signal_score(
    clean,
    x,
    y
):
    """
    Measure linear and monotonic relationships.

    Pearson:
        Linear association.

    Spearman:
        Monotonic association.

    The strongest absolute correlation is converted
    directly to a 0-100 signal.
    """

    if len(clean) < 3:

        return (
            0,
            0,
            0
        )

    if (
        clean[x].nunique() <= 1
        or clean[y].nunique() <= 1
    ):

        return (
            0,
            0,
            0
        )

    pearson = clean[x].corr(
        clean[y],
        method="pearson"
    )

    spearman = clean[x].corr(
        clean[y],
        method="spearman"
    )

    if pd.isna(
        pearson
    ):

        pearson = 0

    if pd.isna(
        spearman
    ):

        spearman = 0

    strongest_relationship = max(
        abs(pearson),
        abs(spearman)
    )

    signal = (
        strongest_relationship
        * 100
    )

    return (
        signal,
        float(pearson),
        float(spearman)
    )


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

    using scikit-learn's nearest-neighbor
    mutual information estimator.
    """

    feature = np.asarray(
        feature,
        dtype=float
    )

    target = np.asarray(
        target,
        dtype=float
    )

    result = mutual_info_regression(
        feature.reshape(
            -1,
            1
        ),
        target,
        discrete_features=False,
        random_state=random_state
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
    Mutual information estimation is directional
    because one variable is treated as the feature
    and the other as the target.

    A scatterplot is symmetric, so estimate MI in
    both directions and average the results.
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

    Why permutation calibration?

    Raw mutual information is not naturally bounded
    between 0 and 1 and may contain positive estimator
    bias even when two variables are independent.

    Procedure:

    1. Estimate the observed mutual information.
    2. Shuffle y several times to destroy the
       relationship.
    3. Estimate the MI expected under independence.
    4. Require observed MI to exceed both:
         - an absolute minimum effect threshold
         - the null mean + 3 standard deviations
    5. Convert only the excess mutual information
       into a bounded 0-100 signal.

    This intentionally favors false-negative
    protection over aggressively detecting weak
    nonlinear patterns.
    """

    if len(clean) < 20:

        return {
            "signal": 0,
            "observed_mi": 0,
            "null_mean": 0,
            "null_std": 0,
            "threshold": 0,
            "excess_mi": 0,
            "sample_size": len(clean)
        }

    if (
        clean[x].nunique() <= 5
        or clean[y].nunique() <= 5
    ):

        return {
            "signal": 0,
            "observed_mi": 0,
            "null_mean": 0,
            "null_std": 0,
            "threshold": 0,
            "excess_mi": 0,
            "sample_size": len(clean)
        }

    # -----------------------------------
    # Limit MI computation on very
    # large datasets.
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

    # -----------------------------------
    # Conservative detection threshold
    # -----------------------------------

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
    # Convert MI to correlation-like scale
    # -----------------------------------
    #
    # For a bivariate Gaussian:
    #
    # MI = -0.5 * ln(1 - rho^2)
    #
    # Rearranging gives:
    #
    # |rho| = sqrt(1 - exp(-2 * MI))
    #
    # We apply the same transformation to the
    # excess MI to obtain a bounded,
    # correlation-like nonlinear signal.
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

    Three forms of dependence are considered:

    Pearson:
        linear relationships

    Spearman:
        monotonic relationships

    Mutual information:
        nonlinear relationships

    The final scatter relationship signal uses the
    strongest supported form of dependence.
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
            "nonlinear_signal": 0,
            "pearson": 0,
            "spearman": 0,
            "mutual_information": 0,
            "mi_null_mean": 0,
            "mi_null_std": 0,
            "mi_threshold": 0,
            "mi_excess": 0,
            "mi_sample_size": len(clean)
        }

    if (
        clean[x].nunique() <= 1
        or clean[y].nunique() <= 1
    ):

        return {
            "signal": 0,
            "linear_signal": 0,
            "nonlinear_signal": 0,
            "pearson": 0,
            "spearman": 0,
            "mutual_information": 0,
            "mi_null_mean": 0,
            "mi_null_std": 0,
            "mi_threshold": 0,
            "mi_excess": 0,
            "mi_sample_size": len(clean)
        }

    (
        linear_signal,
        pearson,
        spearman
    ) = correlation_signal_score(
        clean,
        x,
        y
    )

    nonlinear = (
        nonlinear_signal_score(
            clean,
            x,
            y
        )
    )

    nonlinear_signal = (
        nonlinear["signal"]
    )

    # Conservative combination:
    #
    # Do not add the signals together, which would
    # double-count related evidence.
    #
    # Instead use whichever relationship detector
    # provides stronger evidence.

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

        "nonlinear_signal": float(
            nonlinear_signal
        ),

        "pearson": float(
            pearson
        ),

        "spearman": float(
            spearman
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

    x = candidate["x"]
    y = candidate["y"]

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
            len(clean)
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
    # Explainability
    # -----------------------------------

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
            f"Spearman correlation: "
            f"{relationship['spearman']:.3f}."
        ),
        (
            "Correlation-based signal: "
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

        "reasons": reasons
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