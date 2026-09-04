import numpy as np
import pandas as pd


# ==================================================
# SEMANTIC TYPES
# ==================================================

NUMERIC_TYPES = {
    "numeric_continuous",
    "numeric_discrete"
}

TIME_TYPES = {
    "datetime",
    "temporal"
}


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


# ==================================================
# SEMANTIC FIT
# ==================================================

def semantic_fit_score(
    x_type,
    y_type
):
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


# ==================================================
# TEMPORAL DATA PREPARATION
# ==================================================

def prepare_temporal_data(
    df,
    x,
    y,
    x_type
):
    """
    Prepare a temporal/numeric pair for line-chart
    analysis.

    Datetimes are converted into elapsed days.
    Numeric temporal fields such as years are kept
    as numeric values.

    Duplicate time points are aggregated by mean.
    """

    clean = (
        df[[x, y]]
        .copy()
    )

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

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    clean = clean.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    clean = clean.dropna(
        subset=[
            x,
            y
        ]
    )

    if clean.empty:

        return (
            clean,
            pd.DataFrame()
        )

    clean = clean.sort_values(
        x
    )

    # -----------------------------------
    # Convert time into a numeric axis
    # -----------------------------------

    if x_type == "datetime":

        first_time = (
            clean[x].min()
        )

        clean[
            "_time_numeric"
        ] = (
            (
                clean[x]
                - first_time
            )
            .dt.total_seconds()
            / 86400
        )

    else:

        clean[
            "_time_numeric"
        ] = (
            clean[x]
            .astype(float)
        )

    # -----------------------------------
    # Aggregate duplicate timestamps
    # -----------------------------------

    trend_data = (
        clean
        .groupby(
            "_time_numeric",
            as_index=False
        )[y]
        .mean()
        .sort_values(
            "_time_numeric"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        clean,
        trend_data
    )


# ==================================================
# VISUALIZATION QUALITY
# ==================================================

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
        complete_rows
        / total_rows
    )

    return (
        completeness
        * 100
    )


def observation_support_score(
    n
):
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


def time_point_support_score(
    unique_time_points
):
    """
    Score how many distinct temporal positions are
    available for detecting temporal patterns.
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
    repeated observations at only a few dates.
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


def readability_score(
    unique_time_points
):
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


# ==================================================
# DETRENDING
# ==================================================

def detrend_values(
    trend_data,
    y
):
    """
    Remove the best-fitting linear trend from the
    time series.

    Seasonality, level-shift, and volatility
    detectors operate primarily on these residuals
    so that a simple upward/downward trend is not
    mistaken for another temporal pattern.
    """

    if len(
        trend_data
    ) < 3:

        return np.array(
            [],
            dtype=float
        )

    time_values = (
        trend_data[
            "_time_numeric"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_values = (
        trend_data[y]
        .to_numpy(
            dtype=float
        )
    )

    if (
        np.std(
            time_values
        )
        == 0
    ):

        return (
            y_values
            - np.mean(
                y_values
            )
        )

    slope, intercept = (
        np.polyfit(
            time_values,
            y_values,
            1
        )
    )

    fitted = (
        slope
        * time_values
        + intercept
    )

    residuals = (
        y_values
        - fitted
    )

    return residuals


# ==================================================
# TREND SIGNAL
# ==================================================

def trend_signal_score(
    trend_data,
    y
):
    """
    Estimate monotonic temporal trend strength.

    Pearson:
        Linear trend.

    Spearman:
        Monotonic trend.

    The stronger absolute relationship is used.
    """

    if len(
        trend_data
    ) < 3:

        return {
            "signal": 0,
            "pearson": 0,
            "spearman": 0,
            "direction": "insufficient"
        }

    time_values = (
        trend_data[
            "_time_numeric"
        ]
    )

    y_values = (
        trend_data[y]
    )

    if (
        time_values.nunique()
        <= 1
        or y_values.nunique()
        <= 1
    ):

        return {
            "signal": 0,
            "pearson": 0,
            "spearman": 0,
            "direction": "none"
        }

    pearson = (
        time_values.corr(
            y_values,
            method="pearson"
        )
    )

    spearman = (
        time_values.corr(
            y_values,
            method="spearman"
        )
    )

    if pd.isna(
        pearson
    ):

        pearson = 0

    if pd.isna(
        spearman
    ):

        spearman = 0

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

    else:

        strongest = (
            spearman
        )

    signal = (
        abs(
            strongest
        )
        * 100
    )

    if strongest > 0.05:

        direction = (
            "upward"
        )

    elif strongest < -0.05:

        direction = (
            "downward"
        )

    else:

        direction = (
            "flat"
        )

    return {
        "signal": float(
            signal
        ),

        "pearson": float(
            pearson
        ),

        "spearman": float(
            spearman
        ),

        "direction": (
            direction
        )
    }


# ==================================================
# SAMPLING REGULARITY
# ==================================================

def temporal_sampling_regularity(
    trend_data
):
    """
    Estimate whether temporal observations are
    approximately evenly spaced.

    Frequency-based seasonality detection is only
    meaningful when spacing is reasonably regular.

    Returns:
        is_regular
        median_interval
        relative_interval_variation
    """

    if len(
        trend_data
    ) < 4:

        return {
            "is_regular": False,
            "median_interval": 0,
            "relative_variation": 0
        }

    time_values = (
        trend_data[
            "_time_numeric"
        ]
        .to_numpy(
            dtype=float
        )
    )

    differences = (
        np.diff(
            time_values
        )
    )

    differences = differences[
        differences > 0
    ]

    if len(
        differences
    ) == 0:

        return {
            "is_regular": False,
            "median_interval": 0,
            "relative_variation": 0
        }

    median_interval = float(
        np.median(
            differences
        )
    )

    if median_interval <= 0:

        return {
            "is_regular": False,
            "median_interval": 0,
            "relative_variation": 0
        }

    median_deviation = float(
        np.median(
            np.abs(
                differences
                - median_interval
            )
        )
    )

    relative_variation = (
        median_deviation
        / median_interval
    )

    is_regular = (
        relative_variation
        <= 0.10
    )

    return {
        "is_regular": bool(
            is_regular
        ),

        "median_interval": (
            median_interval
        ),

        "relative_variation": float(
            relative_variation
        )
    }


# ==================================================
# SEASONALITY SIGNAL
# ==================================================

def autocorrelation_at_lag(
    values,
    lag
):
    """
    Calculate correlation between a series and
    itself shifted by a specified lag.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if (
        lag <= 0
        or lag >= len(
            values
        )
    ):

        return 0

    left = (
        values[:-lag]
    )

    right = (
        values[lag:]
    )

    if (
        len(left) < 3
        or np.std(left) == 0
        or np.std(right) == 0
    ):

        return 0

    correlation = (
        np.corrcoef(
            left,
            right
        )[0, 1]
    )

    if not np.isfinite(
        correlation
    ):

        return 0

    return float(
        correlation
    )


def seasonality_signal_score(
    trend_data,
    residuals
):
    """
    Detect repeating temporal behavior using a
    combination of spectral concentration and
    autocorrelation.

    Procedure:

    1. Remove the linear trend.
    2. Verify approximately regular time spacing.
    3. Use an FFT to find the strongest repeating
       frequency.
    4. Estimate its period.
    5. Confirm the pattern using autocorrelation
       near that period.

    Requiring both spectral concentration and
    autocorrelation makes the detector conservative
    against ordinary white noise.
    """

    n = len(
        residuals
    )

    default_result = {
        "signal": 0,
        "peak_power_ratio": 0,
        "autocorrelation": 0,
        "period_observations": 0,
        "period_temporal_units": 0,
        "regular_sampling": False
    }

    if n < 20:

        return default_result

    if (
        len(
            trend_data
        )
        != n
    ):

        return default_result

    if (
        np.std(
            residuals
        )
        <= 1e-12
    ):

        return default_result

    regularity = (
        temporal_sampling_regularity(
            trend_data
        )
    )

    if not regularity[
        "is_regular"
    ]:

        return default_result

    centered = (
        residuals
        - np.mean(
            residuals
        )
    )

    # -----------------------------------
    # Frequency spectrum
    # -----------------------------------

    spectrum = (
        np.fft.rfft(
            centered
        )
    )

    power = (
        np.abs(
            spectrum
        )
        ** 2
    )

    frequencies = (
        np.fft.rfftfreq(
            n,
            d=1.0
        )
    )

    # Remove zero frequency.

    frequencies = (
        frequencies[1:]
    )

    power = (
        power[1:]
    )

    if len(
        power
    ) == 0:

        return default_result

    # -----------------------------------
    # Only consider periods where at
    # least two cycles are present and
    # each cycle spans at least 3 points.
    # -----------------------------------

    valid = (
        frequencies
        > 0
    )

    periods = np.zeros_like(
        frequencies
    )

    periods[
        valid
    ] = (
        1
        / frequencies[
            valid
        ]
    )

    valid = (
        valid
        & (
            periods >= 3
        )
        & (
            periods
            <= n / 2
        )
    )

    if not np.any(
        valid
    ):

        return default_result

    valid_power = (
        power[
            valid
        ]
    )

    valid_periods = (
        periods[
            valid
        ]
    )

    total_power = float(
        np.sum(
            power
        )
    )

    if total_power <= 0:

        return default_result

    best_index = int(
        np.argmax(
            valid_power
        )
    )

    peak_power = float(
        valid_power[
            best_index
        ]
    )

    period_observations = float(
        valid_periods[
            best_index
        ]
    )

    peak_power_ratio = (
        peak_power
        / total_power
    )

    # -----------------------------------
    # Confirm the frequency-domain signal
    # with autocorrelation.
    # -----------------------------------

    lag = int(
        round(
            period_observations
        )
    )

    lag = max(
        2,
        min(
            lag,
            n - 2
        )
    )

    autocorrelation = (
        autocorrelation_at_lag(
            centered,
            lag
        )
    )

    absolute_autocorrelation = abs(
        autocorrelation
    )

    # -----------------------------------
    # Conservative normalization
    # -----------------------------------

    # White noise generally has diffuse spectral
    # power. Require at least 8% of residual power
    # to concentrate at one repeating frequency.

    spectral_score = clamp(
        (
            peak_power_ratio
            - 0.08
        )
        / (
            0.35
            - 0.08
        )
        * 100
    )

    # Require meaningful repeated similarity at
    # the inferred lag.

    autocorrelation_score = clamp(
        (
            absolute_autocorrelation
            - 0.30
        )
        / (
            0.80
            - 0.30
        )
        * 100
    )

    # Both forms of evidence must be present.

    if (
        spectral_score <= 0
        or autocorrelation_score <= 0
    ):

        signal = 0

    else:

        signal = (
            np.sqrt(
                spectral_score
                * autocorrelation_score
            )
        )

    period_temporal_units = (
        period_observations
        * regularity[
            "median_interval"
        ]
    )

    return {
        "signal": float(
            clamp(
                signal
            )
        ),

        "peak_power_ratio": float(
            peak_power_ratio
        ),

        "autocorrelation": float(
            autocorrelation
        ),

        "period_observations": float(
            period_observations
        ),

        "period_temporal_units": float(
            period_temporal_units
        ),

        "regular_sampling": True
    }


# ==================================================
# LEVEL-SHIFT SIGNAL
# ==================================================

def pooled_standard_deviation(
    left,
    right
):
    """
    Calculate pooled standard deviation for two
    temporal segments.
    """

    left = np.asarray(
        left,
        dtype=float
    )

    right = np.asarray(
        right,
        dtype=float
    )

    if (
        len(left) < 2
        or len(right) < 2
    ):

        return 0

    left_variance = (
        np.var(
            left,
            ddof=1
        )
    )

    right_variance = (
        np.var(
            right,
            ddof=1
        )
    )

    numerator = (
        (
            len(left)
            - 1
        )
        * left_variance
        + (
            len(right)
            - 1
        )
        * right_variance
    )

    denominator = (
        len(left)
        + len(right)
        - 2
    )

    if denominator <= 0:

        return 0

    pooled_variance = (
        numerator
        / denominator
    )

    if pooled_variance <= 0:

        return 0

    return float(
        np.sqrt(
            pooled_variance
        )
    )


def level_shift_signal_score(
    residuals
):
    """
    Detect persistent changes in the temporal level.

    Several central split points are tested. For
    each split, the mean difference is standardized
    using pooled within-segment variation.

    Linear trend is removed beforehand so a simple
    slope is less likely to be mislabeled as a
    change point.
    """

    values = np.asarray(
        residuals,
        dtype=float
    )

    n = len(
        values
    )

    default_result = {
        "signal": 0,
        "effect_size": 0,
        "split_fraction": 0
    }

    if n < 20:

        return default_result

    split_fractions = [
        0.25,
        0.33,
        0.50,
        0.67,
        0.75
    ]

    strongest_effect = 0
    strongest_split = 0

    for split_fraction in (
        split_fractions
    ):

        split_index = int(
            round(
                n
                * split_fraction
            )
        )

        if (
            split_index < 8
            or (
                n
                - split_index
            ) < 8
        ):

            continue

        left = (
            values[
                :split_index
            ]
        )

        right = (
            values[
                split_index:
            ]
        )

        pooled_sd = (
            pooled_standard_deviation(
                left,
                right
            )
        )

        if pooled_sd <= 0:

            continue

        effect_size = abs(
            (
                np.mean(
                    right
                )
                - np.mean(
                    left
                )
            )
            / pooled_sd
        )

        if (
            effect_size
            > strongest_effect
        ):

            strongest_effect = (
                effect_size
            )

            strongest_split = (
                split_fraction
            )

    # Conservative effect-size mapping.
    #
    # |d| <= 0.50:
    #     treated as insufficient evidence.
    #
    # |d| >= 2:
    #     extremely strong persistent level shift.

    signal = clamp(
        (
            strongest_effect
            - 0.50
        )
        / (
            2.00
            - 0.50
        )
        * 100
    )

    return {
        "signal": float(
            signal
        ),

        "effect_size": float(
            strongest_effect
        ),

        "split_fraction": float(
            strongest_split
        )
    }


# ==================================================
# VOLATILITY-SHIFT SIGNAL
# ==================================================

def robust_scale(
    values
):
    """
    Estimate dispersion using the median absolute
    deviation.

    The 1.4826 factor makes MAD approximately
    comparable to standard deviation for normal data.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if len(
        values
    ) == 0:

        return 0

    median = np.median(
        values
    )

    mad = np.median(
        np.abs(
            values
            - median
        )
    )

    return float(
        1.4826
        * mad
    )


def volatility_shift_signal_score(
    residuals
):
    """
    Detect a sustained change in temporal variance.

    Robust dispersion is compared before and after
    several candidate split points.

    A dispersion ratio near 1 indicates similar
    volatility.

    A large ratio indicates a potential regime
    change in variability.
    """

    values = np.asarray(
        residuals,
        dtype=float
    )

    n = len(
        values
    )

    default_result = {
        "signal": 0,
        "dispersion_ratio": 1,
        "split_fraction": 0,
        "left_scale": 0,
        "right_scale": 0
    }

    if n < 30:

        return default_result

    split_fractions = [
        0.25,
        0.33,
        0.50,
        0.67,
        0.75
    ]

    strongest_ratio = 1
    strongest_split = 0
    strongest_left_scale = 0
    strongest_right_scale = 0

    epsilon = 1e-12

    for split_fraction in (
        split_fractions
    ):

        split_index = int(
            round(
                n
                * split_fraction
            )
        )

        if (
            split_index < 10
            or (
                n
                - split_index
            ) < 10
        ):

            continue

        left = (
            values[
                :split_index
            ]
        )

        right = (
            values[
                split_index:
            ]
        )

        left_scale = (
            robust_scale(
                left
            )
        )

        right_scale = (
            robust_scale(
                right
            )
        )

        larger = max(
            left_scale,
            right_scale
        )

        smaller = max(
            min(
                left_scale,
                right_scale
            ),
            epsilon
        )

        ratio = (
            larger
            / smaller
        )

        if (
            ratio
            > strongest_ratio
        ):

            strongest_ratio = (
                ratio
            )

            strongest_split = (
                split_fraction
            )

            strongest_left_scale = (
                left_scale
            )

            strongest_right_scale = (
                right_scale
            )

    # Conservative mapping:
    #
    # ratio <= 1.5:
    #     ordinary variation
    #
    # ratio >= 4:
    #     very strong volatility regime change

    signal = clamp(
        (
            strongest_ratio
            - 1.50
        )
        / (
            4.00
            - 1.50
        )
        * 100
    )

    return {
        "signal": float(
            signal
        ),

        "dispersion_ratio": float(
            strongest_ratio
        ),

        "split_fraction": float(
            strongest_split
        ),

        "left_scale": float(
            strongest_left_scale
        ),

        "right_scale": float(
            strongest_right_scale
        )
    }


# ==================================================
# SUPPORT ADJUSTMENT
# ==================================================

def adjust_signal_for_support(
    signal,
    support
):
    """
    Reduce confidence in temporal patterns when
    sample support is weak.

    The square-root adjustment avoids overly harsh
    penalties while still reducing small-sample
    confidence.
    """

    reliability = (
        support
        / 100
    ) ** 0.5

    return (
        signal
        * reliability
    )


# ==================================================
# COMBINED TEMPORAL SIGNAL
# ==================================================

def temporal_signal_score(
    trend_data,
    y,
    support
):
    """
    Detect multiple forms of temporal structure.

    Supported patterns:

        trend
        seasonality
        level shift
        volatility shift

    Signals are NOT added together because these
    detectors may capture overlapping evidence.

    The strongest supported temporal pattern
    determines the overall temporal signal.
    """

    trend = (
        trend_signal_score(
            trend_data,
            y
        )
    )

    residuals = (
        detrend_values(
            trend_data,
            y
        )
    )

    seasonality = (
        seasonality_signal_score(
            trend_data,
            residuals
        )
    )

    level_shift = (
        level_shift_signal_score(
            residuals
        )
    )

    volatility = (
        volatility_shift_signal_score(
            residuals
        )
    )

    raw_signals = {
        "trend": (
            trend[
                "signal"
            ]
        ),

        "seasonality": (
            seasonality[
                "signal"
            ]
        ),

        "level_shift": (
            level_shift[
                "signal"
            ]
        ),

        "volatility_shift": (
            volatility[
                "signal"
            ]
        )
    }

    adjusted_signals = {
        pattern: (
            adjust_signal_for_support(
                signal,
                support
            )
        )
        for (
            pattern,
            signal
        ) in raw_signals.items()
    }

    dominant_pattern = max(
        adjusted_signals,
        key=adjusted_signals.get
    )

    final_signal = (
        adjusted_signals[
            dominant_pattern
        ]
    )

    return {
        "signal": float(
            clamp(
                final_signal
            )
        ),

        "dominant_pattern": (
            dominant_pattern
        ),

        "trend_signal": float(
            adjusted_signals[
                "trend"
            ]
        ),

        "seasonality_signal": float(
            adjusted_signals[
                "seasonality"
            ]
        ),

        "level_shift_signal": float(
            adjusted_signals[
                "level_shift"
            ]
        ),

        "volatility_shift_signal": float(
            adjusted_signals[
                "volatility_shift"
            ]
        ),

        "raw_trend_signal": float(
            raw_signals[
                "trend"
            ]
        ),

        "raw_seasonality_signal": float(
            raw_signals[
                "seasonality"
            ]
        ),

        "raw_level_shift_signal": float(
            raw_signals[
                "level_shift"
            ]
        ),

        "raw_volatility_shift_signal": float(
            raw_signals[
                "volatility_shift"
            ]
        ),

        "pearson": (
            trend[
                "pearson"
            ]
        ),

        "spearman": (
            trend[
                "spearman"
            ]
        ),

        "direction": (
            trend[
                "direction"
            ]
        ),

        "seasonality": (
            seasonality
        ),

        "level_shift": (
            level_shift
        ),

        "volatility": (
            volatility
        )
    }


# ==================================================
# FINAL RECOMMENDATION SCORE
# ==================================================

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
# LINE CHART SCORER
# ==================================================

def score_line_chart(
    df,
    candidate,
    profile
):
    """
    Score one line-chart candidate from 0 to 100.

    The score combines:

        visualization suitability
        +
        strongest supported temporal pattern

    Temporal patterns currently include:

        trend
        seasonality
        level shift
        volatility shift
    """

    x = (
        candidate["x"]
    )

    y = (
        candidate["y"]
    )

    x_type = (
        get_semantic_type(
            profile,
            x
        )
    )

    y_type = (
        get_semantic_type(
            profile,
            y
        )
    )

    semantic_fit = (
        semantic_fit_score(
            x_type,
            y_type
        )
    )

    (
        clean,
        trend_data
    ) = prepare_temporal_data(
        df,
        x,
        y,
        x_type
    )

    complete_count = (
        len(
            clean
        )
    )

    unique_time_points = (
        len(
            trend_data
        )
    )

    quality = (
        data_quality_score(
            len(
                df
            ),
            complete_count
        )
    )

    support = (
        sample_support_score(
            complete_count,
            unique_time_points
        )
    )

    readability = (
        readability_score(
            unique_time_points
        )
    )

    temporal = (
        temporal_signal_score(
            trend_data,
            y,
            support
        )
    )

    signal = (
        temporal[
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

    # -----------------------------------
    # Temporal span
    # -----------------------------------

    if (
        unique_time_points
        > 1
    ):

        temporal_span = (
            trend_data[
                "_time_numeric"
            ].max()
            - trend_data[
                "_time_numeric"
            ].min()
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

    # -----------------------------------
    # Human-readable evidence
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
            "Semantic fit for line chart: "
            f"{semantic_fit:.1f}/100."
        ),

        (
            f"{complete_count} complete "
            "observations."
        ),

        (
            f"{unique_time_points} distinct "
            "time points."
        ),

        (
            f"Temporal span: "
            f"{span_description}."
        ),

        (
            f"Data completeness: "
            f"{quality:.1f}%."
        ),

        (
            "Pearson time correlation: "
            f"{temporal['pearson']:.3f}."
        ),

        (
            "Spearman time correlation: "
            f"{temporal['spearman']:.3f}."
        ),

        (
            "Trend signal: "
            f"{temporal['trend_signal']:.1f}/100."
        ),

        (
            "Seasonality signal: "
            f"{temporal['seasonality_signal']:.1f}/100."
        ),

        (
            "Level-shift signal: "
            f"{temporal['level_shift_signal']:.1f}/100."
        ),

        (
            "Volatility-shift signal: "
            f"{temporal['volatility_shift_signal']:.1f}/100."
        ),

        (
            "Strongest temporal pattern: "
            f"{temporal['dominant_pattern'].replace('_', ' ')}."
        ),

        (
            "Overall temporal signal: "
            f"{signal:.1f}/100."
        )
    ]

    # Add pattern-specific evidence.

    if (
        temporal[
            "seasonality"
        ][
            "signal"
        ]
        > 0
    ):

        reasons.append(
            (
                "Estimated repeating period: "
                f"{temporal['seasonality']['period_temporal_units']:.1f} "
                "temporal units."
            )
        )

        reasons.append(
            (
                "Seasonal autocorrelation: "
                f"{temporal['seasonality']['autocorrelation']:.3f}."
            )
        )

    if (
        temporal[
            "level_shift"
        ][
            "signal"
        ]
        > 0
    ):

        reasons.append(
            (
                "Strongest level-shift effect size: "
                f"{temporal['level_shift']['effect_size']:.2f}."
            )
        )

    if (
        temporal[
            "volatility"
        ][
            "signal"
        ]
        > 0
    ):

        reasons.append(
            (
                "Strongest temporal dispersion ratio: "
                f"{temporal['volatility']['dispersion_ratio']:.2f}."
            )
        )

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
            # Preserve these existing fields for
            # compatibility with the frontend.

            "pearson": round(
                temporal[
                    "pearson"
                ],
                4
            ),

            "spearman": round(
                temporal[
                    "spearman"
                ],
                4
            ),

            "direction": (
                temporal[
                    "direction"
                ]
            ),

            "time_points": (
                unique_time_points
            ),

            "temporal_span": round(
                temporal_span,
                2
            ),

            # Existing raw_signal concept now
            # represents the strongest temporal
            # pattern before support adjustment.

            "raw_signal": round(
                max(
                    temporal[
                        "raw_trend_signal"
                    ],
                    temporal[
                        "raw_seasonality_signal"
                    ],
                    temporal[
                        "raw_level_shift_signal"
                    ],
                    temporal[
                        "raw_volatility_shift_signal"
                    ]
                ),
                2
            ),

            # New temporal intelligence.

            "dominant_pattern": (
                temporal[
                    "dominant_pattern"
                ]
            ),

            "trend_signal": round(
                temporal[
                    "trend_signal"
                ],
                2
            ),

            "seasonality_signal": round(
                temporal[
                    "seasonality_signal"
                ],
                2
            ),

            "level_shift_signal": round(
                temporal[
                    "level_shift_signal"
                ],
                2
            ),

            "volatility_shift_signal": round(
                temporal[
                    "volatility_shift_signal"
                ],
                2
            ),

            "seasonal_peak_power_ratio": round(
                temporal[
                    "seasonality"
                ][
                    "peak_power_ratio"
                ],
                4
            ),

            "seasonal_autocorrelation": round(
                temporal[
                    "seasonality"
                ][
                    "autocorrelation"
                ],
                4
            ),

            "seasonal_period_observations": round(
                temporal[
                    "seasonality"
                ][
                    "period_observations"
                ],
                2
            ),

            "seasonal_period_temporal_units": round(
                temporal[
                    "seasonality"
                ][
                    "period_temporal_units"
                ],
                2
            ),

            "level_shift_effect_size": round(
                temporal[
                    "level_shift"
                ][
                    "effect_size"
                ],
                4
            ),

            "level_shift_split_fraction": round(
                temporal[
                    "level_shift"
                ][
                    "split_fraction"
                ],
                2
            ),

            "volatility_dispersion_ratio": round(
                temporal[
                    "volatility"
                ][
                    "dispersion_ratio"
                ],
                4
            ),

            "volatility_split_fraction": round(
                temporal[
                    "volatility"
                ][
                    "split_fraction"
                ],
                2
            )
        },

        "reasons": (
            reasons
        )
    }