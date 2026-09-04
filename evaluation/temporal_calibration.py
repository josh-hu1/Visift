from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.line import (
    prepare_temporal_data,
    sample_support_score,
    score_line_chart,
    temporal_signal_score
)


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results/temporal_calibration_v3"
)

DEFAULT_SEEDS = range(
    25
)

DEFAULT_SAMPLE_SIZE = 500

SMALL_SAMPLE_NULL_SIZES = [
    20,
    30,
    50,
    100
]

SMALL_SAMPLE_NULL_SEEDS = range(
    100
)

STRENGTH_LABELS = {
    0: "ordinary",
    1: "weak",
    2: "moderate",
    3: "strong",
    4: "extreme"
}

EXPECTED_PATTERN = {
    "trend": "trend",
    "seasonality": "seasonality",
    "level_shift": "level_shift",
    "volatility_shift": "volatility_shift"
}


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class TemporalCalibrationScenario:
    """
    One controlled temporal-strength scenario.

    The hardened Visift benchmark tests retrieval
    across the full recommendation pipeline.

    This benchmark instead isolates the line-chart
    scorer and asks whether scores behave sensibly
    as temporal structure becomes stronger.
    """

    name: str
    family: str
    level: str
    strength_index: int
    seed: int
    description: str
    dataframe: pd.DataFrame
    x: str
    y: str


@dataclass(frozen=True)
class TemporalRobustnessScenario:
    """
    One temporal robustness or null scenario.
    """

    name: str
    family: str
    setting: str
    setting_value: float
    seed: int
    description: str
    dataframe: pd.DataFrame
    x: str
    y: str


# ==================================================
# COMMON HELPERS
# ==================================================

def strength_label(
    strength_index
):
    return STRENGTH_LABELS[
        strength_index
    ]


def manual_line_profile(
    x,
    y
):
    """
    Build the minimal profile required by the line
    scorer.

    Semantic inference is fixed intentionally so
    this benchmark measures line-score calibration,
    not type inference.
    """

    return {
        "column_profiles": [
            {
                "name": x,
                "semantic_type": "temporal"
            },
            {
                "name": y,
                "semantic_type":
                    "numeric_continuous"
            }
        ]
    }


def make_dataframe(
    time_values,
    y_values,
    x="time",
    y="value"
):
    return pd.DataFrame({
        x: time_values,
        y: y_values
    })


def standardized_time(
    n
):
    values = np.arange(
        n,
        dtype=float
    )

    std = values.std()

    if std <= 0:
        return values

    return (
        values
        - values.mean()
    ) / std


# ==================================================
# TREND FAMILY
# ==================================================

def trend_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate a controlled linear temporal trend.

    The planted strength approximately corresponds
    to the population correlation between time and
    the response.
    """

    rng = np.random.default_rng(
        seed
    )

    relationship = {
        0: 0.00,
        1: 0.15,
        2: 0.30,
        3: 0.55,
        4: 0.85
    }[
        strength_index
    ]

    time_values = np.arange(
        n,
        dtype=float
    )

    time_standardized = (
        standardized_time(
            n
        )
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    values = (
        relationship
        * time_standardized
        + np.sqrt(
            max(
                0.0,
                1.0
                - relationship ** 2
            )
        )
        * noise
    )

    return (
        time_values,
        values
    )


def trend_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    time_values, values = trend_values(
        seed=seed,
        strength_index=
            strength_index,
        n=n
    )

    return TemporalCalibrationScenario(
        name=(
            f"temporal_trend_{level}_"
            f"seed_{seed}"
        ),
        family="trend",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Controlled monotonic temporal trend "
            f"at {level} strength."
        ),
        dataframe=make_dataframe(
            time_values,
            values
        ),
        x="time",
        y="value"
    )


# ==================================================
# SEASONALITY FAMILY
# ==================================================

def seasonality_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE,
    period=25
):
    """
    Generate a regularly sampled sinusoidal pattern.

    The amplitude schedule intentionally spans from
    invisible/weak structure to dominant periodic
    behavior so the current seasonality detector's
    activation threshold can be measured.
    """

    rng = np.random.default_rng(
        seed
    )

    amplitude = {
        0: 0.00,
        1: 0.60,
        2: 1.00,
        3: 1.50,
        4: 2.00
    }[
        strength_index
    ]

    time_values = np.arange(
        n,
        dtype=float
    )

    seasonal_component = (
        amplitude
        * np.sin(
            2
            * np.pi
            * time_values
            / period
        )
    )

    values = (
        seasonal_component
        + rng.normal(
            0,
            1,
            n
        )
    )

    return (
        time_values,
        values
    )


def seasonality_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    time_values, values = (
        seasonality_values(
            seed=seed,
            strength_index=
                strength_index,
            n=n
        )
    )

    return TemporalCalibrationScenario(
        name=(
            f"temporal_seasonality_{level}_"
            f"seed_{seed}"
        ),
        family="seasonality",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Controlled repeating temporal pattern "
            f"at {level} strength."
        ),
        dataframe=make_dataframe(
            time_values,
            values
        ),
        x="time",
        y="value"
    )


# ==================================================
# LEVEL-SHIFT FAMILY
# ==================================================

def level_shift_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate an abrupt persistent change in level.

    The shift begins after the first quarter of the
    series. This placement is intentional: the
    current detector checks central split locations
    including 25%.

    Because a persistent step can also correlate
    with time, this family is useful for measuring
    cross-talk between the trend and level-shift
    detectors.
    """

    rng = np.random.default_rng(
        seed
    )

    shift = {
        0: 0.00,
        1: 0.50,
        2: 1.00,
        3: 2.00,
        4: 4.00
    }[
        strength_index
    ]

    time_values = np.arange(
        n,
        dtype=float
    )

    values = rng.normal(
        0,
        1,
        n
    )

    split_index = int(
        round(
            0.25
            * n
        )
    )

    values[
        split_index:
    ] += shift

    return (
        time_values,
        values
    )


def level_shift_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    time_values, values = (
        level_shift_values(
            seed=seed,
            strength_index=
                strength_index,
            n=n
        )
    )

    return TemporalCalibrationScenario(
        name=(
            f"temporal_level_shift_{level}_"
            f"seed_{seed}"
        ),
        family="level_shift",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Controlled persistent level shift "
            f"at {level} strength."
        ),
        dataframe=make_dataframe(
            time_values,
            values
        ),
        x="time",
        y="value"
    )


# ==================================================
# VOLATILITY-SHIFT FAMILY
# ==================================================

def volatility_shift_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate a sustained change in temporal
    variability while keeping the mean near zero.
    """

    rng = np.random.default_rng(
        seed
    )

    scale_ratio = {
        0: 1.00,
        1: 1.30,
        2: 1.75,
        3: 2.50,
        4: 4.00
    }[
        strength_index
    ]

    time_values = np.arange(
        n,
        dtype=float
    )

    split_index = (
        n // 2
    )

    left = rng.normal(
        0,
        1,
        split_index
    )

    right = rng.normal(
        0,
        scale_ratio,
        n - split_index
    )

    values = np.concatenate([
        left,
        right
    ])

    return (
        time_values,
        values
    )


def volatility_shift_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    time_values, values = (
        volatility_shift_values(
            seed=seed,
            strength_index=
                strength_index,
            n=n
        )
    )

    return TemporalCalibrationScenario(
        name=(
            "temporal_volatility_shift_"
            f"{level}_seed_{seed}"
        ),
        family="volatility_shift",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Controlled volatility-regime change "
            f"at {level} strength."
        ),
        dataframe=make_dataframe(
            time_values,
            values
        ),
        x="time",
        y="value"
    )


# ==================================================
# STRENGTH SUITE
# ==================================================

def build_strength_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Build the main temporal calibration suite.

    Default configuration:

        4 temporal pattern families
        5 strength levels
        25 seeds
        500 total scenarios
    """

    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for strength_index in range(
            5
        ):

            scenarios.extend([
                trend_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                seasonality_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                level_shift_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                volatility_shift_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                )
            ])

    return scenarios


# ==================================================
# SAMPLE-SIZE RELIABILITY
# ==================================================

def sample_size_scenario(
    seed,
    n
):
    """
    Hold a moderately strong temporal trend fixed
    while changing the number of observations.
    """

    rng = np.random.default_rng(
        seed
    )

    relationship = 0.55

    time_values = np.arange(
        n,
        dtype=float
    )

    time_standardized = (
        standardized_time(
            n
        )
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    values = (
        relationship
        * time_standardized
        + np.sqrt(
            1
            - relationship ** 2
        )
        * noise
    )

    df = make_dataframe(
        time_values,
        values
    )

    return TemporalRobustnessScenario(
        name=(
            f"temporal_sample_size_n{n}_"
            f"seed_{seed}"
        ),
        family="sample_size",
        setting=f"n={n}",
        setting_value=float(
            n
        ),
        seed=seed,
        description=(
            "Fixed temporal trend with "
            f"{n} observations."
        ),
        dataframe=df,
        x="time",
        y="value"
    )


def build_sample_size_suite(
    seeds=None
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    sample_sizes = [
        20,
        30,
        50,
        100,
        300
    ]

    scenarios = []

    for seed in seeds:

        for n in sample_sizes:

            scenarios.append(
                sample_size_scenario(
                    seed=seed,
                    n=n
                )
            )

    return scenarios


# ==================================================
# MISSINGNESS ROBUSTNESS
# ==================================================

def missingness_scenario(
    seed,
    missingness,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Hold a strong temporal trend fixed while
    progressively removing response observations
    at random.
    """

    rng = np.random.default_rng(
        seed
    )

    relationship = 0.70

    time_values = np.arange(
        n,
        dtype=float
    )

    time_standardized = (
        standardized_time(
            n
        )
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    values = (
        relationship
        * time_standardized
        + np.sqrt(
            1
            - relationship ** 2
        )
        * noise
    )

    df = make_dataframe(
        time_values,
        values
    )

    missing_rng = (
        np.random.default_rng(
            seed
            + 100_000
        )
    )

    missing_count = int(
        n
        * missingness
    )

    if missing_count > 0:

        indices = (
            missing_rng.choice(
                df.index,
                size=missing_count,
                replace=False
            )
        )

        df.loc[
            indices,
            "value"
        ] = np.nan

    return TemporalRobustnessScenario(
        name=(
            "temporal_missingness_"
            f"{int(missingness * 100)}pct_"
            f"seed_{seed}"
        ),
        family="missingness",
        setting=(
            f"{int(missingness * 100)}%"
        ),
        setting_value=float(
            missingness
        ),
        seed=seed,
        description=(
            "Fixed strong temporal trend with "
            f"{missingness * 100:.0f}% "
            "missing response values."
        ),
        dataframe=df,
        x="time",
        y="value"
    )


def build_missingness_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    missingness_levels = [
        0.00,
        0.20,
        0.40,
        0.60,
        0.80
    ]

    scenarios = []

    for seed in seeds:

        for missingness in (
            missingness_levels
        ):

            scenarios.append(
                missingness_scenario(
                    seed=seed,
                    missingness=
                        missingness,
                    n=n
                )
            )

    return scenarios


# ==================================================
# SAMPLING-REGULARITY ROBUSTNESS
# ==================================================

def irregular_seasonality_scenario(
    seed,
    jitter,
    n=DEFAULT_SAMPLE_SIZE,
    period=25
):
    """
    Generate a real sinusoidal pattern on an
    increasingly irregular temporal grid.

    The current FFT-based seasonality detector
    deliberately requires approximately regular
    sampling, so this suite measures where that
    protection begins to suppress a real seasonal
    signal.
    """

    rng = np.random.default_rng(
        seed
    )

    if jitter == 0:

        intervals = np.ones(
            n,
            dtype=float
        )

    else:

        intervals = (
            1
            + rng.normal(
                0,
                jitter,
                n
            )
        )

        intervals = np.clip(
            intervals,
            0.20,
            None
        )

    time_values = np.cumsum(
        intervals
    )

    time_values = (
        time_values
        - time_values[0]
    )

    values = (
        1.50
        * np.sin(
            2
            * np.pi
            * time_values
            / period
        )
        + rng.normal(
            0,
            1,
            n
        )
    )

    return TemporalRobustnessScenario(
        name=(
            "temporal_sampling_jitter_"
            f"{jitter:.2f}_seed_{seed}"
        ),
        family="sampling_regularity",
        setting=(
            f"jitter={jitter:.2f}"
        ),
        setting_value=float(
            jitter
        ),
        seed=seed,
        description=(
            "Strong seasonal relationship with "
            f"temporal interval jitter {jitter:.2f}."
        ),
        dataframe=make_dataframe(
            time_values,
            values
        ),
        x="time",
        y="value"
    )


def build_sampling_regularity_suite(
    seeds=None
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    jitter_levels = [
        0.00,
        0.03,
        0.08,
        0.15,
        0.30
    ]

    scenarios = []

    for seed in seeds:

        for jitter in jitter_levels:

            scenarios.append(
                irregular_seasonality_scenario(
                    seed=seed,
                    jitter=jitter
                )
            )

    return scenarios


# ==================================================
# NULL ROBUSTNESS
# ==================================================

def null_white_noise_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
    rng = np.random.default_rng(
        seed
    )

    return TemporalRobustnessScenario(
        name=(
            f"temporal_null_white_noise_"
            f"seed_{seed}"
        ),
        family="null_white_noise",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Regular time axis with independent "
            "Gaussian white noise."
        ),
        dataframe=make_dataframe(
            np.arange(
                n,
                dtype=float
            ),
            rng.normal(
                0,
                1,
                n
            )
        ),
        x="time",
        y="value"
    )


def null_skewed_noise_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
    rng = np.random.default_rng(
        seed
    )

    return TemporalRobustnessScenario(
        name=(
            f"temporal_null_skewed_noise_"
            f"seed_{seed}"
        ),
        family="null_skewed_noise",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Regular time axis with independent "
            "strongly skewed observations."
        ),
        dataframe=make_dataframe(
            np.arange(
                n,
                dtype=float
            ),
            rng.lognormal(
                0,
                1,
                n
            )
        ),
        x="time",
        y="value"
    )


def null_isolated_spikes_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Random isolated spikes should not be mistaken
    for a persistent trend, level shift, or
    volatility regime change.
    """

    rng = np.random.default_rng(
        seed
    )

    values = rng.normal(
        0,
        1,
        n
    )

    spike_count = max(
        1,
        int(
            0.02
            * n
        )
    )

    spike_indices = rng.choice(
        n,
        size=spike_count,
        replace=False
    )

    values[
        spike_indices
    ] += rng.normal(
        0,
        10,
        spike_count
    )

    return TemporalRobustnessScenario(
        name=(
            "temporal_null_isolated_spikes_"
            f"seed_{seed}"
        ),
        family="null_isolated_spikes",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Independent temporal noise with "
            "random isolated extreme spikes."
        ),
        dataframe=make_dataframe(
            np.arange(
                n,
                dtype=float
            ),
            values
        ),
        x="time",
        y="value"
    )


def null_irregular_noise_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
    rng = np.random.default_rng(
        seed
    )

    intervals = rng.uniform(
        0.50,
        2.00,
        n
    )

    time_values = np.cumsum(
        intervals
    )

    values = rng.normal(
        0,
        1,
        n
    )

    return TemporalRobustnessScenario(
        name=(
            "temporal_null_irregular_noise_"
            f"seed_{seed}"
        ),
        family="null_irregular_noise",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Irregular temporal grid with "
            "independent Gaussian noise."
        ),
        dataframe=make_dataframe(
            time_values,
            values
        ),
        x="time",
        y="value"
    )


def build_null_suite(
    seeds=None
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        scenarios.extend([
            null_white_noise_scenario(
                seed
            ),
            null_skewed_noise_scenario(
                seed
            ),
            null_isolated_spikes_scenario(
                seed
            ),
            null_irregular_noise_scenario(
                seed
            )
        ])

    return scenarios



# ==================================================
# SMALL-SAMPLE NULL ROBUSTNESS
# ==================================================

def small_sample_null_scenario(
    seed,
    n
):
    """
    Independent Gaussian white noise on a regular
    temporal grid with deliberately limited sample
    size.

    This suite is separate from the ordinary null
    benchmark because small temporal samples are a
    particularly difficult false-positive regime:
    random correlations, spectral peaks, and segment
    dispersion differences can all look large by
    chance.
    """

    rng = np.random.default_rng(
        seed
    )

    return TemporalRobustnessScenario(
        name=(
            "temporal_null_small_sample_"
            f"n{n}_seed_{seed}"
        ),
        family="null_small_sample",
        setting=f"n={n}",
        setting_value=float(
            n
        ),
        seed=seed,
        description=(
            "Regular temporal grid with "
            f"{n} independent Gaussian "
            "white-noise observations."
        ),
        dataframe=make_dataframe(
            np.arange(
                n,
                dtype=float
            ),
            rng.normal(
                0,
                1,
                n
            )
        ),
        x="time",
        y="value"
    )


def build_small_sample_null_suite(
    seeds=None
):
    """
    Build a higher-replication small-sample null
    suite.

    By default:

        4 sample sizes
        x 100 random seeds
        = 400 null scenarios
    """

    if seeds is None:
        seeds = SMALL_SAMPLE_NULL_SEEDS

    scenarios = []

    for seed in seeds:

        for n in (
            SMALL_SAMPLE_NULL_SIZES
        ):

            scenarios.append(
                small_sample_null_scenario(
                    seed=seed,
                    n=n
                )
            )

    return scenarios


# ==================================================
# SCORING
# ==================================================

def score_dataframe(
    df,
    x,
    y
):
    """
    Run one dataframe through the current line-chart
    scorer and expose the detector-level statistics
    needed for calibration.
    """

    candidate = {
        "chart": "line",
        "x": x,
        "y": y
    }

    profile = (
        manual_line_profile(
            x,
            y
        )
    )

    result = score_line_chart(
        df=df,
        candidate=candidate,
        profile=profile
    )

    clean, trend_data = (
        prepare_temporal_data(
            df,
            x,
            y,
            "temporal"
        )
    )

    support = sample_support_score(
        len(
            clean
        ),
        len(
            trend_data
        )
    )

    temporal = temporal_signal_score(
        trend_data,
        y,
        support
    )

    statistics = result[
        "statistics"
    ]

    return {
        "observations":
            len(
                clean
            ),

        "time_points":
            len(
                trend_data
            ),

        "score":
            result[
                "score"
            ],

        "signal":
            result[
                "components"
            ][
                "signal"
            ],

        "visualization_quality":
            result[
                "components"
            ][
                "visualization_quality"
            ],

        "support":
            result[
                "components"
            ][
                "sample_support"
            ],

        "data_quality":
            result[
                "components"
            ][
                "data_quality"
            ],

        "dominant_pattern":
            statistics[
                "dominant_pattern"
            ],

        "pearson":
            statistics[
                "pearson"
            ],

        "spearman":
            statistics[
                "spearman"
            ],

        "trend_signal":
            statistics[
                "trend_signal"
            ],

        "trend_p_value":
            statistics.get(
                "trend_p_value",
                1
            ),

        "trend_reliability":
            statistics.get(
                "trend_reliability",
                0
            ),

        "raw_linear_trend_signal":
            statistics.get(
                "raw_linear_trend_signal",
                0
            ),

        "seasonality_signal":
            statistics[
                "seasonality_signal"
            ],

        "level_shift_signal":
            statistics[
                "level_shift_signal"
            ],

        "volatility_shift_signal":
            statistics[
                "volatility_shift_signal"
            ],

        "raw_trend_signal":
            temporal[
                "raw_trend_signal"
            ],

        "raw_seasonality_signal":
            temporal[
                "raw_seasonality_signal"
            ],

        "raw_level_shift_signal":
            temporal[
                "raw_level_shift_signal"
            ],

        "raw_volatility_shift_signal":
            temporal[
                "raw_volatility_shift_signal"
            ],

        "seasonal_peak_power_ratio":
            statistics[
                "seasonal_peak_power_ratio"
            ],

        "seasonal_autocorrelation":
            statistics[
                "seasonal_autocorrelation"
            ],

        "seasonal_period":
            statistics[
                "seasonal_period_observations"
            ],

        "level_shift_effect_size":
            statistics[
                "level_shift_effect_size"
            ],

        "level_shift_reliability":
            statistics.get(
                "level_shift_reliability",
                0
            ),

        "level_shift_adjusted_p_value":
            statistics.get(
                "level_shift_adjusted_p_value",
                1
            ),

        "volatility_dispersion_ratio":
            statistics[
                "volatility_dispersion_ratio"
            ],

        "volatility_raw_signal":
            statistics.get(
                "volatility_raw_signal",
                0
            ),

        "volatility_reliability":
            statistics.get(
                "volatility_reliability",
                0
            ),

        "volatility_adjusted_p_value":
            statistics.get(
                "volatility_adjusted_p_value",
                1
            ),

        "seasonality_regular_sampling":
            bool(
                temporal[
                    "seasonality"
                ][
                    "regular_sampling"
                ]
            )
    }


def run_strength_scenario(
    scenario
):
    metrics = score_dataframe(
        df=scenario.dataframe,
        x=scenario.x,
        y=scenario.y
    )

    return {
        "name": scenario.name,
        "family": scenario.family,
        "level": scenario.level,
        "strength_index":
            scenario.strength_index,
        "seed": scenario.seed,
        "expected_pattern":
            EXPECTED_PATTERN[
                scenario.family
            ],
        **metrics
    }


def run_robustness_scenario(
    scenario
):
    metrics = score_dataframe(
        df=scenario.dataframe,
        x=scenario.x,
        y=scenario.y
    )

    return {
        "name": scenario.name,
        "family": scenario.family,
        "setting": scenario.setting,
        "setting_value":
            scenario.setting_value,
        "seed": scenario.seed,
        **metrics
    }


# ==================================================
# SUMMARY HELPERS
# ==================================================

def percentile_10(
    series
):
    return series.quantile(
        0.10
    )


def percentile_90(
    series
):
    return series.quantile(
        0.90
    )


def strength_level_summary(
    results
):
    summary = (
        results
        .groupby(
            [
                "family",
                "strength_index",
                "level"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "score",
                "size"
            ),

            median_score=(
                "score",
                "median"
            ),

            mean_score=(
                "score",
                "mean"
            ),

            p10_score=(
                "score",
                percentile_10
            ),

            p90_score=(
                "score",
                percentile_90
            ),

            median_signal=(
                "signal",
                "median"
            ),

            median_trend_signal=(
                "trend_signal",
                "median"
            ),

            median_trend_reliability=(
                "trend_reliability",
                "median"
            ),

            median_seasonality_signal=(
                "seasonality_signal",
                "median"
            ),

            median_level_shift_signal=(
                "level_shift_signal",
                "median"
            ),

            median_volatility_signal=(
                "volatility_shift_signal",
                "median"
            ),

            median_pearson=(
                "pearson",
                "median"
            ),

            median_spearman=(
                "spearman",
                "median"
            ),

            median_seasonal_power=(
                "seasonal_peak_power_ratio",
                "median"
            ),

            median_seasonal_autocorr=(
                "seasonal_autocorrelation",
                "median"
            ),

            median_level_effect=(
                "level_shift_effect_size",
                "median"
            ),

            median_level_reliability=(
                "level_shift_reliability",
                "median"
            ),

            median_volatility_ratio=(
                "volatility_dispersion_ratio",
                "median"
            ),

            median_volatility_reliability=(
                "volatility_reliability",
                "median"
            )
        )
        .sort_values(
            [
                "family",
                "strength_index"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    dominance = (
        results.assign(
            correct_pattern=(
                results[
                    "dominant_pattern"
                ]
                == results[
                    "expected_pattern"
                ]
            )
        )
        .groupby(
            [
                "family",
                "strength_index",
                "level"
            ],
            as_index=False
        )
        .agg(
            expected_pattern_dominant_pct=(
                "correct_pattern",
                lambda values:
                values.mean()
                * 100
            )
        )
    )

    return summary.merge(
        dominance,
        on=[
            "family",
            "strength_index",
            "level"
        ],
        how="left"
    )


def adjacent_monotonicity(
    results
):
    rows = []

    for family in sorted(
        results[
            "family"
        ].unique()
    ):

        family_results = (
            results[
                results[
                    "family"
                ]
                == family
            ]
        )

        for lower_strength in range(
            4
        ):

            higher_strength = (
                lower_strength
                + 1
            )

            lower = (
                family_results[
                    family_results[
                        "strength_index"
                    ]
                    == lower_strength
                ][
                    [
                        "seed",
                        "score"
                    ]
                ]
                .rename(
                    columns={
                        "score":
                            "lower_score"
                    }
                )
            )

            higher = (
                family_results[
                    family_results[
                        "strength_index"
                    ]
                    == higher_strength
                ][
                    [
                        "seed",
                        "score"
                    ]
                ]
                .rename(
                    columns={
                        "score":
                            "higher_score"
                    }
                )
            )

            paired = lower.merge(
                higher,
                on="seed",
                how="inner"
            )

            if paired.empty:
                continue

            paired[
                "delta"
            ] = (
                paired[
                    "higher_score"
                ]
                - paired[
                    "lower_score"
                ]
            )

            rows.append({
                "family": family,

                "comparison": (
                    f"{strength_label(lower_strength)} "
                    "-> "
                    f"{strength_label(higher_strength)}"
                ),

                "paired_seeds":
                    len(
                        paired
                    ),

                "median_score_delta":
                    paired[
                        "delta"
                    ].median(),

                "mean_score_delta":
                    paired[
                        "delta"
                    ].mean(),

                "stronger_higher_pct":
                    (
                        paired[
                            "delta"
                        ]
                        .gt(
                            0
                        )
                        .mean()
                        * 100
                    )
            })

    return pd.DataFrame(
        rows
    )


def full_monotonicity(
    results
):
    rows = []

    for family in sorted(
        results[
            "family"
        ].unique()
    ):

        family_results = (
            results[
                results[
                    "family"
                ]
                == family
            ]
        )

        pivot = (
            family_results
            .pivot_table(
                index="seed",
                columns="strength_index",
                values="score",
                aggfunc="first"
            )
        )

        required = [
            0,
            1,
            2,
            3,
            4
        ]

        if not all(
            column in pivot.columns
            for column in required
        ):

            pct = float(
                "nan"
            )

        else:

            ordered = (
                pivot[
                    required
                ]
                .dropna()
            )

            monotonic = (
                (
                    ordered[0]
                    < ordered[1]
                )
                & (
                    ordered[1]
                    < ordered[2]
                )
                & (
                    ordered[2]
                    < ordered[3]
                )
                & (
                    ordered[3]
                    < ordered[4]
                )
            )

            pct = (
                monotonic.mean()
                * 100
            )

        rows.append({
            "family": family,
            "fully_monotonic_seed_pct":
                pct
        })

    return pd.DataFrame(
        rows
    )


def robustness_summary(
    results
):
    return (
        results
        .groupby(
            [
                "family",
                "setting",
                "setting_value"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "score",
                "size"
            ),

            median_observations=(
                "observations",
                "median"
            ),

            median_time_points=(
                "time_points",
                "median"
            ),

            median_score=(
                "score",
                "median"
            ),

            mean_score=(
                "score",
                "mean"
            ),

            p10_score=(
                "score",
                percentile_10
            ),

            p90_score=(
                "score",
                percentile_90
            ),

            median_signal=(
                "signal",
                "median"
            ),

            median_support=(
                "support",
                "median"
            ),

            median_quality=(
                "data_quality",
                "median"
            ),

            median_trend_signal=(
                "trend_signal",
                "median"
            ),

            median_seasonality_signal=(
                "seasonality_signal",
                "median"
            ),

            median_level_shift_signal=(
                "level_shift_signal",
                "median"
            ),

            median_volatility_signal=(
                "volatility_shift_signal",
                "median"
            ),

            regular_sampling_pct=(
                "seasonality_regular_sampling",
                lambda values:
                values.mean()
                * 100
            )
        )
        .sort_values(
            [
                "family",
                "setting_value"
            ]
        )
        .reset_index(
            drop=True
        )
    )



def small_sample_null_summary(
    results
):
    """
    Summarize false-positive behavior separately by
    small-sample size and show which temporal detector
    most often becomes dominant.
    """

    rows = []

    for setting_value in sorted(
        results[
            "setting_value"
        ].unique()
    ):

        subset = (
            results[
                results[
                    "setting_value"
                ]
                == setting_value
            ]
        )

        dominant = (
            subset[
                "dominant_pattern"
            ]
            .value_counts(
                normalize=True
            )
        )

        rows.append({
            "sample_size":
                int(
                    setting_value
                ),

            "scenarios":
                len(
                    subset
                ),

            "mean_score":
                subset[
                    "score"
                ].mean(),

            "median_score":
                subset[
                    "score"
                ].median(),

            "p90_score":
                subset[
                    "score"
                ].quantile(
                    0.90
                ),

            "p95_score":
                subset[
                    "score"
                ].quantile(
                    0.95
                ),

            "highest_score":
                subset[
                    "score"
                ].max(),

            "pct_above_50":
                subset[
                    "score"
                ].ge(
                    50
                ).mean()
                * 100,

            "pct_above_65":
                subset[
                    "score"
                ].ge(
                    65
                ).mean()
                * 100,

            "trend_dominant_pct":
                dominant.get(
                    "trend",
                    0
                )
                * 100,

            "seasonality_dominant_pct":
                dominant.get(
                    "seasonality",
                    0
                )
                * 100,

            "level_shift_dominant_pct":
                dominant.get(
                    "level_shift",
                    0
                )
                * 100,

            "volatility_dominant_pct":
                dominant.get(
                    "volatility_shift",
                    0
                )
                * 100
        })

    return pd.DataFrame(
        rows
    )


def null_summary(
    results
):
    return (
        results
        .groupby(
            "family",
            as_index=False
        )
        .agg(
            scenarios=(
                "score",
                "size"
            ),

            mean_score=(
                "score",
                "mean"
            ),

            median_score=(
                "score",
                "median"
            ),

            p90_score=(
                "score",
                percentile_90
            ),

            highest_score=(
                "score",
                "max"
            ),

            pct_above_50=(
                "score",
                lambda values:
                values.ge(
                    50
                ).mean()
                * 100
            ),

            pct_above_65=(
                "score",
                lambda values:
                values.ge(
                    65
                ).mean()
                * 100
            )
        )
    )


# ==================================================
# PRINT HELPERS
# ==================================================

def format_numeric_columns(
    dataframe,
    columns,
    decimals=2
):
    output = dataframe.copy()

    for column in columns:

        if column in output.columns:

            output[
                column
            ] = output[
                column
            ].map(
                lambda value:
                (
                    f"{value:.{decimals}f}"
                    if pd.notna(
                        value
                    )
                    else "N/A"
                )
            )

    return output


def format_percentage_columns(
    dataframe,
    columns
):
    output = dataframe.copy()

    for column in columns:

        if column in output.columns:

            output[
                column
            ] = output[
                column
            ].map(
                lambda value:
                (
                    f"{value:.1f}%"
                    if pd.notna(
                        value
                    )
                    else "N/A"
                )
            )

    return output


# ==================================================
# REPORT
# ==================================================

def print_report(
    strength_results,
    level_summary,
    adjacent_results,
    full_results,
    sample_summary,
    missingness_summary,
    sampling_summary,
    null_results,
    small_sample_null_results
):
    print()
    print(
        "=" * 118
    )
    print(
        "VISIFT TEMPORAL SCORE CALIBRATION"
    )
    print(
        "=" * 118
    )
    print()

    print(
        f"Strength scenarios tested:        "
        f"{len(strength_results)}"
    )

    print(
        f"Temporal pattern families:        "
        f"{strength_results['family'].nunique()}"
    )

    print(
        f"Random seeds:                     "
        f"{strength_results['seed'].nunique()}"
    )

    print(
        f"Default observations/scenario:    "
        f"{DEFAULT_SAMPLE_SIZE}"
    )

    print()

    print(
        "PURPOSE"
    )
    print(
        "-" * 118
    )

    print(
        "This benchmark isolates Visift's line-chart "
        "scorer. It measures graduated trend, "
        "seasonality, level-shift, and volatility-"
        "shift behavior, then tests sample support, "
        "missingness, sampling regularity, and null "
        "robustness."
    )

    print()

    print(
        "TEMPORAL STRENGTH SUMMARY"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        level_summary,
        [
            "median_score",
            "mean_score",
            "p10_score",
            "p90_score",
            "median_signal",
            "median_trend_signal",
            "median_trend_reliability",
            "median_seasonality_signal",
            "median_level_shift_signal",
            "median_level_reliability",
            "median_volatility_signal",
            "median_volatility_reliability",
            "median_pearson",
            "median_spearman",
            "median_seasonal_power",
            "median_seasonal_autocorr",
            "median_level_effect",
            "median_volatility_ratio"
        ]
    )

    printable = format_percentage_columns(
        printable,
        [
            "expected_pattern_dominant_pct"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "ADJACENT STRENGTH COMPARISONS"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        adjacent_results,
        [
            "median_score_delta",
            "mean_score_delta"
        ]
    )

    printable = format_percentage_columns(
        printable,
        [
            "stronger_higher_pct"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "FULL FIVE-LEVEL MONOTONICITY"
    )
    print(
        "-" * 118
    )

    printable = format_percentage_columns(
        full_results,
        [
            "fully_monotonic_seed_pct"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    for title, dataframe in [
        (
            "SAMPLE-SIZE RELIABILITY",
            sample_summary
        ),
        (
            "MISSINGNESS ROBUSTNESS",
            missingness_summary
        ),
        (
            "SAMPLING-REGULARITY / SEASONALITY ROBUSTNESS",
            sampling_summary
        )
    ]:

        print(
            title
        )
        print(
            "-" * 118
        )

        printable = format_numeric_columns(
            dataframe,
            [
                "median_observations",
                "median_time_points",
                "median_score",
                "mean_score",
                "p10_score",
                "p90_score",
                "median_signal",
                "median_support",
                "median_quality",
                "median_trend_signal",
                "median_seasonality_signal",
                "median_level_shift_signal",
                "median_volatility_signal"
            ]
        )

        printable = format_percentage_columns(
            printable,
            [
                "regular_sampling_pct"
            ]
        )

        print(
            printable.to_string(
                index=False
            )
        )

        print()

    print(
        "SMALL-SAMPLE NULL ROBUSTNESS"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        small_sample_null_results,
        [
            "mean_score",
            "median_score",
            "p90_score",
            "p95_score",
            "highest_score"
        ]
    )

    printable = format_percentage_columns(
        printable,
        [
            "pct_above_50",
            "pct_above_65",
            "trend_dominant_pct",
            "seasonality_dominant_pct",
            "level_shift_dominant_pct",
            "volatility_dominant_pct"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "NULL TEMPORAL ROBUSTNESS"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        null_results,
        [
            "mean_score",
            "median_score",
            "p90_score",
            "highest_score"
        ]
    )

    printable = format_percentage_columns(
        printable,
        [
            "pct_above_50",
            "pct_above_65"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "INTERPRETATION GUIDE"
    )
    print(
        "-" * 118
    )

    print(
        "Look for:"
    )
    print(
        "  1. Ordinary/null temporal series staying "
        "near the bottom of the recommendation scale."
    )
    print(
        "  2. Weak < moderate < strong < extreme "
        "within each temporal pattern family."
    )
    print(
        "  3. The intended detector becoming dominant "
        "as its planted pattern becomes strong."
    )
    print(
        "  4. Seasonality activating only when both "
        "spectral and autocorrelation evidence are "
        "meaningful."
    )
    print(
        "  5. Level-shift scenarios revealing whether "
        "the trend detector masks abrupt changes."
    )
    print(
        "  6. Volatility scores tracking persistent "
        "dispersion-ratio changes."
    )
    print(
        "  7. Small samples reducing confidence without "
        "destroying genuine temporal structure."
    )
    print(
        "  8. Missingness degrading scores gradually."
    )
    print(
        "  9. Irregular sampling suppressing FFT-based "
        "seasonality when the regularity assumption "
        "breaks."
    )
    print(
        " 10. Small-sample white noise staying below "
        "recommendation thresholds despite random "
        "correlations or detector spikes."
    )
    print(
        " 11. No null temporal family regularly crossing "
        "the 50-point recommendation threshold."
    )

    print()

    print(
        "=" * 118
    )
    print(
        f"Raw results saved to {RESULTS_DIR}/"
    )
    print(
        "=" * 118
    )
    print()


# ==================================================
# MAIN BENCHMARK
# ==================================================

def run_temporal_calibration(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    # -----------------------------------
    # Graduated temporal strength
    # -----------------------------------

    strength_scenarios = (
        build_strength_suite(
            seeds=seeds,
            n=n
        )
    )

    strength_rows = []

    for index, scenario in enumerate(
        strength_scenarios,
        start=1
    ):

        print(
            "[strength "
            f"{index:03d}/"
            f"{len(strength_scenarios):03d}] "
            f"{scenario.name}"
        )

        strength_rows.append(
            run_strength_scenario(
                scenario
            )
        )

    strength_results = pd.DataFrame(
        strength_rows
    )

    # -----------------------------------
    # Sample-size reliability
    # -----------------------------------

    sample_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_sample_size_suite(
            seeds=seeds
        )
    ]

    sample_raw = pd.DataFrame(
        sample_rows
    )

    sample_summary = (
        robustness_summary(
            sample_raw
        )
    )

    # -----------------------------------
    # Missingness
    # -----------------------------------

    missingness_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_missingness_suite(
            seeds=seeds,
            n=n
        )
    ]

    missingness_raw = pd.DataFrame(
        missingness_rows
    )

    missingness_summary = (
        robustness_summary(
            missingness_raw
        )
    )

    # -----------------------------------
    # Sampling regularity
    # -----------------------------------

    sampling_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_sampling_regularity_suite(
            seeds=seeds
        )
    ]

    sampling_raw = pd.DataFrame(
        sampling_rows
    )

    sampling_summary = (
        robustness_summary(
            sampling_raw
        )
    )

    # -----------------------------------
    # Small-sample null robustness
    # -----------------------------------

    small_sample_null_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_small_sample_null_suite()
    ]

    small_sample_null_raw = (
        pd.DataFrame(
            small_sample_null_rows
        )
    )

    small_sample_null_results = (
        small_sample_null_summary(
            small_sample_null_raw
        )
    )

    # -----------------------------------
    # Null robustness
    # -----------------------------------

    null_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_null_suite(
            seeds=seeds
        )
    ]

    null_raw = pd.DataFrame(
        null_rows
    )

    null_results = (
        null_summary(
            null_raw
        )
    )

    # -----------------------------------
    # Strength summaries
    # -----------------------------------

    level_summary = (
        strength_level_summary(
            strength_results
        )
    )

    adjacent_results = (
        adjacent_monotonicity(
            strength_results
        )
    )

    full_results = (
        full_monotonicity(
            strength_results
        )
    )

    # -----------------------------------
    # Save
    # -----------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {
        "strength_raw.csv":
            strength_results,
        "strength_level_summary.csv":
            level_summary,
        "adjacent_monotonicity.csv":
            adjacent_results,
        "full_monotonicity.csv":
            full_results,
        "sample_size_raw.csv":
            sample_raw,
        "sample_size_summary.csv":
            sample_summary,
        "missingness_raw.csv":
            missingness_raw,
        "missingness_summary.csv":
            missingness_summary,
        "sampling_regularity_raw.csv":
            sampling_raw,
        "sampling_regularity_summary.csv":
            sampling_summary,
        "small_sample_null_raw.csv":
            small_sample_null_raw,
        "small_sample_null_summary.csv":
            small_sample_null_results,
        "null_raw.csv":
            null_raw,
        "null_summary.csv":
            null_results
    }

    for filename, dataframe in (
        outputs.items()
    ):

        dataframe.to_csv(
            RESULTS_DIR
            / filename,
            index=False
        )

    # -----------------------------------
    # Report
    # -----------------------------------

    print_report(
        strength_results=
            strength_results,
        level_summary=
            level_summary,
        adjacent_results=
            adjacent_results,
        full_results=
            full_results,
        sample_summary=
            sample_summary,
        missingness_summary=
            missingness_summary,
        sampling_summary=
            sampling_summary,
        null_results=
            null_results,
        small_sample_null_results=
            small_sample_null_results
    )

    return {
        "strength_raw":
            strength_results,
        "strength_level_summary":
            level_summary,
        "adjacent_monotonicity":
            adjacent_results,
        "full_monotonicity":
            full_results,
        "sample_size_raw":
            sample_raw,
        "sample_size_summary":
            sample_summary,
        "missingness_raw":
            missingness_raw,
        "missingness_summary":
            missingness_summary,
        "sampling_regularity_raw":
            sampling_raw,
        "sampling_regularity_summary":
            sampling_summary,
        "small_sample_null_raw":
            small_sample_null_raw,
        "small_sample_null_summary":
            small_sample_null_results,
        "null_raw":
            null_raw,
        "null_summary":
            null_results
    }


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_temporal_calibration()
