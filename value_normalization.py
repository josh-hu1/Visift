import pandas as pd


BOOLEAN_TEXT_TO_NUMBER = {
    "yes": 1.0,
    "true": 1.0,
    "y": 1.0,
    "on": 1.0,
    "1": 1.0,
    "no": 0.0,
    "false": 0.0,
    "n": 0.0,
    "off": 0.0,
    "0": 0.0,
}


def is_boolean_like_series(series: pd.Series) -> bool:
    non_null = series.dropna()

    if non_null.empty:
        return False

    if pd.api.types.is_bool_dtype(series):
        return True

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(
            non_null,
            errors="coerce"
        ).dropna()

        if len(numeric) != len(non_null):
            return False

        return set(
            numeric.astype(float).unique()
        ) == {
            0.0,
            1.0
        }

    normalized = (
        non_null
        .astype("string")
        .str.strip()
        .str.lower()
    )

    observed = set(
        normalized.dropna().unique()
    )

    recognized_pairs = (
        {"yes", "no"},
        {"true", "false"},
        {"y", "n"},
        {"on", "off"},
        {"1", "0"},
    )

    return any(
        observed == pair
        for pair in recognized_pairs
    )


def coerce_numeric_or_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        )

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    observed = set(
        normalized.dropna().unique()
    )

    recognized_pairs = (
        {"yes", "no"},
        {"true", "false"},
        {"y", "n"},
        {"on", "off"},
        {"1", "0"},
    )

    if any(
        observed == pair
        for pair in recognized_pairs
    ):
        return (
            normalized
            .map(
                BOOLEAN_TEXT_TO_NUMBER
            )
            .astype(float)
        )

    return pd.to_numeric(
        series,
        errors="coerce"
    )
