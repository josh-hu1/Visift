import pandas as pd

from semantic_types import classify_column


def profile_numeric_column(series: pd.Series) -> dict:
    """
    Generate additional statistics for numeric columns.
    """

    clean = series.dropna()

    if clean.empty:
        return {}

    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
    }


def profile_categorical_column(series: pd.Series) -> dict:
    """
    Generate additional statistics for category-like columns.
    """

    clean = series.dropna()

    if clean.empty:
        return {}

    counts = clean.value_counts()

    return {
        "top_values": counts.head(5).to_dict()
    }


def profile_column(series: pd.Series) -> dict:
    """
    Profile one DataFrame column.
    """

    semantic_type = classify_column(
        series
    )

    row_count = len(series)
    missing_count = int(
        series.isna().sum()
    )
    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    profile = {
        "name": str(series.name),
        "semantic_type": semantic_type,
        "pandas_dtype": str(series.dtype),
        "row_count": row_count,
        "missing_count": missing_count,
        "missing_percent": (
            round(
                missing_count
                / row_count
                * 100,
                2
            )
            if row_count > 0
            else 0
        ),
        "unique_count": unique_count,
    }

    if semantic_type in {
        "numeric_discrete",
        "numeric_continuous"
    }:
        profile.update(
            profile_numeric_column(
                series
            )
        )

    if semantic_type in {
        "categorical",
        "ordinal",
        "boolean",
        "geographic"
    }:
        profile.update(
            profile_categorical_column(
                series
            )
        )

    return profile


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Profile an entire DataFrame.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_profiles": [
            profile_column(
                df[column]
            )
            for column in df.columns
        ],
    }
