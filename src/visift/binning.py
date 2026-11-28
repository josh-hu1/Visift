"""
Quantile bins of a numeric column against a binary outcome (binned-rate
bars, pre-registered in evaluation/binned_rate_preregistration.md).

One helper, used by the scorer, the renderer, the takeaway and the exported
code, so every surface bins and labels the same way: quintiles over the
rows where both columns are present, each bin labeled by the smallest and
largest value it holds ("18–30").
"""

import pandas as pd

from visift.value_normalization import coerce_numeric_or_boolean

QUANTILE_BINS = 5
MIN_BINS = 3
MIN_DISTINCT_VALUES = 10


def format_bound(value: float) -> str:
    """Bin bound for a label: 30, 1,250, 0.25, 3.5."""

    value = float(value)
    return (
        f"{value:,.0f}" if abs(value) >= 100 or value.is_integer() else f"{value:.3g}"
    )


def binned_rate_table(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame | None:
    """
    One row per bin, in order: columns "bin" (label), "low", "high",
    "count" and "rate" (share of y == 1). None when fewer than MIN_BINS
    distinct quantile bins exist or y is not 0/1.
    """

    data = df[[x, y]].copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = coerce_numeric_or_boolean(data[y])
    data = data.dropna(subset=[x, y])

    if data.empty or not set(data[y].unique()) <= {0, 1}:
        return None

    codes = pd.qcut(data[x], QUANTILE_BINS, labels=False, duplicates="drop")

    if codes.nunique() < MIN_BINS:
        return None

    table = (
        data.assign(bin_code=codes)
        .groupby("bin_code")
        .agg(low=(x, "min"), high=(x, "max"), count=(y, "size"), rate=(y, "mean"))
        .reset_index(drop=True)
    )
    table.insert(
        0,
        "bin",
        [
            f"{format_bound(low)}–{format_bound(high)}"
            for low, high in zip(table["low"], table["high"], strict=True)
        ],
    )
    return table


def binned_frame(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame | None:
    """Rows with their bin label in column "bin" and y as 0/1 (for scoring)."""

    data = df[[x, y]].copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = coerce_numeric_or_boolean(data[y])
    data = data.dropna(subset=[x, y])

    if data.empty:
        return None

    codes = pd.qcut(data[x], QUANTILE_BINS, labels=False, duplicates="drop")

    if codes.nunique() < MIN_BINS:
        return None

    return pd.DataFrame({"bin": codes.astype(int), y: data[y]})
