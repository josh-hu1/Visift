"""
Correlation overview: one heatmap of Spearman rank correlations between the
numeric columns, as a map of the table before reading the ranked charts.

It is context, not a recommendation: nothing here is scored, ranked or
gated by significance, and it does not change any other output. Strong
pairs reach the ranking (or the dataset notes) through their own scorers.

- Columns: numeric semantic types (numeric_continuous, numeric_discrete), at
  least MIN_COLUMNS of them; with more than MAX_COLUMNS, the MAX_COLUMNS
  whose strongest correlation with another column is largest (ties by
  column order), so the busiest part of a wide table is shown.
- Each cell uses the rows where both columns are present, and is blank
  with fewer than MIN_PAIR_ROWS of them.
- Order: average-linkage clustering on 1 - |rho|, so columns that move
  together sit next to each other. Deterministic.
"""

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from visift.types import DatasetProfile

NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete"}
MIN_COLUMNS = 3
MAX_COLUMNS = 25
MIN_PAIR_ROWS = 10


class CorrelationOverview(TypedDict):
    columns: list[str]  # in display order
    rho: pd.DataFrame  # Spearman rho, NaN on the diagonal and for thin pairs
    rows: pd.DataFrame  # pairwise-complete row counts
    numeric_columns: int  # numeric columns in the table, before any cap


def correlation_overview(
    df: pd.DataFrame, profile: DatasetProfile
) -> CorrelationOverview | None:
    """Spearman correlations between numeric columns, clustered; None if < 3."""

    types = {c["name"]: c["semantic_type"] for c in profile["column_profiles"]}
    numeric = [str(c) for c in df.columns if types.get(str(c)) in NUMERIC_TYPES]

    if len(numeric) < MIN_COLUMNS:
        return None

    values = df[numeric].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    rho = values.corr(method="spearman", min_periods=MIN_PAIR_ROWS)
    present = values.notna().astype(float)
    rows = present.T @ present

    if len(numeric) > MAX_COLUMNS:
        strongest = rho.abs().where(~np.eye(len(numeric), dtype=bool)).max()
        strongest = strongest.fillna(0.0)
        order = sorted(range(len(numeric)), key=lambda i: (-strongest.iloc[i], i))
        keep = [numeric[i] for i in sorted(order[:MAX_COLUMNS])]
        rho = rho.loc[keep, keep]
        rows = rows.loc[keep, keep]

    columns = _cluster_order(rho)
    matrix = rho.loc[columns, columns].to_numpy(dtype=float, copy=True)
    np.fill_diagonal(matrix, np.nan)
    rho = pd.DataFrame(matrix, index=columns, columns=columns)

    return {
        "columns": columns,
        "rho": rho,
        "rows": rows.loc[columns, columns].astype(int),
        "numeric_columns": len(numeric),
    }


def _cluster_order(rho: pd.DataFrame) -> list[str]:
    distance = 1.0 - rho.abs().fillna(0.0).to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2, 0.0, 1.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    return [str(rho.columns[i]) for i in leaves_list(tree)]
