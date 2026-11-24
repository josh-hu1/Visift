"""
Typed records passed through the Visift pipeline.

The pipeline exchanges plain dicts (the Streamlit app and the evaluation
code index them with ["key"]), so these are TypedDicts: they document and
type-check the shape without changing any runtime behavior.

    profile_dataset            -> DatasetProfile
    generate_candidates        -> list[Candidate]
    score_<chart> (per family) -> ScoreResult
    score_all_candidates       -> list[ScoredCandidate]
    diversify_recommendations  -> list[Recommendation]

Optional keys use a required base class plus a total=False subclass, which
works on Python 3.10 without typing.NotRequired.
"""

from typing import Any, TypedDict


class _ColumnProfileRequired(TypedDict):
    name: str
    semantic_type: str
    pandas_dtype: str
    row_count: int
    missing_count: int
    missing_percent: float
    unique_count: int


class NumericSummary(TypedDict, total=False):
    """Extra statistics for numeric columns (empty when all values are missing)."""

    min: float
    max: float
    mean: float
    median: float
    std: float


class CategorySummary(TypedDict, total=False):
    """Extra statistics for category-like columns."""

    # value -> count for the five most common values
    top_values: dict[Any, int]


class ColumnProfile(_ColumnProfileRequired, NumericSummary, CategorySummary):
    """One column's inferred semantic type and summary statistics."""


class DatasetProfile(TypedDict):
    rows: int
    columns: int
    column_profiles: list[ColumnProfile]


class _CandidateRequired(TypedDict):
    chart: str
    x: str


class Candidate(_CandidateRequired, total=False):
    """A semantically valid chart: chart type plus the columns it uses."""

    y: str
    # "count" or "mean" for bar charts
    aggregation: str


class _ScoreResultRequired(TypedDict):
    score: float
    components: dict[str, Any]
    reasons: list[str]


class ScoreResult(_ScoreResultRequired, total=False):
    """What a chart-specific scorer returns for one candidate."""

    statistics: dict[str, Any]


class ScoredCandidate(Candidate, ScoreResult):
    """A candidate merged with its score result."""


class AlternativeChart(TypedDict):
    chart: str
    chart_name: str
    title: str
    score: float


class Recommendation(ScoredCandidate):
    """A ranked recommendation, with lower-scoring views of the same variables."""

    alternative_charts: list[AlternativeChart]
