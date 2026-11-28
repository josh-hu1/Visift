"""
The notebook API: from a DataFrame to ranked, explained charts in one call.

    import visift

    recs = visift.recommend(df, target="species")
    recs                    # a ranked table in Jupyter
    recs[0].figure()        # the Plotly figure
    recs[0].takeaway        # "Average ... is highest for ..."
    print(recs[0].code())   # a pandas + Plotly snippet that rebuilds it
    visift.report(df, "report.html", target="species")

Nothing here opens a browser or calls fig.show(); Jupyter renders the
returned figures and tables inline.
"""

from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, SupportsIndex, overload

import pandas as pd
import plotly.graph_objects as go

from visift.explain import ScoreBreakdown, score_breakdown
from visift.export import chart_code, html_report
from visift.insight import plain_english_insight
from visift.paradox import Paradox
from visift.pipeline import Analysis, DatasetNote, analyze
from visift.scoring.engine import chart_display_name, recommendation_strength
from visift.types import Recommendation as RankedChart
from visift.visualization.renderer import render_candidate, visualization_title

__all__ = ["Recommendation", "Recommendations", "recommend", "report", "stats_line"]


@dataclass(frozen=True, eq=False)
class Recommendation:
    """One ranked chart: what it shows, how it scored, and why."""

    rank: int
    title: str
    chart: str
    x: str
    y: str | None
    score: float
    band: str
    takeaway: str
    driver: bool
    breakdown: ScoreBreakdown = field(repr=False)
    evidence: list[str] = field(repr=False)
    raw: RankedChart = field(repr=False)
    data: pd.DataFrame = field(repr=False)
    source_name: str = field(repr=False, default="data.csv")

    @property
    def chart_name(self) -> str:
        return chart_display_name(self.chart)

    def figure(self) -> go.Figure:
        """The Plotly figure, as the app draws it."""

        return render_candidate(self.data, self.raw)

    def code(self) -> str:
        """A pandas + Plotly snippet that rebuilds the chart from source_name."""

        return chart_code(self.raw, self.source_name, self.data)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary: identity, score, takeaway, evidence, breakdown."""

        return {
            "rank": self.rank,
            "title": self.title,
            "chart": self.chart,
            "x": self.x,
            "y": self.y,
            "aggregation": self.raw.get("aggregation"),
            "score": round(self.score, 2),
            "band": self.band,
            "driver": self.driver,
            "takeaway": self.takeaway,
            "evidence": self.evidence,
            "components": {
                key: value
                for key, value in self.raw["components"].items()
                if isinstance(value, (int, float))
            },
            "breakdown": self.breakdown,
        }

    def _repr_html_(self) -> str:
        return (
            f"<p><b>#{self.rank} {escape(self.title)}</b> · {escape(self.chart_name)}"
            f" · score {self.score:.1f} ({escape(self.band)})<br>"
            f"{escape(self.takeaway)}<br><small>{escape(self.breakdown['formula'])}"
            "</small></p>"
        )


class Recommendations(list[Recommendation]):
    """
    The ranked recommendations (a plain list), plus what the app shows
    around them: Simpson's paradoxes and dataset notes.
    """

    def __init__(
        self,
        items: list[Recommendation],
        *,
        target: str | None,
        analysis: Analysis,
        data: pd.DataFrame,
        source_name: str,
    ) -> None:
        super().__init__(items)
        self.target = target
        self.analysis = analysis
        self.data = data
        self.source_name = source_name

    @overload
    def __getitem__(self, index: SupportsIndex) -> Recommendation: ...

    @overload
    def __getitem__(self, index: slice) -> list[Recommendation]: ...

    def __getitem__(
        self, index: SupportsIndex | slice
    ) -> Recommendation | list[Recommendation]:
        return list.__getitem__(self, index)

    @property
    def paradoxes(self) -> list[Paradox]:
        """Simpson's paradox findings ("Worth a closer look"); never scored."""

        return self.analysis["paradoxes"]

    @property
    def notes(self) -> list[DatasetNote]:
        """Redundant or trivially related columns, kept out of the ranking."""

        return self.analysis["notes"]

    def to_frame(self) -> pd.DataFrame:
        columns = ["rank", "title", "chart", "x", "y", "score", "band", "driver"]
        rows = [{c: getattr(r, c) for c in columns} for r in self]
        frame = pd.DataFrame(rows, columns=columns)
        frame["takeaway"] = [r.takeaway for r in self]
        return frame

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td>{r.rank}</td><td>{escape(r.title)}</td>"
            f"<td>{escape(r.chart_name)}</td><td>{r.score:.1f}</td>"
            f"<td>{escape(r.band)}</td><td>{escape(r.takeaway)}</td></tr>"
            for r in self
        )
        heading = f"Visift: {len(self)} recommendations" + (
            f", drivers of <b>{escape(self.target)}</b> first" if self.target else ""
        )
        if self.paradoxes:
            heading += f"; {len(self.paradoxes)} Simpson's paradox finding(s)"
        return (
            f"<p>{heading}</p><table><thead><tr><th>#</th><th>Chart</th>"
            "<th>Type</th><th>Score</th><th>Band</th><th>Takeaway</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>"
        )


def stats_line(analysis: Analysis, source_name: str) -> str:
    """'penguins.csv · 344 rows · 8 columns · 41 chart candidates' (report header)."""

    profile = analysis["profile"]
    return " · ".join(
        [
            source_name,
            f"{profile['rows']:,} rows",
            f"{profile['columns']} columns",
            f"{len(analysis['candidates'])} chart candidates",
        ]
    )


def _wrap(
    analysis: Analysis,
    df: pd.DataFrame,
    target: str | None,
    top: int | None,
    source_name: str,
) -> Recommendations:
    drivers = {id(r) for r in analysis["drivers"]} if target is not None else set()
    ranked = analysis["recommendations"]
    ranked = ranked if top is None else ranked[:top]

    items = [
        Recommendation(
            rank=rank,
            title=visualization_title(r),
            chart=r["chart"],
            x=r["x"],
            y=r.get("y"),
            score=float(r["score"]),
            band=recommendation_strength(float(r["score"])),
            takeaway=plain_english_insight(df, r),
            driver=id(r) in drivers,
            breakdown=score_breakdown(r),
            evidence=list(r["reasons"]),
            raw=r,
            data=df,
            source_name=source_name,
        )
        for rank, r in enumerate(ranked, start=1)
    ]

    return Recommendations(
        items, target=target, analysis=analysis, data=df, source_name=source_name
    )


def recommend(
    df: pd.DataFrame,
    target: str | None = None,
    top: int | None = 5,
    *,
    source_name: str = "data.csv",
    n_jobs: int | None = 1,
) -> Recommendations:
    """
    Rank the charts worth looking at first in df.

    target: an outcome column to explain; charts involving it ("drivers")
    come first. top: how many to return (None for all). source_name is the
    file name the code snippets read from.
    """

    if target is not None and target not in df.columns:
        raise KeyError(f"target {target!r} is not a column of the DataFrame")

    analysis = analyze(df, target=target, n_jobs=n_jobs)
    return _wrap(analysis, df, target, top, source_name)


def report(
    df: pd.DataFrame,
    path: str | Path,
    target: str | None = None,
    top: int | None = 10,
    *,
    source_name: str = "data.csv",
    n_jobs: int | None = 1,
) -> Path:
    """
    Write the offline HTML report (paradoxes, the top recommendations with
    takeaways and score breakdowns, dataset notes) to path and return it.
    """

    recs = recommend(df, target, top, source_name=source_name, n_jobs=n_jobs)
    return write_report(recs, path)


def write_report(recs: Recommendations, path: str | Path) -> Path:
    page = html_report(
        recs.data,
        source_name=recs.source_name,
        stats_line=stats_line(recs.analysis, recs.source_name),
        target=recs.target,
        paradoxes=recs.paradoxes,
        recommendations=[r.raw for r in recs],
        notes=recs.notes,
        insight=plain_english_insight,
    )
    out = Path(path)
    out.write_text(page, encoding="utf-8")
    return out
