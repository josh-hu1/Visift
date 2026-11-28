"""
Exports: a runnable pandas + Plotly snippet per chart, and a one-page HTML
report.

chart_code() reproduces the chart's *data* exactly as
visift.visualization.renderer prepares it (aggregation, ordering, datetime
resampling, histogram bins and display trimming, scatter sampling, trend
lines). Data-dependent choices the renderer makes at draw time (which
resampling, which sort, which trimming bounds) are resolved when the snippet
is generated and written into it as literals, which keeps snippets short.
Styling and theme are left out.
"""

import html
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

import visift
from visift.explain import score_breakdown
from visift.scoring.engine import (
    chart_display_name,
    recommendation_strength,
    summary_reasons,
)
from visift.visualization.renderer import (
    box_category_order,
    histogram_bin_count,
    prepare_histogram_display_data,
    render_candidate,
    visualization_title,
)

REPO_URL = "https://github.com/josh-hu1/Visift"

SCATTER_MAX_POINTS = 5000  # mirrors renderer.prepare_scatter

_BOOLEAN_PAIRS = (
    ({"yes", "no"}, {"yes": 1.0, "no": 0.0}),
    ({"true", "false"}, {"true": 1.0, "false": 0.0}),
    ({"y", "n"}, {"y": 1.0, "n": 0.0}),
    ({"on", "off"}, {"on": 1.0, "off": 0.0}),
    ({"1", "0"}, {"1": 1.0, "0": 0.0}),
)


# --------------------------------------------------
# Code snippets
# --------------------------------------------------


def _header(
    title: str, source_name: str, extra_imports: Sequence[str] = ()
) -> list[str]:
    return [
        f"# {title}",
        "import pandas as pd",
        "import plotly.express as px",
        *extra_imports,
        "",
        f"df = pd.read_csv({source_name!r})",
    ]


def _numeric_conversion(series: pd.Series, column: Any) -> str:
    """Code that converts a column the way coerce_numeric_or_boolean does."""

    c = repr(column)

    if pd.api.types.is_bool_dtype(series):
        return f"data[{c}] = data[{c}].astype(float)"

    if pd.api.types.is_numeric_dtype(series):
        return f"data[{c}] = pd.to_numeric(data[{c}], errors='coerce')"

    observed = set(series.astype("string").str.strip().str.lower().dropna().unique())

    for pair, mapping in _BOOLEAN_PAIRS:
        if observed == pair:
            return (
                f"data[{c}] = data[{c}].astype('string').str.strip().str.lower()"
                f".map({mapping!r}).astype(float)"
            )

    return f"data[{c}] = pd.to_numeric(data[{c}], errors='coerce')"


def _trend_function() -> list[str]:
    return [
        "def trend(d):  # least-squares line from the smallest to the largest x",
        "    sx, sy = d['x'], d['y']",
        "    slope = ((sx - sx.mean()) * (sy - sy.mean())).sum() / ((sx - sx.mean()) ** 2).sum()",
        "    intercept = sy.mean() - slope * sx.mean()",
        "    return [sx.min(), sx.max()], [slope * v + intercept for v in (sx.min(), sx.max())]",
    ]


def _scatter_code(candidate: Mapping[str, Any], df: pd.DataFrame) -> list[str]:
    x, y = candidate["x"], candidate["y"]
    rows = len(df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna())
    lines = [
        f"data = df[[{x!r}, {y!r}]].apply(pd.to_numeric, errors='coerce').dropna()",
        "data.columns = ['x', 'y']",
        "",
        *_trend_function(),
        "",
    ]
    if rows > SCATTER_MAX_POINTS:
        lines.append(
            f"shown = data.sample(n={SCATTER_MAX_POINTS}, random_state=42)  "
            "# trend still uses every row"
        )
    else:
        lines.append("shown = data")
    lines += [
        f"fig = px.scatter(shown, x='x', y='y', labels={{'x': {x!r}, 'y': {y!r}}})",
        "line_x, line_y = trend(data)",
        "fig.add_scatter(x=line_x, y=line_y, mode='lines', name='Linear trend')",
    ]
    return lines


def _grouped_scatter_code(candidate: Mapping[str, Any]) -> list[str]:
    x, y, g = candidate["x"], candidate["y"], candidate["group"]
    return [
        f"data = df[[{x!r}, {y!r}, {g!r}]].copy()",
        f"data[[{x!r}, {y!r}]] = data[[{x!r}, {y!r}]].apply(pd.to_numeric, errors='coerce')",
        "data = data.dropna()",
        "data.columns = ['x', 'y', 'group']",
        "data['group'] = data['group'].astype(str)",
        "",
        *_trend_function(),
        "",
        "fig = go.Figure()",
        "for level in sorted(data['group'].unique()):",
        "    part = data[data['group'] == level]",
        "    fig.add_scatter(x=part['x'], y=part['y'], mode='markers', name=level)",
        "    if len(part) >= 3 and part['x'].nunique() > 1 and part['y'].nunique() > 1:",
        "        line_x, line_y = trend(part)",
        "        fig.add_scatter(x=line_x, y=line_y, mode='lines', name=f'{level} trend')",
        "line_x, line_y = trend(data)",
        "fig.add_scatter(x=line_x, y=line_y, mode='lines', name='All rows', line={'dash': 'dash'})",
        f"fig.update_layout(xaxis_title={x!r}, yaxis_title={y!r}, legend_title={g!r})",
    ]


def _line_code(candidate: Mapping[str, Any], df: pd.DataFrame) -> list[str]:
    x, y = candidate["x"], candidate["y"]
    lines = [
        f"data = df[[{x!r}, {y!r}]].copy()",
        f"data[{y!r}] = pd.to_numeric(data[{y!r}], errors='coerce')",
    ]

    if pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_bool_dtype(df[x]):
        lines += [
            f"data[{x!r}] = pd.to_numeric(data[{x!r}], errors='coerce')",
            f"data = data.dropna(subset=[{x!r}, {y!r}])",
            f"data = data.groupby({x!r}, as_index=False)[{y!r}].mean().sort_values({x!r})",
        ]
        rows = df[x].nunique()
    else:
        parsed = pd.DataFrame(
            {
                "x": pd.to_datetime(df[x], errors="coerce"),
                "y": pd.to_numeric(df[y], errors="coerce"),
            }
        ).dropna()
        unique = parsed["x"].nunique()
        span = (parsed["x"].max() - parsed["x"].min()).days if len(parsed) else 0

        lines += [
            f"data[{x!r}] = pd.to_datetime(data[{x!r}], errors='coerce')",
            f"data = data.dropna(subset=[{x!r}, {y!r}]).sort_values({x!r})",
        ]

        if unique <= 100:
            lines.append(
                f"data = data.groupby({x!r}, as_index=False)[{y!r}].mean().sort_values({x!r})"
            )
            rows = unique
        else:
            rule, label = (
                ("MS", "monthly") if span > 365 or unique > 180 else ("W", "weekly")
            )
            lines.append(
                f"data = data.set_index({x!r})[{y!r}].resample({rule!r}).mean()"
                f".dropna().reset_index()  # {label} average"
            )
            rows = 0  # resampled: no markers

    markers = 0 < rows <= 60
    lines.append(f"fig = px.line(data, x={x!r}, y={y!r}, markers={markers})")
    return lines


def _histogram_code(candidate: Mapping[str, Any], df: pd.DataFrame) -> list[str]:
    x = candidate["x"]
    full = pd.to_numeric(df[x], errors="coerce").dropna()
    display, info = prepare_histogram_display_data(full)
    bins = histogram_bin_count(display)

    lines = [f"values = pd.to_numeric(df[{x!r}], errors='coerce').dropna()"]

    if info["trimmed"]:
        conditions = []
        if info["trim_lower"]:
            conditions.append(f"(values >= {info['lower_bound']!r})")
        if info["trim_upper"]:
            conditions.append(f"(values <= {info['upper_bound']!r})")
        lines.append(
            f"values = values[{' & '.join(conditions)}]  "
            f"# {info['excluded_total']} extreme values hidden from the display"
        )

    lines.append(f"fig = px.histogram(values.to_frame(), x={x!r}, nbins={bins})")
    return lines


def _box_code(candidate: Mapping[str, Any], df: pd.DataFrame) -> list[str]:
    x, y = candidate["x"], candidate["y"]
    clean = df[[x, y]].copy()
    clean[y] = pd.to_numeric(clean[y], errors="coerce")
    clean = clean.dropna(subset=[x, y])
    order = [_python_value(v) for v in box_category_order(clean, x, y)]

    return [
        f"data = df[[{x!r}, {y!r}]].copy()",
        f"data[{y!r}] = pd.to_numeric(data[{y!r}], errors='coerce')",
        f"data = data.dropna(subset=[{x!r}, {y!r}])",
        f"fig = px.box(data, x={x!r}, y={y!r}, points='outliers',",
        f"             category_orders={{{x!r}: {order!r}}})",
    ]


def _bar_code(candidate: Mapping[str, Any], df: pd.DataFrame) -> list[str]:
    x = candidate["x"]

    if candidate.get("aggregation", "count") == "count":
        sort = (
            f"sort_values({x!r})"
            if pd.api.types.is_numeric_dtype(df[x].dropna())
            else "sort_values('count', ascending=False)"
        )
        return [
            f"counts = df[{x!r}].dropna().value_counts().rename_axis({x!r})"
            ".reset_index(name='count')",
            f"counts = counts.{sort}",
            f"fig = px.bar(counts, x={x!r}, y='count')",
        ]

    y = candidate["y"]
    sort = (
        f"sort_values({x!r})"
        if pd.api.types.is_numeric_dtype(df[x].dropna())
        else f"sort_values({y!r}, ascending=False)"
    )
    return [
        f"data = df[[{x!r}, {y!r}]].copy()",
        _numeric_conversion(df[y], y),
        f"data = data.dropna(subset=[{x!r}, {y!r}])",
        f"means = data.groupby({x!r}, as_index=False)[{y!r}].mean().{sort}",
        f"fig = px.bar(means, x={x!r}, y={y!r})",
    ]


def _python_value(value: Any) -> Any:
    """numpy scalars -> plain Python values so repr() is valid code."""

    return value.item() if hasattr(value, "item") else value


def chart_code(candidate: Mapping[str, Any], source_name: str, df: pd.DataFrame) -> str:
    """
    A self-contained pandas + Plotly snippet that rebuilds the chart from the
    CSV called source_name. Only pandas and plotly are imported (numpy is not
    needed); no visift import.
    """

    chart = candidate["chart"]
    title = visualization_title(candidate)
    extra: list[str] = []

    if chart == "scatter" and candidate.get("group"):
        body = _grouped_scatter_code(candidate)
        extra = ["import plotly.graph_objects as go"]
        title = (
            f"{candidate['x']} vs {candidate['y']} by {candidate['group']} "
            "(Simpson's paradox view)"
        )
    elif chart == "scatter":
        body = _scatter_code(candidate, df)
    elif chart == "line":
        body = _line_code(candidate, df)
    elif chart == "histogram":
        body = _histogram_code(candidate, df)
    elif chart == "box":
        body = _box_code(candidate, df)
    elif chart == "bar":
        body = _bar_code(candidate, df)
    else:
        raise ValueError(f"Unsupported chart type: {chart!r}")

    return "\n".join([*_header(title, source_name, extra), *body, "fig.show()"]) + "\n"


# --------------------------------------------------
# HTML report
# --------------------------------------------------

_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
       max-width: 1000px; padding: 0 1rem; color: #111; }
h1 { margin-bottom: 0.2rem; } .stats { color: #555; margin-top: 0; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 1rem 1.2rem;
        margin: 1.2rem 0; }
.badge { display: inline-block; border-radius: 6px; padding: 0 0.45rem;
         font-size: 0.85rem; background: #eef; margin-right: 0.3rem; }
.badge.paradox { background: #fde7c8; } .score { float: right; font-weight: 700; }
.insight { background: #f3f8f3; border-radius: 6px; padding: 0.6rem 0.8rem; }
.formula { color: #555; font-size: 0.9rem; }
footer { color: #777; font-size: 0.85rem; margin-top: 2rem; }
"""


def _figure_html(fig: Any, first: bool) -> str:
    # plotly.js is embedded once (with the first figure) so the file works
    # offline without repeating the ~3.5 MB library.
    return str(fig.to_html(full_html=False, include_plotlyjs=bool(first)))


def html_report(
    df: pd.DataFrame,
    *,
    source_name: str,
    stats_line: str,
    target: str | None,
    paradoxes: Sequence[Mapping[str, Any]],
    recommendations: Sequence[Mapping[str, Any]],
    notes: Sequence[Mapping[str, Any]],
    insight: Callable[[pd.DataFrame, Mapping[str, Any]], str] | None = None,
    now: datetime | None = None,
) -> str:
    """
    One self-contained HTML page: dataset stats, the selected outcome,
    "Worth a closer look", the given recommendations (in order, as shown) and
    the dataset notes. Works offline.
    """

    esc = html.escape
    parts: list[str] = []
    first = True

    def add_figure(fig: Any) -> None:
        nonlocal first
        parts.append(_figure_html(fig, first))
        first = False

    parts.append(f"<h1>Visift report: {esc(source_name)}</h1>")
    parts.append(f"<p class='stats'>{esc(stats_line)}</p>")
    parts.append(
        f"<p>Outcome in focus: <b>{esc(target)}</b></p>"
        if target
        else "<p>No outcome selected (general ranking).</p>"
    )

    if paradoxes:
        parts.append("<h2>Worth a closer look</h2>")
        for paradox in paradoxes:
            grouped_chart = {
                "chart": "scatter",
                "x": paradox["x"],
                "y": paradox["y"],
                "group": paradox["group"],
            }
            parts.append("<div class='card'>")
            parts.append(
                f"<h3>{esc(str(paradox['x']))} vs {esc(str(paradox['y']))} by "
                f"{esc(str(paradox['group']))}</h3>"
                "<span class='badge paradox'>Simpson's paradox</span>"
            )
            parts.append(f"<p class='insight'>{esc(paradox['sentence'])}</p>")
            add_figure(render_candidate(df, grouped_chart))
            parts.append("</div>")

    parts.append("<h2>Recommended visualizations</h2>")

    for rank, candidate in enumerate(recommendations, start=1):
        score = float(candidate["score"])
        badges = [
            recommendation_strength(score),
            chart_display_name(candidate["chart"]),
        ]
        parts.append("<div class='card'>")
        parts.append(
            f"<span class='score'>{score:.1f}</span>"
            f"<h3>#{rank}: {esc(visualization_title(candidate))}</h3>"
            + "".join(f"<span class='badge'>{esc(b)}</span>" for b in badges)
            + (
                "<span class='badge paradox'>Simpson's paradox</span>"
                if candidate.get("paradox")
                else ""
            )
        )
        if insight is not None:
            parts.append(f"<p class='insight'>{esc(insight(df, candidate))}</p>")
        parts.append(
            f"<p class='formula'>{esc(score_breakdown(candidate)['formula'])}</p>"
        )
        reasons = summary_reasons(candidate)
        if reasons:
            parts.append(
                "<ul>" + "".join(f"<li>{esc(r)}</li>" for r in reasons) + "</ul>"
            )
        add_figure(render_candidate(df, candidate))
        parts.append("</div>")

    if notes:
        parts.append("<h2>Dataset notes</h2><ul>")
        for note in notes:
            value = float(note["value"])
            stat = (
                f" ({esc(str(note['statistic']))} = {value:.3f})"
                if math.isfinite(value)
                else ""
            )
            parts.append(f"<li>{esc(str(note['reason']))}{stat}</li>")
        parts.append("</ul>")

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"<footer>Generated by Visift {esc(visift.__version__)} on {stamp} · "
        f"<a href='{REPO_URL}'>{REPO_URL}</a></footer>"
    )

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Visift report: {esc(source_name)}</title><style>{_STYLE}</style>"
        "</head><body>" + "\n".join(parts) + "</body></html>"
    )
