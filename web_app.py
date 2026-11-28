import secrets
import time
import uuid
from html import escape as html_escape
from pathlib import Path

import pandas as pd
import streamlit as st

from visift.explain import SUITABILITY_WEIGHTS, score_breakdown
from visift.export import chart_code, html_report
from visift.feedback import REASONS, default_store, feedback_enabled, make_event
from visift.insight import plain_english_insight
from visift.io import MAX_COLUMNS, MAX_UPLOAD_MB, UploadError, load_table
from visift.pipeline import analyze
from visift.profiler import profile_dataset
from visift.scoring.engine import (
    chart_display_name,
    recommendation_strength,
    summary_reasons,
)
from visift.target import split_by_target, suggest_target
from visift.visualization.renderer import (
    humanize_column_name,
    is_currency_column,
    render_candidate,
    visualization_title,
)


def apply_custom_theme():
    """
    Apply a dark neon-green UI theme to Streamlit.
    """

    st.markdown(
        """
        <style>

        /* =========================
           Global app
        ========================= */

        .stApp {
            background-color: #000000;
            color: #E8FFE8;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .visift-header {
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
            gap: 0.4rem 0.9rem;
            margin-bottom: 0.25rem;
        }

        .visift-title {
            font-size: 2rem;
            font-weight: 800;
            color: #39FF14;
            letter-spacing: -0.02em;
        }

        .visift-tagline {
            font-size: 1.05rem;
            color: #9CB39C;
        }

        .visift-scorebar {
            display: flex;
            width: 100%;
            height: 14px;
            border-radius: 7px;
            overflow: hidden;
            background: #111111;
            margin: 0.35rem 0 0.25rem 0;
        }

        .visift-scorebar span {
            display: block;
            height: 100%;
        }


        /* =========================
           Sidebar
        ========================= */

        section[data-testid="stSidebar"] {
            background-color: #050505;
            border-right: 1px solid #1A1A1A;
        }

        section[data-testid="stSidebar"] * {
            color: #E8FFE8 !important;
        }


        /* =========================
           Headers / text
        ========================= */

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {
            color: #E8FFE8 !important;
            font-weight: 700 !important;
        }

        p,
        li,
        label {
            color: #E8FFE8;
        }

        small {
            color: #A5B8A5 !important;
        }

        .stCaption {
            color: #93A593 !important;
        }


        /* =========================
           Links
        ========================= */

        a {
            color: #39FF14 !important;
        }

        a:hover {
            color: #7CFF5B !important;
        }


        /* =========================
           Tabs
        ========================= */

        button[data-baseweb="tab"] {
            color: #A5B8A5 !important;
            background: transparent !important;
        }

        button[data-baseweb="tab"]:hover {
            color: #E8FFE8 !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #39FF14 !important;
            border-bottom: 2px solid #39FF14 !important;
        }


        /* =========================
           Buttons
        ========================= */

        .stButton > button {
            background-color: #0A0A0A;
            color: #39FF14;
            border: 1px solid #39FF14;
            border-radius: 8px;
        }

        .stButton > button:hover {
            background-color: #102010;
            color: #7CFF5B;
            border-color: #7CFF5B;
        }


        /* =========================
           Radio buttons
        ========================= */

        div[role="radiogroup"] label {
            color: #E8FFE8 !important;
        }


        /* =========================
           Inputs / dropdowns
        ========================= */

        div[data-baseweb="select"] > div {
            background-color: #050505 !important;
            border: 1px solid #242424 !important;
            color: #E8FFE8 !important;
        }

        div[data-baseweb="select"] input {
            color: #E8FFE8 !important;
        }

        input,
        textarea {
            background-color: #0A0A0A !important;
            color: #E8FFE8 !important;
        }


       /* =========================
        Multiselect chart-type tags
        ========================= */

        /* Entire selected tag */
        div[data-testid="stMultiSelect"] [data-baseweb="tag"],
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background: #0B0B0B !important;
            background-color: #0B0B0B !important;
            background-image: none !important;

            border: 1px solid #39FF14 !important;
            border-radius: 6px !important;

            box-shadow: none !important;
        }

        /* All text inside the selected tag */
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
            color: #39FF14 !important;
            font-weight: 600 !important;
        }

        /* Close / X button */
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] button {
            background: transparent !important;
            background-color: transparent !important;
            color: #39FF14 !important;
            border: none !important;
        }

        /* X icon */
        div[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
            fill: #39FF14 !important;
            color: #39FF14 !important;
        }

        /* Hover */
        div[data-testid="stMultiSelect"] [data-baseweb="tag"]:hover {
            background: #102010 !important;
            background-color: #102010 !important;
            border-color: #7CFF5B !important;
        }

        /* Main multiselect box */
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
            background-color: #050505 !important;
            border: 1px solid #242424 !important;
        }


        /* =========================
           Dropdown menu
        ========================= */

        div[data-baseweb="popover"] {
            background-color: #080808 !important;
        }

        div[data-baseweb="menu"] {
            background-color: #080808 !important;
        }

        li[role="option"] {
            background-color: #080808 !important;
            color: #E8FFE8 !important;
        }

        li[role="option"]:hover {
            background-color: #102010 !important;
            color: #39FF14 !important;
        }

        li[role="option"][aria-selected="true"] {
            background-color: #0D1A0D !important;
            color: #39FF14 !important;
        }


        /* =========================
           Sliders
        ========================= */

        .stSlider * {
            color: #E8FFE8 !important;
        }

        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #39FF14 !important;
            border-color: #39FF14 !important;
        }


        /* =========================
           Metric cards
        ========================= */

        div[data-testid="stMetric"] {
            background-color: #070707;
            border: 1px solid #1A1A1A;
            border-left: 3px solid #39FF14;
            padding: 12px;
            border-radius: 10px;
        }

        div[data-testid="stMetricLabel"] {
            color: #A5B8A5 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #E8FFE8 !important;
            font-weight: 700;
        }


        /* =========================
           Containers / cards
        ========================= */

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #030303;
            border-color: #1D1D1D !important;
            border-radius: 10px;
        }


        /* =========================
           Expanders
        ========================= */

        details {
            background-color: #050505 !important;
            border: 1px solid #1A1A1A !important;
            border-radius: 8px !important;
        }

        details summary {
            color: #E8FFE8 !important;
        }

        details summary:hover {
            color: #39FF14 !important;
        }


        /* =========================
           Dataframes / tables
        ========================= */

        .stDataFrame,
        .stTable {
            background-color: #050505;
            border-radius: 8px;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #1A1A1A;
            border-radius: 8px;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] div {
            color: #E8FFE8 !important;
        }


        /* =========================
           Alerts / insight boxes
        ========================= */

        div[data-testid="stAlert"] {
            background-color: #071107 !important;
            border: 1px solid #1F6F1F !important;
            color: #E8FFE8 !important;
            border-radius: 8px;
        }

        div[data-testid="stAlert"] p {
            color: #DFFFF0 !important;
        }


        /* =========================
           File uploader
        ========================= */

        section[data-testid="stFileUploaderDropzone"] {
            background-color: #070707 !important;
            border: 1px dashed #39FF14 !important;
        }

        section[data-testid="stFileUploaderDropzone"] * {
            color: #E8FFE8 !important;
        }


        /* =========================
           Code blocks
        ========================= */

        code {
            color: #39FF14 !important;
        }

        pre {
            background-color: #050505 !important;
            border: 1px solid #1A1A1A !important;
            color: #E8FFE8 !important;
        }


        /* =========================
           Horizontal rules
        ========================= */

        hr {
            border-color: #1A1A1A !important;
        }


        /* =========================
           Scrollbars
        ========================= */

        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #050505;
        }

        ::-webkit-scrollbar-thumb {
            background: #292929;
            border-radius: 8px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #39FF14;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Visift", page_icon="📊", layout="wide", initial_sidebar_state="expanded"
)


# ==================================================
# DISPLAY HELPERS
# ==================================================


def strength_color(strength):
    """
    Map recommendation strength to a
    Streamlit badge color.
    """

    colors = {
        "Excellent": "green",
        "Strong": "blue",
        "Moderate": "orange",
        "Weak": "gray",
        "Very weak": "red",
    }

    return colors.get(strength, "gray")


def format_semantic_type(value):
    """
    Convert semantic-type identifiers into
    clean display text.
    """

    return str(value).replace("_", " ").title()


# ==================================================
# DATA LOADING
# ==================================================


APP_DIR = Path(__file__).resolve().parent

UPLOAD_OPTION = "Upload your own CSV"


def plotly_config(title):
    """
    Chart toolbar settings. The camera button downloads a 2x PNG named after
    the chart (rendered in the browser; no server-side image export).
    """

    slug = "".join(c if c.isalnum() else "-" for c in str(title).lower()).strip("-")

    return {
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": slug or "visift-chart",
            "scale": 2,
        },
    }


def export_csv_name():
    """File name the exported code reads (the sample's file or the upload)."""

    return st.session_state.get("export_csv_name", "data.csv")


# One-click sample datasets: label -> (file, outcome to preselect, hook, credit).
SAMPLES = {
    "Palmer Penguins": (
        "src/visift/samples/penguins.csv",
        "species",
        "A relationship that flips within species (Simpson's paradox).",
        "Gorman, Williams & Fraser (2014), palmerpenguins, CC0.",
    ),
    "Titanic": (
        "src/visift/samples/titanic.csv",
        "survived",
        "Who survived?",
        "titanic3 by Thomas Cason, from hbiostat.org/data courtesy of "
        "Vanderbilt Biostatistics.",
    ),
    "Bike Sharing (daily)": (
        "src/visift/samples/bike_sharing_day.csv",
        "cnt",
        "What drives daily rentals?",
        "Fanaee-T (2013), UCI Machine Learning Repository, CC BY 4.0.",
    ),
    "Bank Marketing (5,000-row sample)": (
        "src/visift/samples/bank_marketing_sample.csv",
        "y",
        "Who subscribes to a term deposit?",
        "Moro, Rita & Cortez (2014), UCI Machine Learning Repository, CC BY 4.0.",
    ),
    "Synthetic sales": (
        "synthetic_sales.csv",
        None,
        "Generated data with planted relationships.",
        "Generated by scripts/generate_synthetic_sales.py.",
    ),
}

DEFAULT_SAMPLE = "Palmer Penguins"


@st.cache_data(show_spinner=False)
def read_sample(path):
    return pd.read_csv(APP_DIR / path)


@st.cache_data(show_spinner=False, max_entries=4)
def read_upload(data, filename, sheet=None):
    """Cached load_table, so filter changes don't re-parse the upload."""

    return load_table(data, filename, sheet=sheet)


def show_upload_error(error):
    st.error(error.message)

    if error.hint:
        st.caption(error.hint)

    with st.expander("Technical details"):
        st.code(error.detail or error.message, language=None)


def load_upload():
    """
    Upload widget plus everything around it: sheet choice for workbooks,
    a column picker for very wide tables, notices, friendly errors.
    Returns (df, file name); stops the script until a usable file is loaded.
    """

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["csv", "tsv", "txt", "xlsx", "xlsm"],
        help=(
            "CSV, TSV, TXT (comma, semicolon, tab or pipe separated) or Excel "
            f".xlsx, up to {MAX_UPLOAD_MB} MB. Files over 100,000 rows are "
            "analyzed on a random sample."
        ),
    )

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to begin.")

        st.stop()

    data = uploaded_file.getvalue()

    # Load with Visift's default choice first (for workbooks: the first sheet
    # that holds a table). Even if that fails, the sheet picker is shown
    # before the error, so a different sheet can always be chosen.
    try:
        table = read_upload(data, uploaded_file.name)
        error = None
        sheet_names = table["sheet_names"]
        default_sheet = table["sheet"]
    except UploadError as caught:
        table = None
        error = caught
        sheet_names = caught.sheet_names
        default_sheet = sheet_names[0] if sheet_names else None

    if len(sheet_names) > 1:
        sheet = st.selectbox(
            "Sheet",
            options=sheet_names,
            index=sheet_names.index(default_sheet),
            key=f"sheet::{uploaded_file.name}",
        )

        if table is None or sheet != table["sheet"]:
            try:
                table = read_upload(data, uploaded_file.name, sheet)
                error = None
            except UploadError as caught:
                table = None
                error = caught

    if table is None:
        show_upload_error(error)

        st.stop()

    df = table["df"]

    if df.shape[1] > MAX_COLUMNS:
        st.warning(
            f"This table has {df.shape[1]} columns. Wide tables are slow to "
            f"analyze, so Visift uses up to {MAX_COLUMNS} at a time."
        )

        columns = st.multiselect(
            "Columns to analyze",
            options=list(df.columns),
            default=list(df.columns[:MAX_COLUMNS]),
            max_selections=MAX_COLUMNS,
            key=f"columns::{uploaded_file.name}",
        )

        if len(columns) < 2:
            st.info("Pick at least two columns to analyze.")

            st.stop()

        df = df[columns]

    for notice in table["notices"]:
        st.caption(f"ℹ️ {notice}")

    return (df, uploaded_file.name)


def load_data():
    """
    Sidebar dataset picker: one-click samples or an uploaded CSV.

    Returns (df, source_name, outcome to preselect or None).
    """

    with st.sidebar:
        st.header("Dataset")

        options = [*SAMPLES, UPLOAD_OPTION]

        choice = st.selectbox(
            "Choose a dataset", options=options, index=options.index(DEFAULT_SAMPLE)
        )

        if choice == UPLOAD_OPTION:
            return (*load_upload(), None)

        path, outcome, hook, credit = SAMPLES[choice]

        st.caption(f"{hook}  \n*{credit}*")

        try:
            df = read_sample(path)

        except FileNotFoundError:
            st.error(f"{path} could not be found.")

            st.stop()

    return (df, choice, outcome)


# ==================================================
# ENGINE
# ==================================================


@st.cache_data(show_spinner=False)
def profile_data(df):
    """
    Profile the dataset (cached; needed before analysis for the target picker).
    """

    return profile_dataset(df)


ANALYSIS_CACHE_SIZE = 4


def analyze_dataset(df, target=None, progress=None):
    """
    Run the full recommendation pipeline.

    Results are cached per session so changing interface filters does not
    rerun the analysis. This is a plain session-state cache rather than
    st.cache_data because the progress callback updates a Streamlit element
    created outside the function, which st.cache_data forbids
    (CacheReplayClosureError on a real server).
    """

    key = (
        int(pd.util.hash_pandas_object(df, index=True).sum()),
        tuple(map(str, df.columns)),
        tuple(map(str, df.dtypes)),
        target,
    )

    cache = st.session_state.setdefault("_visift_analysis_cache", {})

    if key in cache:
        return cache[key]

    start = time.perf_counter()

    analysis = analyze(df, target=target, progress=progress)

    elapsed = time.perf_counter() - start

    result = (
        analysis["profile"],
        analysis["candidates"],
        analysis["recommendations"],
        analysis["notes"],
        analysis["noted_charts"],
        elapsed,
        analysis["paradoxes"],
    )

    while len(cache) >= ANALYSIS_CACHE_SIZE:
        cache.pop(next(iter(cache)))

    cache[key] = result

    return result


# ==================================================
# HEADER
# ==================================================


def display_header():
    """
    One-line header: name and what it does.
    """

    st.markdown(
        "<div class='visift-header'><span class='visift-title'>Visift</span>"
        "<span class='visift-tagline'>ranks the charts worth looking at first "
        "in a dataset, with the evidence for each</span></div>",
        unsafe_allow_html=True,
    )


# ==================================================
# DATASET OVERVIEW
# ==================================================


def display_dataset_metrics(
    df, profile, candidates, recommendations, elapsed=None, source_name=None
):
    """
    One compact line of dataset and engine facts.
    """

    missing_values = int(df.isna().sum().sum())

    top_score = recommendations[0]["score"] if recommendations else 0

    parts = [
        f"**{source_name}**" if source_name else None,
        f"{profile['rows']:,} rows",
        f"{profile['columns']} columns",
        f"{len(candidates)} chart candidates",
        f"top score {top_score:.1f}",
        f"{missing_values:,} missing values",
        f"analyzed in {elapsed:.1f} s" if elapsed is not None else None,
    ]

    line = " · ".join(part for part in parts if part)

    st.caption(line)

    return line.replace("**", "")


def build_column_profile_table(profile):
    """
    Convert column profiles into a cleaner
    dataframe for display.
    """

    rows = []

    for column in profile["column_profiles"]:
        rows.append(
            {
                "Column": (column["name"]),
                "Semantic Type": (format_semantic_type(column["semantic_type"])),
                "Unique Values": (column["unique_count"]),
                "Missing": (column["missing_count"]),
                "Missing %": (f"{column['missing_percent']:.1f}%"),
                "Pandas Type": (column["pandas_dtype"]),
            }
        )

    return pd.DataFrame(rows)


def build_display_preview(df, profile, rows=100):
    """
    Build a display-only dataframe with
    friendlier dates, currency formatting,
    and numeric formatting.

    The original dataframe remains unchanged.
    """

    preview = df.head(rows).copy()

    semantic_types = {
        column["name"]: (column["semantic_type"])
        for column in profile["column_profiles"]
    }

    for column in preview.columns:
        semantic_type = semantic_types.get(column)

        original_series = df[column]

        # -----------------------------------
        # Datetime display
        # -----------------------------------

        if semantic_type == "datetime":
            parsed = pd.to_datetime(preview[column], errors="coerce")

            preview[column] = parsed.dt.strftime("%b %d, %Y")

        # -----------------------------------
        # Currency display
        # -----------------------------------

        elif semantic_type in {
            "numeric_continuous",
            "numeric_discrete",
        } and is_currency_column(column):
            numeric = pd.to_numeric(preview[column], errors="coerce")

            preview[column] = numeric.map(
                lambda value: f"${value:,.2f}" if pd.notna(value) else None
            )

        # -----------------------------------
        # Integer numeric display
        # -----------------------------------

        elif pd.api.types.is_integer_dtype(original_series):
            numeric = pd.to_numeric(preview[column], errors="coerce")

            preview[column] = numeric.map(
                lambda value: f"{int(value):,}" if pd.notna(value) else None
            )

        # -----------------------------------
        # Continuous numeric display
        # -----------------------------------

        elif semantic_type == "numeric_continuous":
            numeric = pd.to_numeric(preview[column], errors="coerce")

            preview[column] = numeric.map(
                lambda value: f"{value:,.2f}" if pd.notna(value) else None
            )

    return preview


# ==================================================
# SCORE DISPLAY
# ==================================================


def display_score_breakdown(candidate):
    """
    The score formula with this chart's numbers: the same math as the bar.
    """

    breakdown = score_breakdown(candidate)
    components = candidate["components"]

    rows = [
        {
            "Part": label,
            "Value": round(float(components[key]), 1),
            "Weight": weight,
            "Contribution": round(weight * float(components[key]), 1),
        }
        for key, label, weight in SUITABILITY_WEIGHTS
    ]

    rows.append(
        {
            "Part": "= Chart suitability",
            "Value": round(breakdown["suitability"], 1),
            "Weight": None,
            "Contribution": None,
        }
    )

    st.markdown("**Chart suitability** (is this a good chart for these columns?)")

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown(
        f"**Pattern strength** {breakdown['pattern_strength']:.1f}/100 gives a "
        f"pattern factor of 0.35 + 0.65 × {breakdown['pattern_strength']:.1f}/100 "
        f"= **{breakdown['pattern_factor']:.2f}**. Even no pattern keeps 35% of "
        "the suitability, which is why pure noise lands around 35."
    )

    st.markdown(f"**Score:** {breakdown['formula']}")


# ==================================================
# RECOMMENDATION CARD
# ==================================================


# ==================================================
# MY BOARD (pinned charts)
# ==================================================

MAX_PINS = 4


def chart_key(chart):
    """Stable identity of a chart: type, columns, aggregation, grouping."""

    return "|".join(
        str(chart.get(part) or "")
        for part in ("chart", "x", "y", "aggregation", "group")
    )


def board_pins(dataset_key):
    """
    Pinned charts for the current dataset (a list of snapshots). Pins
    survive outcome and filter changes and reset when the dataset changes.
    """

    board = st.session_state.get("board")

    if board is None or board["dataset"] != dataset_key:
        board = {"dataset": dataset_key, "pins": []}
        st.session_state["board"] = board

    return board["pins"]


def toggle_pin(dataset_key, snapshot):
    pins = board_pins(dataset_key)
    key = chart_key(snapshot["chart"])

    for index, pin in enumerate(pins):
        if chart_key(pin["chart"]) == key:
            del pins[index]
            return

    if len(pins) >= MAX_PINS:
        st.toast(
            f"You can pin up to {MAX_PINS} charts. Unpin one on the My board tab "
            "first.",
            icon="📌",
        )
        return

    pins.append(snapshot)


def pin_button(snapshot, context):
    """Pin/Unpin toggle for one chart (context keeps widget keys unique)."""

    dataset_key = st.session_state.get("export_csv_name", "")
    key = chart_key(snapshot["chart"])
    pinned = any(chart_key(pin["chart"]) == key for pin in board_pins(dataset_key))

    st.button(
        "Unpin from board" if pinned else "📌 Pin to board",
        key=f"pin::{context}::{key}",
        on_click=toggle_pin,
        args=(dataset_key, snapshot),
        help="Collect up to 4 charts on the My board tab to compare them.",
    )


# ==================================================
# FEEDBACK (thumbs on recommendations)
# ==================================================


def feedback_session_id():
    """A random id per app session (not an identity)."""

    if "feedback_session" not in st.session_state:
        st.session_state["feedback_session"] = uuid.uuid4().hex

    return st.session_state["feedback_session"]


def feedback_salt():
    """Random per session; salts uploaded column names and is never stored."""

    if "feedback_salt" not in st.session_state:
        st.session_state["feedback_salt"] = secrets.token_hex(16)

    return st.session_state["feedback_salt"]


def record_vote(widget_key, reason_key, candidate, rank):
    """on_change for the thumbs and the reason pills: upsert or clear."""

    context = st.session_state.get("feedback_context", {})
    value = st.session_state.get(widget_key)  # 1 = up, 0 = down, None = cleared
    reason = st.session_state.get(reason_key) if value == 0 else None

    event = make_event(
        session_id=feedback_session_id(),
        salt=feedback_salt(),
        dataset=context.get("dataset", ""),
        uploaded=context.get("uploaded", False),
        target=context.get("target"),
        candidate=candidate,
        rank=rank,
        vote=1 if value == 1 else -1,
        reason=reason,
    )

    store = default_store()

    if value is None:
        store.clear(event)
    else:
        store.record(event)


def feedback_controls(candidate, rank):
    """Thumbs up/down (and a reason after a thumbs down) for one card."""

    if not feedback_enabled():
        return

    context = st.session_state.get("feedback_context", {})
    base = (
        f"{context.get('dataset', '')}::{context.get('target')}::{chart_key(candidate)}"
    )
    widget_key = f"vote::{base}"
    reason_key = f"reason::{base}"

    vote = st.feedback(
        "thumbs",
        key=widget_key,
        on_change=record_vote,
        args=(widget_key, reason_key, dict(candidate), rank),
    )

    if vote == 0:
        st.pills(
            "What's wrong with it? (optional)",
            REASONS,
            key=reason_key,
            on_change=record_vote,
            args=(widget_key, reason_key, dict(candidate), rank),
        )

    if vote is not None:
        st.caption("Thanks, this helps tune the ranking.")


SEGMENT_COLORS = {
    "semantic_fit": "#2B2F2B",
    "readability": "#363B36",
    "data_quality": "#414741",
    "sample_support": "#4C534C",
    "pattern": "#1E4D16",
    "score": "#39FF14",
}


def score_bar_html(breakdown):
    """
    A stacked bar from 0 to 100: the score (accent) followed by what each
    part of the formula cost (muted). Hover a segment for its explanation;
    aria-label carries the formula as text.
    """

    ordered = [breakdown["segments"][-1], *breakdown["segments"][:-1]]
    spans = "".join(
        f"<span style='width:{segment['value']:.3f}%;"
        f"background:{SEGMENT_COLORS[segment['key']]}' "
        f"title='{html_escape(segment['detail'])}'></span>"
        for segment in ordered
        if segment["value"] > 0
    )

    return (
        f"<div class='visift-scorebar' role='img' "
        f"aria-label='{html_escape(breakdown['formula'])}'>{spans}</div>"
    )


def display_recommendation(df, candidate, rank):
    """
    Render one polished recommendation card.
    """

    title = visualization_title(candidate)

    chart_name = chart_display_name(candidate["chart"])

    score = candidate["score"]

    strength = recommendation_strength(score)

    color = strength_color(strength)

    alternatives = candidate.get("alternative_charts", [])

    with st.container(border=True):
        # -----------------------------------
        # Header
        # -----------------------------------

        header_left, header_right = st.columns([6, 1])

        with header_left:
            st.markdown(f"### #{rank} — {title}")

            badges = f":{color}-badge[{strength}] :gray-badge[{chart_name}]"

            if candidate.get("paradox"):
                badges += " :orange-badge[Simpson's paradox]"

            st.markdown(badges)

        with header_right:
            st.metric("Score", f"{score:.1f}")

        # -----------------------------------
        # Plain-English takeaway
        # -----------------------------------

        insight = plain_english_insight(df, candidate)

        # Escape "$" so Streamlit Markdown doesn't read it as math.
        st.info(insight.replace("$", "\\$"), icon="💡")

        # -----------------------------------
        # Visualization
        # -----------------------------------

        fig = render_candidate(df, candidate)

        st.plotly_chart(fig, width="stretch", config=plotly_config(title))

        # -----------------------------------
        # How the score is built
        # -----------------------------------

        breakdown = score_breakdown(candidate)

        st.markdown(score_bar_html(breakdown), unsafe_allow_html=True)

        caption = breakdown["formula"]

        if alternatives:
            caption += (
                f" · {len(alternatives)} alternative "
                f"view{'' if len(alternatives) == 1 else 's'}"
            )

        st.caption(caption)

        pin_column, feedback_column = st.columns([1, 2])

        with pin_column:
            pin_button({"kind": "recommendation", "chart": dict(candidate)}, "card")

        with feedback_column:
            feedback_controls(candidate, rank)

        # -----------------------------------
        # Details
        # -----------------------------------

        with st.expander("Recommendation details"):
            evidence_tab, scoring_tab, code_tab = st.tabs(
                ["Evidence", "Score breakdown", "Python code"]
            )

            with evidence_tab:
                st.markdown("#### Key evidence")

                for reason in summary_reasons(candidate):
                    st.write(f"• {reason}")

                if alternatives:
                    st.markdown("#### Alternative views")

                    for alternative in alternatives:
                        if "score" in alternative:
                            st.write(
                                f"• "
                                f"{alternative['chart_name']} "
                                f"— "
                                f"{alternative['score']:.2f}/100"
                            )

                        else:
                            st.write(
                                f"• {alternative['chart_name']} — grouped view "
                                "(see Worth a closer look)"
                            )

            with scoring_tab:
                st.markdown("#### Component scores")

                display_score_breakdown(candidate)

            with code_tab:
                st.caption(
                    "pandas + Plotly code that rebuilds this chart's data "
                    "(styling left out). Copy it with the button on the code."
                )

                st.code(chart_code(candidate, export_csv_name(), df), language="python")


# ==================================================
# SIDEBAR FILTERS
# ==================================================


NON_TARGET_TYPES = {"identifier", "text", "empty", "unknown"}


def target_control(profile, default_target=None, dataset_key=""):
    """
    Optional outcome column. Defaults to none (or to the sample's known
    outcome); the suggestion is only a hint. The widget is keyed by dataset
    so switching datasets resets it to that dataset's default.
    """

    columns = [
        column["name"]
        for column in profile["column_profiles"]
        if column["semantic_type"] not in NON_TARGET_TYPES
    ]

    suggested, reason = suggest_target(profile)

    options = [None, *columns]
    index = options.index(default_target) if default_target in options else 0

    with st.sidebar:
        target = st.selectbox(
            "Focus on an outcome (optional)",
            options=options,
            index=index,
            key=f"target::{dataset_key}",
            format_func=lambda column: "None" if column is None else column,
            help=(
                "Pick the column you want to explain. Charts involving it are "
                "listed first as its drivers, ranked by their usual score."
            ),
        )

        if suggested is not None:
            st.caption(f"Suggested: **{suggested}** — {reason}")

    return target


def recommendation_controls(recommendations):
    """
    Build recommendation filtering controls.
    """

    with st.sidebar:
        st.divider()

        st.header("Recommendations")

        available_chart_types = sorted(
            {candidate["chart"] for candidate in recommendations}
        )

        selected_chart_types = st.multiselect(
            "Chart types",
            options=available_chart_types,
            default=available_chart_types,
            format_func=chart_display_name,
        )

        minimum_score = st.slider(
            "Minimum score",
            min_value=0,
            max_value=100,
            value=35,
            help=("Hide recommendations below this score."),
        )

        maximum_display = min(15, len(recommendations))

        if maximum_display > 0:
            top_n = st.slider(
                "Maximum recommendations",
                min_value=1,
                max_value=maximum_display,
                value=min(5, maximum_display),
            )

        else:
            top_n = 0

    return (selected_chart_types, minimum_score, top_n)


# ==================================================
# RECOMMENDATIONS TAB
# ==================================================


def display_paradoxes(df, paradoxes):
    """
    "Worth a closer look": relationships that reverse (or only appear)
    within groups. Not scored on the 0-100 scale.
    """

    if not paradoxes:
        return

    st.subheader("Worth a closer look")

    st.caption(
        "These relationships change direction once you split the data into "
        "groups (Simpson's paradox), so the overall trend can mislead. Listed "
        "separately; they do not affect the scores below."
    )

    for paradox in paradoxes:
        with st.container(border=True):
            label = (
                "Reversal" if paradox["kind"] == "reversal" else "Hidden within groups"
            )

            st.markdown(
                f"### {humanize_column_name(paradox['x'])} vs "
                f"{humanize_column_name(paradox['y'])} by "
                f"{humanize_column_name(paradox['group'])}"
            )

            st.markdown(f":orange-badge[Simpson's paradox] :gray-badge[{label}]")

            st.info(paradox["sentence"], icon="🔀")

            grouped_chart = {
                "chart": "scatter",
                "x": paradox["x"],
                "y": paradox["y"],
                "group": paradox["group"],
            }

            fig = render_candidate(df, grouped_chart)
            # The card heading above already names the chart (as on the board).
            fig.update_layout(title_text="", margin={"t": 40})

            st.plotly_chart(
                fig,
                width="stretch",
                config=plotly_config(
                    f"{paradox['x']} vs {paradox['y']} by {paradox['group']}"
                ),
            )

            pin_button(
                {"kind": "paradox", "chart": grouped_chart, "paradox": dict(paradox)},
                "paradox",
            )

            with st.expander("Python code"):
                st.code(
                    chart_code(grouped_chart, export_csv_name(), df), language="python"
                )

            with st.expander("Group-level evidence"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                humanize_column_name(paradox["group"]): g["group"],
                                "Rows": g["n"],
                                "Spearman ρ": round(g["rho"], 3),
                            }
                            for g in paradox["groups"]
                        ]
                        + [
                            {
                                humanize_column_name(paradox["group"]): "All rows",
                                "Rows": paradox["n"],
                                "Spearman ρ": round(paradox["overall"], 3),
                            }
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    f"Pooled within-group ρ = {paradox['within']:+.3f} "
                    f"(p = {paradox['within_p']:.1e}); overall ρ = "
                    f"{paradox['overall']:+.3f} (p = {paradox['overall_p']:.1e}). "
                    "Both survive a Bonferroni correction over every pair and "
                    "grouping checked."
                    if paradox["kind"] == "reversal"
                    else f"Pooled within-group ρ = {paradox['within']:+.3f} "
                    f"(p = {paradox['within_p']:.1e}); overall ρ = "
                    f"{paradox['overall']:+.3f}."
                )

    st.divider()


def display_board_tab(df, source_name, stats_line, target, notes):
    """
    'My board': pinned charts side by side, with an HTML download.
    """

    dataset_key = st.session_state.get("export_csv_name", "")
    pins = board_pins(dataset_key)

    st.subheader("My board")

    if not pins:
        st.info(
            f"Nothing pinned yet. Use **📌 Pin to board** under any chart "
            f"(recommendations or Worth a closer look) to collect up to "
            f"{MAX_PINS} charts here and compare them side by side."
        )

        return

    st.caption(
        f"{len(pins)} of {MAX_PINS} pins · pins stay while you change the outcome "
        "or filters, and reset when you switch datasets."
    )

    columns = st.columns(2)

    for index, pin in enumerate(pins):
        chart = pin["chart"]

        with columns[index % 2], st.container(border=True):
            if pin["kind"] == "paradox":
                title = f"{chart['x']} vs {chart['y']} by {chart['group']}"
                st.markdown(f"**{title}**  \n:orange-badge[Simpson's paradox]")
            else:
                title = visualization_title(chart)
                st.markdown(f"**{title}** · score {chart['score']:.1f}")

            fig = render_candidate(df, chart)

            # The card already shows the title; in a half-width column the
            # figure's own title would collide with its legend.
            fig.update_layout(title_text="", margin={"t": 40})

            st.plotly_chart(
                fig,
                width="stretch",
                config=plotly_config(title),
                key=f"board-chart::{chart_key(chart)}",
            )

            if pin["kind"] == "paradox":
                st.caption(pin["paradox"]["sentence"])
            else:
                st.caption(score_breakdown(chart)["formula"])

            st.button(
                "Unpin",
                key=f"board-unpin::{chart_key(chart)}",
                on_click=toggle_pin,
                args=(dataset_key, pin),
            )

    def build_board_report():
        return html_report(
            df,
            source_name=f"{source_name} (my board)",
            stats_line=stats_line,
            target=target,
            paradoxes=[pin["paradox"] for pin in pins if pin["kind"] == "paradox"],
            recommendations=[
                pin["chart"] for pin in pins if pin["kind"] == "recommendation"
            ],
            notes=notes,
            insight=plain_english_insight,
        )

    st.download_button(
        "Download board as HTML",
        data=build_board_report,
        file_name=f"visift-board-{dataset_key.rsplit('.', 1)[0]}.html",
        mime="text/html",
    )


def display_recommendations_tab(
    df,
    candidates,
    recommendations,
    selected_chart_types,
    minimum_score,
    top_n,
    target=None,
    paradoxes=(),
):
    """
    Display globally ranked recommendations.

    Simpson's paradox findings come first ("Worth a closer look"); with a
    target, its drivers are shown next in their own section.
    """

    display_paradoxes(df, list(paradoxes))

    if target is not None:
        drivers, recommendations = split_by_target(recommendations, target)

        st.subheader(f"Drivers of {humanize_column_name(target)}")

        st.write(
            f"Charts that involve **{target}**, ranked by the same score as "
            f"everything else."
        )

        display_recommendation_list(
            df, drivers, selected_chart_types, minimum_score, top_n
        )

        st.divider()

        st.subheader("Other Recommendations")

        display_recommendation_list(
            df, recommendations, selected_chart_types, minimum_score, top_n
        )

        return

    st.subheader("Recommended Visualizations")

    st.caption(
        f"{len(candidates)} valid chart candidates, ranked by chart "
        "suitability and pattern strength."
    )

    display_recommendation_list(
        df, recommendations, selected_chart_types, minimum_score, top_n
    )


def filter_recommendations(recommendations, selected_chart_types, minimum_score, top_n):
    """The recommendations a list shows under the sidebar filters."""

    return [
        candidate
        for candidate in recommendations
        if (
            candidate["chart"] in selected_chart_types
            and candidate["score"] >= minimum_score
        )
    ][:top_n]


def shown_recommendations(
    recommendations, target, selected_chart_types, minimum_score, top_n
):
    """Everything the Recommendations tab shows, in order (drivers first)."""

    if target is None:
        return filter_recommendations(
            recommendations, selected_chart_types, minimum_score, top_n
        )

    drivers, others = split_by_target(recommendations, target)

    return filter_recommendations(
        drivers, selected_chart_types, minimum_score, top_n
    ) + filter_recommendations(others, selected_chart_types, minimum_score, top_n)


def display_recommendation_list(
    df, recommendations, selected_chart_types, minimum_score, top_n
):
    """
    Filter and render one ranked list of recommendations.
    """

    filtered = filter_recommendations(
        recommendations, selected_chart_types, minimum_score, top_n
    )

    if not filtered:
        st.warning("No recommendations match the current filters.")

        return

    st.caption(
        f"Showing {len(filtered)} recommendation{'' if len(filtered) == 1 else 's'}."
    )

    for rank, candidate in enumerate(filtered, start=1):
        display_recommendation(df, candidate, rank)


# ==================================================
# DATASET TAB
# ==================================================


def display_notes_summary(notes):
    """
    One line near the top when some relationships were set aside.
    """

    if not notes:
        return

    count = len(notes)

    st.caption(
        f"ℹ️ **Dataset notes:** {notes[0]['reason']}"
        + (f" (+{count - 1} more)" if count > 1 else "")
        + " See the Dataset tab."
    )


def display_dataset_notes(notes, noted_charts):
    """
    Redundant or trivially related columns, with their statistics, and the
    charts kept out of the ranking because of them.
    """

    st.markdown("### Dataset notes")

    if not notes:
        st.caption("No near-duplicate or near-deterministic columns found.")

        return

    st.write(
        "These columns are related by construction rather than by anything "
        "worth discovering. Visift treats near-duplicates as one variable and "
        "does not rank charts between related columns; their scores are "
        "unchanged."
    )

    for note in notes:
        st.markdown(
            f"- {note['reason']} "
            f"<span style='opacity:0.7'>({note['statistic']} = "
            f"{note['value']:.3f})</span>",
            unsafe_allow_html=True,
        )

    if noted_charts:
        st.caption("Charts set aside (not ranked):")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Chart": visualization_title(chart),
                        "Type": chart_display_name(chart["chart"]),
                        "Score": round(float(chart["score"]), 1),
                    }
                    for chart in noted_charts
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def display_dataset_tab(df, profile, source_name, notes=(), noted_charts=()):
    """
    Display dataset metadata and preview.
    """

    st.subheader("Dataset")

    st.caption(f"Source: {source_name}")

    display_dataset_notes(list(notes), list(noted_charts))

    st.markdown("### Inferred column types")

    st.write(
        "The engine combines pandas data types, "
        "column names, cardinality, and observed "
        "values to infer how each column should "
        "be interpreted."
    )

    st.dataframe(build_column_profile_table(profile), width="stretch", hide_index=True)

    st.markdown("### Data preview")

    preview_rows = min(100, len(df))

    st.caption(f"Showing the first {preview_rows} rows.")

    preview = build_display_preview(df, profile, rows=preview_rows)

    st.dataframe(preview, width="stretch", hide_index=True)


# ==================================================
# HOW IT WORKS
# ==================================================


def method_card(number, title, description):
    """
    Display one compact methodology card.
    """

    with st.container(border=True):
        st.markdown(f"### {number}. {title}")

        st.write(description)


def display_method_tab():
    """
    Explain the recommendation pipeline.
    """

    st.subheader("How Visift Works")

    st.write(
        "The engine separates whether a chart "
        "**can be made well** from whether it is "
        "**likely to reveal something useful**."
    )

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        method_card(
            1,
            "Profile",
            ("Analyze types, missing values, cardinality, and distributions."),
        )

    with row1_col2:
        method_card(
            2,
            "Generate",
            (
                "Create semantically valid bar, "
                "scatter, line, histogram, and "
                "box-plot candidates."
            ),
        )

    with row1_col3:
        method_card(
            3,
            "Evaluate",
            ("Measure semantic fit, readability, data quality, and sample support."),
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        method_card(
            4,
            "Measure patterns",
            (
                "Use chart-specific statistics to "
                "estimate how much useful structure "
                "the chart may reveal."
            ),
        )

    with row2_col2:
        method_card(
            5,
            "Rank & diversify",
            (
                "Combine suitability and pattern "
                "strength, rank candidates, and "
                "remove redundant views."
            ),
        )

    st.markdown("### Recommendation score")

    st.write(
        "A high score requires both a chart that "
        "is appropriate for the data and evidence "
        "of an informative pattern. A visualization "
        "cannot rank highly simply because it is "
        "technically valid."
    )

    score1, score2, score3, score4 = st.columns(4)

    score1.metric("Excellent", "80–100")

    score2.metric("Strong", "65–79")

    score3.metric("Moderate", "50–64")

    score4.metric("Weak", "35–49")

    st.markdown("### Feedback")

    st.write(
        "The 👍 / 👎 under each recommendation (and an optional reason after a "
        "👎) is saved to help evaluate and tune the ranking later. Each vote "
        "stores the chart type, the columns it uses, its rank, score and score "
        "components, the selected outcome, and a random session id. No data "
        "values are stored. For an uploaded file, the file name is never stored "
        "and column names are replaced by a token salted per session, so they "
        "can't be read back. On the hosted demo the "
        "votes live in a file on the server, which can be reset when the app is "
        "redeployed."
    )


# ==================================================
# MAIN APPLICATION
# ==================================================


def main():
    """
    Run the Visift frontend.
    """
    apply_custom_theme()
    display_header()

    df, source_name, default_target = load_data()

    st.session_state["export_csv_name"] = (
        Path(SAMPLES[source_name][0]).name if source_name in SAMPLES else source_name
    )

    # -----------------------------------
    # Dataset validation
    # -----------------------------------

    if df.empty:
        st.error("The dataset contains no rows.")

        st.stop()

    if len(df.columns) == 0:
        st.error("The dataset contains no columns.")

        st.stop()

    # -----------------------------------
    # Analysis
    # -----------------------------------

    try:
        target = target_control(
            profile_data(df), default_target=default_target, dataset_key=source_name
        )

        st.session_state["feedback_context"] = {
            "dataset": source_name,
            "uploaded": source_name not in SAMPLES,
            "target": target,
        }

        progress_bar = st.progress(0.0, text="Analyzing dataset...")

        def show_progress(stage, fraction):
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=f"{stage}...")

        (
            profile,
            candidates,
            recommendations,
            notes,
            noted_charts,
            elapsed,
            paradoxes,
        ) = analyze_dataset(df, target, progress=show_progress)

        progress_bar.empty()

    except Exception as error:
        st.error(
            "The visualization engine encountered "
            "an error while analyzing this dataset."
        )

        st.exception(error)

        st.stop()

    # -----------------------------------
    # Overview
    # -----------------------------------

    stats_line = display_dataset_metrics(
        df, profile, candidates, recommendations, elapsed, source_name
    )

    display_notes_summary(notes)

    # -----------------------------------
    # Controls
    # -----------------------------------

    (selected_chart_types, minimum_score, top_n) = recommendation_controls(
        recommendations
    )

    report_recommendations = shown_recommendations(
        recommendations, target, selected_chart_types, minimum_score, top_n
    )

    def build_report():
        # Runs only when the button is clicked.
        return html_report(
            df,
            source_name=source_name,
            stats_line=stats_line,
            target=target,
            paradoxes=paradoxes,
            recommendations=report_recommendations,
            notes=notes,
            insight=plain_english_insight,
        )

    with st.sidebar:
        st.download_button(
            "Download HTML report",
            data=build_report,
            file_name=f"visift-report-{export_csv_name().rsplit('.', 1)[0]}.html",
            mime="text/html",
            help=("A self-contained page with the charts shown here (works offline)."),
        )

    # -----------------------------------
    # Navigation
    # -----------------------------------

    # The labels must not change between runs: a changed label makes
    # Streamlit rebuild the tabs and jump back to the first one (e.g. after
    # unpinning on the board). The pin count is shown inside the board tab.
    (recommendations_tab, board_tab, dataset_tab, method_tab) = st.tabs(
        ["Recommendations", "My board", "Dataset", "How it works"], key="main_tabs"
    )

    with recommendations_tab:
        display_recommendations_tab(
            df,
            candidates,
            recommendations,
            selected_chart_types,
            minimum_score,
            top_n,
            target=target,
            paradoxes=paradoxes,
        )

    with board_tab:
        display_board_tab(df, source_name, stats_line, target, notes)

    with dataset_tab:
        display_dataset_tab(df, profile, source_name, notes, noted_charts)

    with method_tab:
        display_method_tab()


if __name__ == "__main__":
    main()
