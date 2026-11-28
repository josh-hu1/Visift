"""
Reading user uploads robustly: encodings, delimiters, Excel, messy headers,
size limits.

load_table() never lets a raw pandas, codec or zip error reach the user. Bad
input raises UploadError with a one-line explanation, a hint for what to
try, and the technical detail for a collapsed expander. Anything the loader
had to adapt (encoding fallback, delimiter, renamed or dropped columns, row
sampling) is returned as plain-language notices.
"""

import csv
import io
import re
from typing import TypedDict

import pandas as pd

MAX_ROWS = 100_000
MAX_COLUMNS = 50
SAMPLE_SEED = 42
MAX_UPLOAD_MB = 50  # keep in sync with .streamlit/config.toml server.maxUploadSize

DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}

EXCEL_SUFFIXES = (".xlsx", ".xlsm")
TEXT_SUFFIXES = (".csv", ".tsv", ".txt")

_DECIMAL_COMMA = re.compile(r"(?:^|;)\s*-?\d+,\d+\s*(?:;|$)", re.MULTILINE)


class UploadError(ValueError):
    """A problem with an uploaded file, explained for the person uploading it."""

    def __init__(self, message: str, hint: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail


class LoadedTable(TypedDict):
    df: pd.DataFrame
    notices: list[str]
    # Excel only: every sheet name and the one that was read.
    sheet_names: list[str]
    sheet: str | None


# --------------------------------------------------
# Text files
# --------------------------------------------------


def _decode(data: bytes) -> tuple[str, list[str]]:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), ["Read as UTF-16 text."]

    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        pass

    try:
        return data.decode("cp1252"), [
            "The file is not UTF-8; it was read as Windows-1252 (Western European). "
            "Check accented characters."
        ]
    except UnicodeDecodeError:
        return data.decode("latin-1"), [
            "The file is not UTF-8; it was read as Latin-1. Check accented characters."
        ]


def _looks_binary(data: bytes) -> bool:
    head = data[:8192]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return False  # UTF-16 text legitimately contains NUL bytes
    return b"\x00" in head


def _sniff_delimiter(text: str) -> str:
    try:
        return (
            csv.Sniffer()
            .sniff(text[:20_000], delimiters="".join(DELIMITER_NAMES))
            .delimiter
        )
    except csv.Error:
        return ","


def _read_text(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    if _looks_binary(data):
        raise UploadError(
            "This doesn't look like a text table.",
            hint="Upload a CSV, TSV or TXT file, or an Excel workbook (.xlsx).",
            detail="NUL bytes found in the first 8 KB.",
        )

    text, notices = _decode(data)

    if not text.strip():
        raise UploadError("The file is empty.", hint="Check that you exported data.")

    delimiter = _sniff_delimiter(text)
    decimal = "."

    if delimiter != ",":
        notices.append(f"Columns are separated by {DELIMITER_NAMES[delimiter]}s.")

    if delimiter == ";" and _DECIMAL_COMMA.search(text[:20_000]):
        decimal = ","
        notices.append("Numbers use a decimal comma (e.g. 3,5).")

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal)
    except pd.errors.EmptyDataError as error:
        raise UploadError("The file is empty.", detail=str(error)) from error
    except pd.errors.ParserError as error:
        raise UploadError(
            "The rows don't split into the same number of columns.",
            hint="Check for stray separators or quotes in the file.",
            detail=str(error),
        ) from error

    return df, notices


# --------------------------------------------------
# Excel
# --------------------------------------------------


def _read_excel(data: bytes, sheet: str | None) -> tuple[pd.DataFrame, list[str], str]:
    try:
        workbook = pd.ExcelFile(io.BytesIO(data))
    except ImportError as error:
        raise UploadError(
            "Excel support isn't installed.",
            hint="Install it with: pip install openpyxl",
            detail=str(error),
        ) from error
    except Exception as error:  # zip / XML errors from a damaged or fake workbook
        raise UploadError(
            "The Excel file could not be opened.",
            hint="Re-save it in Excel as .xlsx, or export the sheet as CSV.",
            detail=f"{type(error).__name__}: {error}",
        ) from error

    names = [str(name) for name in workbook.sheet_names]
    chosen = sheet if sheet in names else names[0]

    return pd.read_excel(workbook, sheet_name=chosen), names, chosen


# --------------------------------------------------
# Cleaning and limits
# --------------------------------------------------


def _clean(df: pd.DataFrame, notices: list[str]) -> pd.DataFrame:
    headers = [str(column).strip() for column in df.columns]

    blank = [
        i for i, h in enumerate(headers) if not h or h.lower().startswith("unnamed:")
    ]
    for i in blank:
        headers[i] = f"column_{i + 1}"
    if blank:
        notices.append(f"Named {len(blank)} unnamed column(s) column_N.")

    seen: dict[str, int] = {}
    renamed = []
    for i, header in enumerate(headers):
        if header in seen:
            seen[header] += 1
            new = f"{header}_{seen[header]}"
            renamed.append(f"{header} → {new}")
            headers[i] = new
        else:
            seen[header] = 1
    if renamed:
        notices.append("Renamed duplicate columns: " + ", ".join(renamed) + ".")

    df = df.copy()
    df.columns = headers

    empty_columns = [c for c in df.columns if df[c].isna().all()]
    if empty_columns:
        df = df.drop(columns=empty_columns)
        notices.append(f"Dropped {len(empty_columns)} completely empty column(s).")

    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        notices.append(f"Dropped {before - len(df)} completely empty row(s).")

    return df.reset_index(drop=True)


def load_table(data: bytes, filename: str, sheet: str | None = None) -> LoadedTable:
    """
    Load an uploaded .csv/.tsv/.txt or .xlsx file into a clean DataFrame.

    More than MAX_ROWS rows are reduced to a fixed-seed random sample (with a
    notice). Width is not limited here: the app lets the user choose columns
    when there are more than MAX_COLUMNS.
    """

    name = filename.lower()

    if not data:
        raise UploadError("The file is empty.", hint="Check that you exported data.")

    sheet_names: list[str] = []
    chosen: str | None = None

    if name.endswith(".xls"):
        raise UploadError(
            "Old .xls workbooks aren't supported.",
            hint="Save the sheet as .xlsx or CSV and upload it again.",
        )

    if name.endswith(EXCEL_SUFFIXES):
        df, sheet_names, chosen = _read_excel(data, sheet)
        notices: list[str] = []
    elif name.endswith(TEXT_SUFFIXES) or "." not in name:
        df, notices = _read_text(data)
    else:
        raise UploadError(
            "Unsupported file type.",
            hint="Upload a .csv, .tsv, .txt or .xlsx file.",
            detail=f"File name: {filename}",
        )

    df = _clean(df, notices)

    if df.empty:
        raise UploadError(
            "The file has a header but no data rows.",
            hint="Check that the export included the data, not just column names.",
        )

    if df.shape[1] < 2:
        raise UploadError(
            "Only one column was found.",
            hint=(
                "Visift compares columns, so it needs at least two. If the file has "
                "more, its separator may be unusual: use commas, semicolons, tabs or "
                "pipes."
            ),
            detail=f"Columns read: {list(df.columns)}",
        )

    if len(df) > MAX_ROWS:
        notices.append(
            f"The file has {len(df):,} rows; Visift analyzes a random sample of "
            f"{MAX_ROWS:,} (fixed seed, so results are repeatable)."
        )
        df = df.sample(n=MAX_ROWS, random_state=SAMPLE_SEED).sort_index()
        df = df.reset_index(drop=True)

    return {"df": df, "notices": notices, "sheet_names": sheet_names, "sheet": chosen}
