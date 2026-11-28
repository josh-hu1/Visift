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

    def __init__(
        self,
        message: str,
        hint: str = "",
        detail: str = "",
        sheet_names: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.detail = detail
        # Excel only: every sheet, so the app can offer another one.
        self.sheet_names = sheet_names or []


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


def _open_workbook(data: bytes) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(io.BytesIO(data))
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


def excel_sheet_names(data: bytes) -> list[str]:
    """Sheet names of an .xlsx upload ([] if it can't be opened)."""

    try:
        return [str(name) for name in _open_workbook(data).sheet_names]
    except UploadError:
        return []


# --------------------------------------------------
# Cleaning and limits
# --------------------------------------------------


def _clean(df: pd.DataFrame, notices: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    # Drop completely empty columns first, so a trailing empty column with a
    # blank header is reported once (dropped), not also renamed.
    # (With no data rows every column is 'empty'; keep them so the error can say
    # the file has a header but no rows.)
    keep = [
        i for i in range(df.shape[1]) if len(df) == 0 or not df.iloc[:, i].isna().all()
    ]
    if len(keep) < df.shape[1]:
        notices.append(f"Dropped {df.shape[1] - len(keep)} completely empty column(s).")
        df = df.iloc[:, keep]

    headers = list(df.columns)

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

    df.columns = headers

    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        notices.append(f"Dropped {before - len(df)} completely empty row(s).")

    return df.reset_index(drop=True)


def _validate(df: pd.DataFrame, *, sheet: str | None, sheet_names: list[str]) -> None:
    """Raise UploadError unless the cleaned table has >= 2 columns and >= 1 row."""

    where = f"Sheet “{sheet}”" if sheet is not None else "The file"

    if df.shape[1] == 0:
        raise UploadError(
            f"{where} is empty.",
            hint="Pick another sheet."
            if sheet is not None
            else "Check that you exported data.",
            sheet_names=sheet_names,
        )

    if df.empty:
        raise UploadError(
            f"{where} has a header but no data rows.",
            hint="Check that the export included the data, not just column names.",
            sheet_names=sheet_names,
        )

    if df.shape[1] < 2:
        hint = (
            "Pick another sheet, or check that the data starts in the first row."
            if sheet is not None
            else "Visift compares columns, so it needs at least two. If the file has "
            "more, its separator may be unusual: use commas, semicolons, tabs or "
            "pipes."
        )
        raise UploadError(
            "Only one column was found.",
            hint=hint,
            detail=f"Columns read: {list(df.columns)}",
            sheet_names=sheet_names,
        )


def _load_excel(
    data: bytes, sheet: str | None
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    """
    (clean df, notices, sheet names, chosen sheet). With no sheet requested,
    opens the first usable sheet; a requested sheet is used as-is.
    """

    workbook = _open_workbook(data)
    names = [str(name) for name in workbook.sheet_names]

    def read(name: str) -> tuple[pd.DataFrame, list[str]]:
        notices: list[str] = []
        df = _clean(pd.read_excel(workbook, sheet_name=name), notices)
        _validate(df, sheet=name, sheet_names=names)
        return df, notices

    if sheet is not None and sheet in names:
        df, notices = read(sheet)
        return df, notices, names, sheet

    first_error: UploadError | None = None

    for index, name in enumerate(names):
        try:
            df, notices = read(name)
        except UploadError as error:
            first_error = first_error or error
            continue

        if index > 0:
            notices.insert(0, f"Opened sheet “{name}”; earlier sheets have no table.")
        return df, notices, names, name

    assert first_error is not None  # a workbook always has at least one sheet
    raise first_error


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
        df, notices, sheet_names, chosen = _load_excel(data, sheet)
    elif name.endswith(TEXT_SUFFIXES) or "." not in name:
        df, notices = _read_text(data)
        df = _clean(df, notices)
        _validate(df, sheet=None, sheet_names=[])
    else:
        raise UploadError(
            "Unsupported file type.",
            hint="Upload a .csv, .tsv, .txt or .xlsx file.",
            detail=f"File name: {filename}",
        )

    if len(df) > MAX_ROWS:
        notices.append(
            f"The file has {len(df):,} rows; Visift analyzes a random sample of "
            f"{MAX_ROWS:,} (fixed seed, so results are repeatable)."
        )
        df = df.sample(n=MAX_ROWS, random_state=SAMPLE_SEED).sort_index()
        df = df.reset_index(drop=True)

    return {"df": df, "notices": notices, "sheet_names": sheet_names, "sheet": chosen}
