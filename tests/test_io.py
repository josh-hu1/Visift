"""Upload loading: encodings, delimiters, Excel, messy headers, limits, errors."""

import io

import numpy as np
import pandas as pd
import pytest

from visift.io import MAX_ROWS, UploadError, excel_sheet_names, load_table

ROWS = "name,city,score\nAnna,Zürich,3.5\nBjörn,Malmö,4.0\nCécile,Paris,2.5\n"


def _load(text: str, filename: str = "data.csv", encoding: str = "utf-8"):
    return load_table(text.encode(encoding), filename)


def test_utf8():
    table = _load(ROWS)
    assert list(table["df"].columns) == ["name", "city", "score"]
    assert table["df"]["city"].tolist() == ["Zürich", "Malmö", "Paris"]
    assert table["notices"] == []


def test_utf8_with_bom():
    table = load_table(b"\xef\xbb\xbf" + ROWS.encode("utf-8"), "data.csv")
    assert list(table["df"].columns) == ["name", "city", "score"]


def test_cp1252_fallback_with_notice():
    table = _load(ROWS, encoding="cp1252")
    assert table["df"]["name"].tolist() == ["Anna", "Björn", "Cécile"]
    assert any("Windows-1252" in n for n in table["notices"])


def test_semicolon_with_decimal_commas():
    text = "region;revenue;units\nNorth;1234,5;3\nSouth;99,25;4\nEast;10,0;5\n"
    table = _load(text)
    assert table["df"]["revenue"].tolist() == [1234.5, 99.25, 10.0]
    assert any("semicolon" in n for n in table["notices"])
    assert any("decimal comma" in n for n in table["notices"])


@pytest.mark.parametrize(
    ("separator", "filename", "word"),
    [("\t", "data.tsv", "tab"), ("|", "data.txt", "pipe")],
)
def test_tab_and_pipe(separator, filename, word):
    text = ROWS.replace(",", separator)
    table = _load(text, filename)
    assert list(table["df"].columns) == ["name", "city", "score"]
    assert table["df"]["score"].tolist() == [3.5, 4.0, 2.5]
    assert any(word in n for n in table["notices"])


def _xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


def test_xlsx_single_sheet():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    table = load_table(_xlsx({"Data": frame}), "book.xlsx")
    assert table["df"].equals(frame)
    assert table["sheet_names"] == ["Data"] and table["sheet"] == "Data"


def test_xlsx_two_sheets_default_first_and_pick_second():
    first = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    second = pd.DataFrame({"c": [5, 6], "d": [7, 8]})
    data = _xlsx({"One": first, "Two": second})

    default = load_table(data, "book.xlsx")
    assert default["sheet_names"] == ["One", "Two"]
    assert default["sheet"] == "One" and default["df"].equals(first)

    picked = load_table(data, "book.xlsx", sheet="Two")
    assert picked["sheet"] == "Two" and picked["df"].equals(second)


def test_duplicate_and_blank_headers_are_cleaned():
    text = " a ,a,,b,\n1,2,3,4,\n5,6,7,8,\n"
    table = _load(text)
    assert list(table["df"].columns) == ["a", "a_2", "column_3", "b"]
    notices = " ".join(table["notices"])
    assert "duplicate" in notices and "empty column" in notices


def test_empty_rows_are_dropped_with_notice():
    table = _load("a,b\n1,2\n,\n3,4\n")
    assert len(table["df"]) == 2
    assert any("empty row" in n for n in table["notices"])


@pytest.mark.parametrize(
    ("data", "filename", "message"),
    [
        (b"", "data.csv", "empty"),
        (b"   \n  ", "data.csv", "empty"),
        (b"a,b,c\n", "data.csv", "no data rows"),
        (b"only\n1\n2\n", "data.csv", "one column"),
        (
            bytes(np.random.default_rng(0).integers(0, 256, 4000, dtype=np.uint8)),
            "x.csv",
            "text table",
        ),
        (
            b"PK\x03\x04 not really a workbook",
            "book.xlsx",
            "Excel file could not be opened",
        ),
        (b"a,b\n1,2\n", "old.xls", ".xls"),
        (b"a,b\n1,2\n", "image.png", "Unsupported"),
    ],
    ids=[
        "empty",
        "blank",
        "header-only",
        "one-column",
        "random-bytes",
        "bad-xlsx",
        "xls",
        "png",
    ],
)
def test_friendly_errors(data, filename, message):
    with pytest.raises(UploadError) as raised:
        load_table(data, filename)
    assert message in raised.value.message


def test_more_than_max_rows_is_sampled():
    n = MAX_ROWS + 5_000
    text = "a,b\n" + "\n".join(f"{i},{i % 7}" for i in range(n)) + "\n"
    table = _load(text)
    assert len(table["df"]) == MAX_ROWS
    assert any("random sample" in n for n in table["notices"])
    # Fixed seed: the same rows every time.
    again = _load(text)
    assert table["df"]["a"].equals(again["df"]["a"])


def test_wide_tables_are_loaded_in_full():
    # The app offers a column picker above 50 columns; the loader keeps all.
    header = ",".join(f"c{i}" for i in range(60))
    row = ",".join(str(i) for i in range(60))
    table = _load(f"{header}\n{row}\n{row}\n")
    assert table["df"].shape == (2, 60)


# --------------------------------------------------
# Workbooks whose first sheet is not a table
# --------------------------------------------------

PENGUINS = pd.DataFrame(
    {"species": ["Adelie", "Gentoo", "Chinstrap"], "mass": [3700, 5000, 3700]}
)
COVER = pd.DataFrame({"note": ["Sales export, see next sheet"]})


def test_cover_sheet_is_skipped_with_a_notice():
    table = load_table(_xlsx({"About": COVER, "Penguins": PENGUINS}), "book.xlsx")
    assert table["sheet"] == "Penguins"
    assert table["sheet_names"] == ["About", "Penguins"]
    assert table["df"].equals(PENGUINS)
    assert (
        table["notices"][0] == "Opened sheet “Penguins”; earlier sheets have no table."
    )


def test_empty_first_sheet_is_skipped():
    table = load_table(_xlsx({"Blank": pd.DataFrame(), "Data": PENGUINS}), "book.xlsx")
    assert table["sheet"] == "Data"


def test_requesting_an_unusable_sheet_raises_with_the_sheet_list():
    data = _xlsx({"About": COVER, "Penguins": PENGUINS})
    with pytest.raises(UploadError) as raised:
        load_table(data, "book.xlsx", sheet="About")
    error = raised.value
    assert error.message == "Only one column was found."
    assert "Pick another sheet" in error.hint and "separator" not in error.hint
    assert error.sheet_names == ["About", "Penguins"]


def test_workbook_without_any_table_raises_the_first_sheets_error():
    data = _xlsx({"About": COVER, "Blank": pd.DataFrame()})
    with pytest.raises(UploadError) as raised:
        load_table(data, "book.xlsx")
    assert raised.value.message == "Only one column was found."
    assert raised.value.sheet_names == ["About", "Blank"]


def test_excel_sheet_names_helper():
    assert excel_sheet_names(_xlsx({"A": PENGUINS, "B": COVER})) == ["A", "B"]
    assert excel_sheet_names(b"not a workbook") == []


def test_trailing_empty_column_is_only_reported_as_dropped():
    table = _load("region;revenue;\nNorth;1,5;\nSouth;2,5;\n")
    assert list(table["df"].columns) == ["region", "revenue"]
    notices = " ".join(table["notices"])
    assert "Dropped 1 completely empty column" in notices
    assert "unnamed" not in notices
