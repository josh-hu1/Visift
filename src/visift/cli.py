"""
Command line: visift analyze PATH [options]

    visift analyze data.csv --target churned --top 10 --html report.html
    visift analyze --sample penguins --json penguins.json
    visift --version

Files load through visift.io.load_table (encodings, delimiters, Excel, the
same friendly errors as the app). Exit codes: 0 ok, 1 the file could not be
read, 2 usage error.
"""

import argparse
import json
import shutil
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import visift
from visift.api import Recommendations, _wrap, write_report
from visift.datasets import SAMPLES, load_sample
from visift.io import UploadError, load_table
from visift.pipeline import analyze


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visift", description="Rank the charts worth looking at first in a table."
    )
    parser.add_argument(
        "--version", action="version", version=f"visift {visift.__version__}"
    )
    commands = parser.add_subparsers(dest="command", metavar="command")

    run = commands.add_parser(
        "analyze",
        help="rank charts for a CSV, TSV, TXT or Excel file",
        description="Rank the charts worth looking at first in a table.",
    )
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("path", nargs="?", help="a .csv, .tsv, .txt or .xlsx file")
    source.add_argument(
        "--sample",
        choices=sorted(SAMPLES),
        help="use a bundled sample dataset (its outcome is preselected)",
    )
    run.add_argument("--target", metavar="COL", help="outcome column to explain")
    run.add_argument(
        "--no-target",
        action="store_true",
        help="with --sample: ignore the sample's preselected outcome",
    )
    run.add_argument(
        "--top", type=int, default=10, metavar="N", help="charts to list (default 10)"
    )
    run.add_argument("--html", metavar="OUT.html", help="write the HTML report")
    run.add_argument(
        "--json",
        metavar="OUT.json",
        help="write recommendations as JSON ('-' = stdout)",
    )
    run.add_argument("--sheet", metavar="NAME", help="Excel sheet to read")
    run.add_argument(
        "--jobs", type=int, default=1, metavar="N", help="worker processes (default 1)"
    )
    run.add_argument(
        "--no-paradoxes",
        action="store_true",
        help="don't list Simpson's paradox findings",
    )
    return parser


def _load(args: argparse.Namespace) -> tuple[Any, str, list[str], str | None]:
    """(DataFrame, source name, loader notices, default target)."""

    if args.sample:
        sample = SAMPLES[args.sample]
        return load_sample(args.sample), sample.file, [], sample.outcome

    path = Path(args.path)

    try:
        data = path.read_bytes()
    except OSError as error:
        raise UploadError(
            f"Could not read {path}.", hint="Check the path.", detail=str(error)
        ) from error

    table = load_table(data, path.name, sheet=args.sheet)
    return table["df"], path.name, table["notices"], None


def _table(recs: Recommendations, out: TextIO) -> None:
    width = shutil.get_terminal_size((100, 20)).columns
    header = f"{'#':>3}  {'score':>5}  {'band':<9}  {'chart':<11}  {'x':<18}  {'y':<18}"
    print(header, file=out)
    print("-" * min(width, len(header) + 2), file=out)
    indent = " " * 5

    for r in recs:
        star = "*" if r.driver else " "
        print(
            f"{r.rank:>3}{star} {r.score:>5.1f}  {r.band:<9}  {r.chart_name:<11}  "
            f"{_cut(r.x, 18):<18}  {_cut(r.y or '', 18):<18}",
            file=out,
        )
        for line in textwrap.wrap(r.takeaway, max(40, width - len(indent))):
            print(indent + line, file=out)


def _cut(text: str, size: int) -> str:
    return text if len(text) <= size else text[: size - 1] + "…"


def _print_report(
    recs: Recommendations, source: str, notices: list[str], args: argparse.Namespace
) -> None:
    out = sys.stdout
    profile = recs.analysis["profile"]
    print(
        f"{source}: {profile['rows']:,} rows, {profile['columns']} columns, "
        f"{len(recs.analysis['candidates'])} chart candidates",
        file=out,
    )
    for notice in notices:
        print(f"note: {notice}", file=out)

    if recs.target:
        drivers = sum(r.driver for r in recs)
        print(
            f"Outcome: {recs.target} ({drivers} of the top {len(recs)} are drivers, "
            "marked *)",
            file=out,
        )
    print(file=out)

    if recs:
        _table(recs, out)
    else:
        print("No recommendations.", file=out)

    if recs.paradoxes and not args.no_paradoxes:
        print("\nWorth a closer look (Simpson's paradox):", file=out)
        for paradox in recs.paradoxes:
            print(
                f"  {paradox['x']} vs {paradox['y']} by {paradox['group']}: "
                f"{paradox['sentence']}",
                file=out,
            )

    if recs.notes:
        print("\nDataset notes (kept out of the ranking):", file=out)
        for note in recs.notes:
            print(f"  {' / '.join(note['columns'])}: {note['reason']}", file=out)


def _write_json(recs: Recommendations, destination: str) -> None:
    payload = {
        "visift_version": visift.__version__,
        "source": recs.source_name,
        "target": recs.target,
        "recommendations": [r.to_dict() for r in recs],
        "paradoxes": recs.paradoxes,
        "notes": recs.notes,
    }
    text = json.dumps(payload, indent=2, default=_json_default)

    if destination == "-":
        print(text)
    else:
        Path(destination).write_text(text + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _analyze(args: argparse.Namespace) -> int:
    if args.top < 1:
        print("visift: --top must be at least 1", file=sys.stderr)
        return 2

    try:
        df, source, notices, default_target = _load(args)
    except UploadError as error:
        print(f"visift: {error.message}", file=sys.stderr)
        if error.hint:
            print(f"hint: {error.hint}", file=sys.stderr)
        if error.sheet_names:
            print(f"sheets: {', '.join(error.sheet_names)}", file=sys.stderr)
        return 1

    target = None if args.no_target else (args.target or default_target)

    if target is not None and target not in df.columns:
        print(f"visift: no column named {target!r}", file=sys.stderr)
        print(f"columns: {', '.join(map(str, df.columns))}", file=sys.stderr)
        return 2

    analysis = analyze(df, target=target, n_jobs=args.jobs)
    recs = _wrap(analysis, df, target, args.top, source)

    if args.json == "-":
        _write_json(recs, "-")
    else:
        _print_report(recs, source, notices, args)
        if args.json:
            _write_json(recs, args.json)
            print(f"\nWrote {args.json}")

    if args.html:
        write_report(recs, args.html)
        stream = sys.stderr if args.json == "-" else sys.stdout
        print(f"Wrote {args.html}", file=stream)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:  # argparse exits 2 on usage errors, 0 on --version
        return int(exit_.code or 0)

    if args.command != "analyze":
        parser.print_help()
        return 2

    return _analyze(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
