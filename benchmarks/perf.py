"""
Wall-clock performance of the full Visift pipeline on wide synthetic data.

    python -m benchmarks.perf                 # default cases
    python -m benchmarks.perf --columns 26 --rows 2000
    python -m benchmarks.perf --profile       # cProfile the largest case

The generated tables mix column kinds the way real CSVs do (mostly numeric,
some categories, a few flags, one date), with a handful of real
relationships so the scorers do realistic work.
"""

import argparse
import cProfile
import pstats
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from evaluation.real_world.expectations import DATASET_DIR, REAL_WORLD_DATASETS
from visift.candidate_generator import generate_candidates
from visift.pipeline import analyze
from visift.profiler import profile_dataset
from visift.redundancy import find_redundancies
from visift.scoring.engine import diversify_recommendations, score_candidate

DEFAULT_CASES = tuple(
    (columns, rows) for rows in (2_000, 10_000) for columns in (10, 26, 50)
)

FAMILIES = ("scatter", "line", "histogram", "bar", "box")


def wide_dataset(columns: int, rows: int, seed: int = 0) -> pd.DataFrame:
    """A reproducible mixed-type table with `columns` columns."""

    rng = np.random.default_rng(seed)

    n_categorical = max(1, columns // 5)
    n_boolean = max(1, columns // 10)
    n_numeric = columns - n_categorical - n_boolean - 1

    data: dict[str, object] = {
        "date": pd.date_range("2020-01-01", periods=rows, freq="h")
    }

    base = rng.normal(size=rows)

    for i in range(n_numeric):
        if i % 4 == 0:
            values = 0.6 * base + rng.normal(size=rows)  # related to base
        elif i % 4 == 1:
            values = rng.lognormal(size=rows)  # skewed
        else:
            values = rng.normal(size=rows)
        data[f"num_{i:02d}"] = values

    for i in range(n_categorical):
        levels = [f"c{i}_{k}" for k in range(3 + i % 6)]
        data[f"cat_{i:02d}"] = rng.choice(levels, rows)

    for i in range(n_boolean):
        data[f"flag_{i:02d}"] = rng.random(rows) < 0.3 + 0.1 * (i % 3)

    return pd.DataFrame(data)


def stage_timings(df: pd.DataFrame) -> dict[str, float]:
    """Time each pipeline stage separately (same functions analyze() uses)."""

    timings: dict[str, float] = defaultdict(float)

    start = time.perf_counter()
    profile = profile_dataset(df)
    timings["profile"] = time.perf_counter() - start

    start = time.perf_counter()
    candidates = generate_candidates(df, profile)
    timings["candidates"] = time.perf_counter() - start

    scored = []
    for candidate in candidates:
        start = time.perf_counter()
        result = score_candidate(df, candidate, profile)
        timings[f"score:{candidate['chart']}"] += time.perf_counter() - start
        if result is not None:
            scored.append(result)

    start = time.perf_counter()
    find_redundancies(df, profile)
    timings["redundancy"] = time.perf_counter() - start

    start = time.perf_counter()
    diversify_recommendations(sorted(scored, key=lambda c: c["score"], reverse=True))
    timings["diversify"] = time.perf_counter() - start

    timings["n_candidates"] = len(candidates)
    return dict(timings)


def time_case(df: pd.DataFrame, repeats: int = 1, n_jobs: int = 1) -> float:
    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        analyze(df, n_jobs=n_jobs)
        timings.append(time.perf_counter() - start)

    return min(timings)


def report(
    name: str, df: pd.DataFrame, repeats: int, stages: bool, jobs: int | None
) -> None:
    total = time_case(df, repeats)
    line = f"{name:<24} {df.shape[1]:>4} {len(df):>7} {total:>8.2f}s"

    if jobs:
        line += f" {time_case(df, repeats, n_jobs=jobs):>8.2f}s"

    if stages:
        st = stage_timings(df)
        scoring = "  ".join(
            f"{family}={st.get(f'score:{family}', 0):.2f}" for family in FAMILIES
        )
        line += (
            f"  n={int(st['n_candidates']):>5}  profile={st['profile']:.2f}"
            f"  redundancy={st['redundancy']:.2f}  {scoring}"
        )

    print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stages", action="store_true", help="per-stage timings")
    parser.add_argument(
        "--real-world", action="store_true", help="also time downloaded datasets"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="also time with this many worker processes (pool warmed first)",
    )
    args = parser.parse_args()

    cases = (
        ((args.columns, args.rows),) if args.columns and args.rows else DEFAULT_CASES
    )

    if args.profile:
        columns, rows = cases[-1]
        df = wide_dataset(columns, rows)
        profiler = cProfile.Profile()
        profiler.enable()
        analyze(df)
        profiler.disable()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(35)
        return

    if args.jobs:
        # Pay the worker start-up once, outside the timings.
        analyze(wide_dataset(26, 2_000), n_jobs=args.jobs)

    header = f"{'dataset':<24} {'cols':>4} {'rows':>7} {'serial':>9}"
    if args.jobs:
        header += f" {f'{args.jobs} jobs':>9}"
    print(header)

    for columns, rows in cases:
        report(
            "synthetic",
            wide_dataset(columns, rows),
            args.repeats,
            args.stages,
            args.jobs,
        )

    if args.real_world:
        for spec in REAL_WORLD_DATASETS:
            path = DATASET_DIR / spec.filename
            if path.exists():
                report(
                    spec.name, pd.read_csv(path), args.repeats, args.stages, args.jobs
                )


if __name__ == "__main__":
    main()
