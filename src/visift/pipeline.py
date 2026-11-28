"""
The end-to-end Visift pipeline in one call.

    analysis = analyze(df, target=None)

1. Profile columns and infer semantic types.
2. Generate semantically valid chart candidates (plus target-oriented
   ones when a target is given).
3. Score every candidate with its chart-specific scorer.
4. Find redundant / trivially related columns (visift.redundancy): pairs,
   and columns determined by two others together. Charts *between* such
   columns are moved to the dataset notes; they are kept, with their
   scores, but not ranked.
5. Diversify: charts on the same variables collapse into one
   recommendation with alternatives, treating each near-duplicate cluster
   as a single variable.
6. With a target: drivers of the target first, then everything else.
"""

from collections.abc import Callable
from typing import TypedDict

import pandas as pd

from visift.candidate_generator import generate_candidates
from visift.paradox import Paradox, find_paradoxes
from visift.profiler import profile_dataset
from visift.redundancy import (
    NEAR_DUPLICATE_KINDS,
    Redundancy,
    duplicate_aliases,
    find_redundancies,
    flagged_pairs,
)
from visift.scoring.engine import diversify_recommendations, score_all_candidates
from visift.target import split_by_target
from visift.types import Candidate, DatasetProfile, Recommendation, ScoredCandidate


class DatasetNote(Redundancy):
    involves_target: bool


class Analysis(TypedDict):
    profile: DatasetProfile
    candidates: list[Candidate]
    # Every scored candidate, including noted charts, in score order.
    scored: list[ScoredCandidate]
    # Final ranking: drivers then others with a target, else all ranked.
    recommendations: list[Recommendation]
    drivers: list[Recommendation]
    others: list[Recommendation]
    notes: list[DatasetNote]
    # Scored charts between redundant pairs, kept out of the ranking.
    noted_charts: list[ScoredCandidate]
    # Near-duplicate column -> cluster representative.
    aliases: dict[str, str]
    # Simpson's paradox findings, largest gap first ("Worth a closer look").
    paradoxes: list[Paradox]


def _attach_paradoxes(
    recommendations: list[Recommendation], paradoxes: list[Paradox]
) -> None:
    """
    Badge plain x-y scatters whose pair reverses within groups, and add the
    grouped scatter as an alternative view. Scores and order are untouched.
    """

    by_pair: dict[frozenset[str], Paradox] = {}

    for paradox in paradoxes:  # largest gap first, so keep the first
        by_pair.setdefault(frozenset((paradox["x"], paradox["y"])), paradox)

    for recommendation in recommendations:
        y = recommendation.get("y")

        if recommendation["chart"] != "scatter" or y is None:
            continue

        match = by_pair.get(frozenset((recommendation["x"], y)))

        if match is None:
            continue

        recommendation["paradox"] = dict(match)
        recommendation["alternative_charts"].append(
            {
                "chart": "scatter",
                "chart_name": f"Scatter by {match['group']}",
                "title": f"{recommendation['x']} vs {y} by {match['group']}",
                "x": recommendation["x"],
                "y": y,
                "group": match["group"],
            }
        )


def _chart_pair(candidate: Candidate) -> frozenset[str] | None:
    y = candidate.get("y")
    return None if y is None else frozenset((candidate["x"], y))


def _note(redundancy: Redundancy, target: str | None) -> DatasetNote:
    involves_target = target is not None and target in redundancy["columns"]
    reason = redundancy["reason"]

    if involves_target and redundancy["kind"] in NEAR_DUPLICATE_KINDS:
        (twin,) = [c for c in redundancy["columns"] if c != target]
        reason = (
            f"Your outcome {target} has a near-duplicate, {twin}: {reason} "
            f"Charts of {twin} vs {target} are not ranked as drivers."
        )

    return {**redundancy, "reason": reason, "involves_target": involves_target}


def analyze(
    df: pd.DataFrame,
    target: str | None = None,
    profile: DatasetProfile | None = None,
    n_jobs: int | None = 1,
    progress: Callable[[str, float], None] | None = None,
) -> Analysis:
    """
    Run the full pipeline. Pass profile to fix semantic types instead of
    inferring them (the calibration suites do this).

    n_jobs: worker processes for scoring (1 = serial, the default; None =
    automatic). The result does not depend on it. Serial is the default
    because starting worker processes costs several seconds the first time
    (each imports pandas and scikit-learn), which is slower than scoring a
    typical upload serially; parallelism pays off for repeated or very large
    analyses in a long-running process.

    progress(stage, fraction) is called as the pipeline advances (for a
    progress bar); fraction goes from 0 to 1 over the whole analysis.
    """

    def report(stage: str, fraction: float) -> None:
        if progress is not None:
            progress(stage, fraction)

    report("Profiling columns", 0.0)

    if profile is None:
        profile = profile_dataset(df)

    report("Generating chart candidates", 0.05)
    candidates = generate_candidates(df, profile, target=target)

    # Scoring dominates the runtime: map it onto 10%-90% of the bar.
    def scoring_progress(done: int, total: int) -> None:
        report(f"Scoring {done}/{total} chart candidates", 0.1 + 0.8 * done / total)

    scored = score_all_candidates(
        df, candidates, profile, n_jobs=n_jobs, progress=scoring_progress
    )

    report("Checking for redundant columns", 0.9)
    redundancies = find_redundancies(df, profile)
    pairs = flagged_pairs(redundancies)

    noted_charts = [c for c in scored if _chart_pair(c) in pairs]
    ranked = [c for c in scored if _chart_pair(c) not in pairs]

    aliases = duplicate_aliases(df, redundancies, target)
    recommendations = diversify_recommendations(ranked, aliases)

    report("Looking for Simpson's paradoxes", 0.95)
    paradoxes = find_paradoxes(df, profile, redundancies)
    _attach_paradoxes(recommendations, paradoxes)

    if target is None:
        drivers: list[Recommendation] = []
        others = recommendations
    else:
        drivers, others = split_by_target(recommendations, target)
        recommendations = drivers + others

    report("Done", 1.0)

    return {
        "profile": profile,
        "candidates": candidates,
        "scored": scored,
        "recommendations": recommendations,
        "drivers": drivers,
        "others": others,
        "notes": [_note(r, target) for r in redundancies],
        "noted_charts": noted_charts,
        "aliases": aliases,
        "paradoxes": paradoxes,
    }
