"""
The end-to-end Visift pipeline in one call.

    analysis = analyze(df, target=None)

1. Profile columns and infer semantic types.
2. Generate semantically valid chart candidates (plus target-oriented
   ones when a target is given).
3. Score every candidate with its chart-specific scorer.
4. Find redundant / trivially related column pairs (visift.redundancy).
   Charts *between* such a pair are moved to the dataset notes; they are
   kept, with their scores, but not ranked.
5. Diversify: charts on the same variables collapse into one
   recommendation with alternatives, treating each near-duplicate cluster
   as a single variable.
6. With a target: drivers of the target first, then everything else.
"""

from typing import TypedDict

import pandas as pd

from visift.candidate_generator import generate_candidates
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
    df: pd.DataFrame, target: str | None = None, profile: DatasetProfile | None = None
) -> Analysis:
    """
    Run the full pipeline. Pass profile to fix semantic types instead of
    inferring them (the calibration suites do this).
    """

    if profile is None:
        profile = profile_dataset(df)

    candidates = generate_candidates(df, profile, target=target)
    scored = score_all_candidates(df, candidates, profile)

    redundancies = find_redundancies(df, profile)
    pairs = flagged_pairs(redundancies)

    noted_charts = [c for c in scored if _chart_pair(c) in pairs]
    ranked = [c for c in scored if _chart_pair(c) not in pairs]

    aliases = duplicate_aliases(df, redundancies, target)
    recommendations = diversify_recommendations(ranked, aliases)

    if target is None:
        drivers: list[Recommendation] = []
        others = recommendations
    else:
        drivers, others = split_by_target(recommendations, target)
        recommendations = drivers + others

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
    }
