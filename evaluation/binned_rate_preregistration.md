# Pre-registration: binned-rate bars (Phase 4.2)

Written and committed **before** the rule was run on the real-world
benchmark or any calibration suite. Results, whatever they are, go in the
pull request and in `docs/design.md`.

## Problem

Monotonic statistics miss non-monotonic effects of a numeric column on a
binary outcome. In Bank Marketing, 24% of clients under 25 and 42% of
clients over 60 subscribe, but the age medians of subscribers and
non-subscribers are nearly equal, so the box plot of age by subscription
ranks 11th even with the outcome selected (README, "Where it still misses").

## Rule

**Candidate.** For every numeric column `x` (semantic type
`numeric_continuous` or `numeric_discrete`) with at least 10 distinct
non-missing values, and every boolean column `y`:
`{"chart": "bar", "x": x, "y": y, "aggregation": "binned_rate"}`.
Generated with and without a target, like the existing rate bars.

**Bins.** Quintiles of `x` (`pandas.qcut(x, 5, duplicates="drop")`) over
the rows where both columns are present. If fewer than 3 bins remain, the
chart is scored 0 (signal 0).

**Score.** No new constants. The binned column is scored by the existing
binary-rate path, exactly as a categorical predictor against a boolean
outcome: absolute and relative rate gaps, minority events per group, and the
Pearson chi-square test of independence through the existing p-value ramp.
Suitability uses the existing bar-chart functions with the bins treated as
an ordinal column (semantic fit 95), readability from the number of bins,
support from the bin sizes, and data quality from completeness of both
columns.

**Ranking.** Unchanged. The chart shares its variables with the box plot of
`x` by `y`, so diversification folds both into one recommendation and the
higher-scoring view is shown first, the other as an alternative. Nothing
else changes: no boosts, no new thresholds, and the expectations in
`evaluation/real_world/expectations.py` are not edited. The benchmark
already counts alternative views, so a recommendation whose box-plot
alternative matches an expectation still counts.

## Acceptance criteria (decided now)

1. **False positives:** on pure noise (numeric `x` independent of a binary
   `y`, base rates 10%, 30%, 50%; n = 100, 300, 1,000, 5,000; 500 seeds
   per cell), binned-rate charts score ≥ 50 in at most 1% of cases in every
   cell, and never ≥ 65.
2. **Power, reported, not gated:** the share of planted U-shaped and
   monotonic effects scoring ≥ 50, by effect size and n.
3. **Real-world benchmark:** report every number before and after, with and
   without an outcome. Ship only if target-mode R@10 and MRR do not fall.
   The Bank age insight moving into the target-mode top 10 is the intended
   effect, but it is not a criterion: missing it is reported, not tuned.
4. **Synthetic suites:** the hardened benchmark keeps its top-1/3/5
   retrieval, and its null ≥ 50 rate stays at 0%.

If a criterion fails, the rule is not adjusted to pass on the same data.
Any amendment is disclosed with the reason, and the failed version is
reported too.
