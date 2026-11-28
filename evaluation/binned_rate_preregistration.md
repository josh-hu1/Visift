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

---

## Results (added after running; the rule above is unchanged)

Reproduce with `python -m evaluation.binned_rate_calibration` and
`python -m evaluation.real_world.benchmark --ranker visift`.

### Criterion 1, false positives: **FAILED**

Share of pure-noise binned-rate charts scoring ≥ 50 (500 seeds per cell):

| Base rate | n = 100 | n = 300 | n = 1,000 | n = 5,000 |
|---|---:|---:|---:|---:|
| 10% | 0.0% | 0.0% | 0.4% | 0.0% |
| 30% | 0.8% | **1.2%** | 0.2% | 0.0% |
| 50% | **1.4%** | **2.0%** | 0.4% | 0.0% |

None scored ≥ 65 (highest 59.8), so that half passes. Three of 12 cells
exceed the 1% limit, all at n ≤ 300 with balanced outcomes. The same
binary-rate scorer is used for categorical rate bars today, so this tail is
likely shared by them with five groups; that was not measured here.

### Criterion 2, power (reported)

Share of planted effects scoring ≥ 50 (100 seeds per cell; base rate 10%):

| Shape, lift | n = 300 | n = 1,000 | n = 5,000 |
|---|---:|---:|---:|
| U-shape, +10 pp | 2% | 7% | 0% |
| U-shape, +20 pp | 16% | 39% | 1% |
| U-shape, +30 pp | 53% | 91% | 100% |
| Linear, +10 pp | 3% | 6% | 0% |
| Linear, +20 pp | 20% | 47% | 38% |
| Linear, +30 pp | 64% | 99% | 100% |

Small effects rarely reach "Moderate" even with 5,000 rows: the existing
rate-gap scale (60 pp for most of the absolute signal) treats a 10-point gap
as weak.

### Criterion 3, real-world benchmark: passed

| | Before (main) | With binned-rate bars |
|---|---:|---:|
| No outcome: R@3 / R@5 / R@10, MRR | 15% / 30% / 50%, 0.205 | 15% / 30% / 50%, 0.205 |
| Outcome: R@3 / R@5 / R@10 | 50% / 70% / 95% | 50% / 70% / **100%** |
| Outcome: MRR | 0.411 | 0.412 |

Bank Marketing age vs subscription moves from rank 11 to rank 9 with the
outcome selected, through the binned-rate view ("higher at both ends").
Titanic age vs survival stays at rank 7.

### Criterion 4, hardened synthetic benchmark: passed

Top-1/3/5 retrieval 100%, null candidates ≥ 50 0%, highest null 43.25,
all unchanged.

### Disclosed changes made while running

1. **Alternative views now keep their own columns** (`engine._alternative`).
   Before, a diversified alternative stored only its chart type, so the
   benchmark read it with the primary chart's x/y. No existing pair of
   folded charts differs in orientation, so this changed nothing before.
   The binned-rate bar (x = age) is the first view drawn the other way
   round from its box plot (x = y). Without the fix, the Bank age insight
   could not be matched at all. The pre-registration above already assumed
   alternatives count; this makes the code do that. No score changes.
2. **The effect-size baseline skips binned-rate bars**, so its committed
   numbers stay identical (verified: 45% / 75% R@10, MRR 0.190 / 0.352). It
   was defined before these charts existed, and it already ranks the same
   pair through the box plot.

### Status

Criterion 1 failed, so by the rule above this is **not proposed for
merge as is**. Options for the owner:
(a) drop it;
(b) accept a 2% noise tail at n ≤ 300 for the R@10 gain;
(c) a disclosed amendment tested on fresh seeds. An example is applying
the binary-rate scorer's own ramp with Bonferroni over the bins, or
requiring n ≥ 500.
