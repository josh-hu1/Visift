# Seasonality on irregular time steps: results

Pre-registration and script: [`evaluation/irregular_seasonality.py`](../../irregular_seasonality.py), committed before the change was implemented. Raw output: [`results.txt`](results.txt).

**Verdict: failed.** Criteria 1, 3 and 4 pass, and criterion 2 fails in 2 of its 8 cells.

| Criterion | Result |
|---|---|
| 1. Detection (jitter 0.15 / 0.30) | Pass. The median raw seasonality is 73.8 / 79.5 (it was 0 / 0), seasonality is dominant in 100% / 100% of series (it was 40% / 0%), and the median line score is 82.2 / 85.8 (it was 35.4 / 35.0). |
| 2. No new false positives | **Fail.** White noise at n = 50 is 9.0% irregular vs 7.0% evenly spaced, and a random walk at n = 100 is 37.0% vs 32.0%. The tolerance is 1 point. The other 6 cells pass. |
| 3. Gaps | Pass (0.0% vs 0.0%). |
| 4. Nothing else moves | Pass. The score dump over all bundled samples and two wide tables is byte-identical, `python app.py` output is identical, and every evenly spaced series in the evaluation scores as before. |

## Post-hoc check (not pre-registered)

The 1-point tolerance on 100 seeds per cell is tighter than sampling noise allows. So the criterion 2 and 3 cells were rerun once on 1,000 fresh seeds each (seeds 10,000 to 10,999). The verdict above does not change.

| Family | n | Irregular | Evenly spaced | Difference |
|---|---|---|---|---|
| white noise | 50 | 8.2% | 5.0% | +3.2 |
| white noise | 100 | 1.0% | 0.2% | +0.8 |
| white noise | 200 | 0.1% | 0.0% | +0.1 |
| white noise | 500 | 0.0% | 0.0% | 0.0 |
| random walk | 50 | 40.4% | 37.7% | +2.7 |
| random walk | 100 | 38.1% | 39.3% | -1.2 |
| random walk | 200 | 33.3% | 32.8% | +0.5 |
| random walk | 500 | 35.0% | 36.6% | -1.6 |
| gaps | 300 | 0.1% | 0.0% | +0.1 |

Pooled over sizes, random walks are the same: 36.7% irregular vs 36.6% evenly spaced. The n = 100 failure was noise. White noise at n = 50 shows a real excess of about 3 points. The nonzero signals there are small (median raw signal 10 of 100, before the sample-size discount the scorer already applies under 100 points).

Separately, about a third of random walks get a nonzero seasonality signal even when they are evenly spaced. That is a pre-existing weakness of the detector, and this change does not affect it.
