# EVAL-FINDINGS

## CLAIM_UNDER_TEST
Stated before the first held-out run (see `scripts/eval.py`; thresholds never edited afterwards):

- **A.** On real CC-BY forest-plot figures with a printed `est [lo, hi]` column, the pixel-geometry reading agrees with the printed text within 5% on est, lo AND hi for **at least 60% of rows** (rows from figures the pipeline could calibrate; unsupported-figure coverage reported separately).
- **B.** When the printed text of a row that agreed is corrupted by a realistic error of >=10% (decimal slip, +/-10% estimate, upper bound +25%, swapped bounds), the cross-check flags **at least 90%** of such corruptions.

Data: 161 figures from CC-BY PMC articles whose caption says "forest plot" (Europe PMC search + PMC S3 open-data bucket). The first 15 were the dev set used to debug; the other 146 are the held-out set used for every number below.

## Runs, in order (all kept in `eval_results.json`)

| run | change | figures calibratable | rows agreeing / rows | **A** | B recall (large corruptions) | clean-row flag rate |
|---|---|---|---|---|---|---|
| run1_initial | pipeline as developed on the dev set | 86/146 | 514 / 960 | **53.5%** (FAIL) | 99.6% | 46.5% |
| run2_per_panel_axes | fix attempt 1 | 90/146 | 595 / 1044 | **57.0%** (FAIL) | 99.7% | 43.0% |
| run3_vertical_line_reject | fix attempt 2 | 90/146 | 600 / 1044 | **57.5%** (FAIL) | 99.7% | 42.5% |

Claim B passes in every run (>=90%); claim A fails in every run (<60%).

### Fix attempt 1
- **Diagnosis:** composite figures have several stacked panels (A/B/C), each with its own x-axis and x-offset, but the pipeline fitted one axis (from the bottom tick row) and applied it to every row. Panels other than the bottom one were measured against the wrong scale.
- **Change:** fit one axis per tick row; assign each printed row to the nearest tick row below it.
- **Effect:** A 53.5% -> 57.0% (+4 calibratable figures, +84 rows). Not enough.

### Fix attempt 2
- **Diagnosis:** the vertical null/reference line is 3-4 px wide at the figures' scale, so it survives the 4x4 morphological opening used to find markers. On rows whose real marker is tiny, it was picked as "the marker" and the estimate landed on the null line (visible in the overlay for stacked R `meta` plots).
- **Change:** reject opened components that span the whole row band and are thin.
- **Effect:** A 57.0% -> 57.5% (+5 rows). Essentially nothing; the hypothesis explains only a handful of rows.
- Stopped here per the two-attempt rule.

## What the remaining failures look like (run 3, 444 flagged rows of 1044)
Not fixed, reported as measured: ~130 rows where both estimate and CI are wrong (row/marker mismatch), ~89 where the estimate is right but the CI is too wide (an extra horizontal line merged into the whisker), ~88 where the CI is too narrow (a large weighted square hides the whisker, or a light/thin whisker is missed), ~85 where the CI width is right but the estimate is shifted, ~57 with no geometry at all. A further contributor is figures whose own tick labels are not evenly spaced on the stated log scale (e.g. `PMC13364986_fimmu-17-1726732-g003`): there the flag is arguably a genuine figure inconsistency, not a pipeline error, and the tool cannot tell the two apart.

## Caveats that bound how far B can be trusted
- **B is partly circular.** The corruption is applied to the printed text of rows where geometry already agreed within 5%, so a >=10% edit is flagged almost by construction. B shows the cross-check has no blind spot for large typos *on rows it can read*; it does not show the tool finds errors in real papers.
- **The cost is the false-alarm rate.** 42.5% of unmodified rows are flagged. If 10% of printed rows carried a large error, precision would be ~21% (computed in `eval_results.json`). Some flags are real figure inconsistencies, but there is no independent truth to say how many.
- **Coverage:** only 90/146 (62%) held-out figures could be calibrated (29 had no 3+ consistent numeric ticks, 27 had no parseable `est [lo, hi]` column). Ratio-scale rows only.
- **Cut from the validated scope:** claim 3 (beat a VLM-only baseline). No VLM was run, so nothing is claimed about it.

## VERDICT: PARTIAL
Claim B holds (99.7% recall on large corruptions, on rows the tool can read). Claim A fails (57.5% vs the pre-set 60%) and the false-alarm rate is high (42.5%), so the tool is a *screening aid with many false flags*, not a reliable digitiser. It works well on clean single-panel plots (20/90 figures have >=88% of rows agreeing, 26 more 63-87%; 12 figures have 0-12%) and badly on dense composites, tiny markers and rule-heavy layouts.
