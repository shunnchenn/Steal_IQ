# Draft ledger: StealIQ: When to Steal (the paper)

Tracked file: `2-Data-Analysis/StealIQ_Paper.pdf`, rendered by `2-Data-Analysis/build_paper.py` (until v003: `report.py`)
from the tables and figures `stealiq.py` (models in `paper.py`) writes to `results/1-Paper-GLMM` and `figures/1-Paper-GLMM`.
The companion `StealIQ_Figure_Discussion.pdf` is rendered by the same code and is snapshotted with each version.
Current snapshot: `v004`

## BLUF

v003 leads with the per-base success GLMM (AUROC 0.791 on 2026, fit on 2023–25, against 0.782 for the tuned v16
logistic) and keeps the pre-pitch decision GLMM for the go/hold call; every input is defined and justified before any
result, and the paper opens with Josh Naylor. v004 changes no content: the same paper, rebuilt from the reorganized code.
Stable, awaiting the user's review.

## Experiment board

### Tried

- Decision GLMM as the paper's model — v001 — Achieved a pre-pitch go/hold model as measured by 2026 AUROC 0.622 and
  calibration slope 0.93 (steals of second) by fitting per-base logistic GLMMs on pre-pitch inputs only.
- Success GLMM, specification chosen by CV — v002 — Achieved the best predictor as measured by 2026 AUROC 0.791
  (paired gain over tuned v16 +0.008, 95% CI +0.001 to +0.016) by adding ground gained, its square and pitch class and
  choosing among three specifications by five-fold log-loss on 2023–25.
- Likelihood-ratio intervals for random-effect SDs — v002 — Achieved intervals that behave at zero as measured by every
  steal-of-third SD interval now reaching zero (the Wald interval on log σ had excluded it) by profiling each SD.
- Two-point effect convention — v002 — Achieved effects that match their captions as measured by P(90th) − P(10th) in
  both the figures and the text, by replacing "P(average + range) − P(average)".

### Ruled out

- Installing R/lme4 — v001 — the user declined the install; GLMMs are fit in Python (Laplace ML, proved by simulation).
- Pooling steals of second and third with a base indicator — v001 — user: a foot of lead off second is not a foot off
  first; the plays get separate models.
- The decision GLMM as the headline predictive model — v002 — AUROC 0.63 against the user's v16 at 0.78; the paper
  must lead with the best model and report its AUROC.
- Wald intervals on log σ for the SDs — v002 — they explode for SDs near zero (3B pitcher upper bound 1.32).
- Savant's per-play run value as the bar — v001 — it is a flat +0.198 / −0.451 for every steal, not state-specific.

### Not tried

1. (done in v003) critique round:
   a. Define and justify every input before any result, with a paragraph on ground gained and a diagram of the race.
   b. Rationale for the video delivery-time measurement in Methods.
   c. Why RE24, computed from 2023–26, and a worked break-even example.
   d. The log-odds weight of every input per SD (as in the v16 weights figure), before and after the jump.
   e. Stolen-base and caught-stealing counts behind the speed figure.
   f. A runner-level view of speed: runners per speed bin, attempts per runner, each runner's success rate.
   g. A 2027 list a reader can follow (tiers, not a mixed top-and-bottom table).
   h. Say plainly that the 0.791 is scored with each 2026 attempt's measured ground gained and pitch.
   i. Cut every number and sentence that does not earn its place; tell it as a story.
2. Delivery time as a model input (measured for 70% of attempts).
3. Pickoff attempts and disengagements as inputs (Powers et al.).
4. Win-expectancy break-even rates for late, close innings.

## Version log

### v001 — 2026-10-06

**XYZ:** Achieved a front-office paper on when to steal, as measured by a 15-page PDF with eight findings and a
companion discussion, by modelling steals of second and third separately with pre-pitch GLMMs set against RE24
break-even rates.

**Follows:** the archived 27-page technical report (4-Archive/reports/StealIQ_Technical_Report_2026-10-06.pdf), which
pooled bases.

**Delta vs previous:** initial capture of the paper (snapshot `v001.pdf`, source `v001_report.py`).

**Code delta:** none recorded (snapshot only).

**Board moves:** initialized from the conversation.

**Not done:** the predictive headline (AUROC 0.62 was not reported).

### v002 — 2026-10-06

**XYZ:** Achieved a paper that leads with the best model as measured by Table 2 (success GLMM AUROC 0.791 against v16
0.783 and published specifications at most 0.618, all scored on the same 3,189 attempts of 2026) by adding the success
model, the forward test against every comparator, and the ground needed to be safe.

**Follows:** v001's structure, wording and findings, kept; the decision model moved from headline to the call.

**Delta vs previous:**
- Abstract and introduction: lead with the success model and its AUROC.
- Methods: two models per base; specification by CV on 2023–25; forecast test against v16 and published models.
- Results: new prediction and ground-needed figures, Table 2 (forward test), Table 3 (success estimates); variance figure
  shows both models; steal-of-third and runner-versus-battery claims corrected to the likelihood-ratio intervals.

**Code delta:**
- files: `stealiq.py` (success_rows, success_cv, run_success, success_fit, fig_prediction, fig_ground, sigma_ci),
  `report.py`
- previous behavior: decision and evaluation GLMMs only; Wald SD intervals.
- new behavior: success GLMM chosen by CV; forward test of eight models; likelihood-ratio SD intervals.
- unchanged: RE24, break-even, decision GLMM, green-light list.
- evidence: `stealiq.py decide successfit paperfigs` → exit 0 in 1,888 s (scratchpad rerun.log).

**Board moves:** Tried: success GLMM, LR intervals, two-point effects. Ruled out: decision GLMM as headline, Wald SDs.

**Not done:** the v003 critique list.

### v003 — 2026-10-06

**XYZ:** Achieved a paper a reader with no priors can follow, as measured by every input defined and justified in §2.2
before any result, the 2026 scoring spelled out in §3.4, and four new figures and tables answering the critique (a 21-page
paper; a 10-page discussion with one page per figure), by adding the race diagram, the RE24 and video rationales, the
log-odds weights, the speed counts, the runner-level view and a tiered 2027 list, and by tightening the text around
Josh Naylor.

**Follows:** v002's structure, voice and findings, kept; the decision-effects figure (points) became Figure 5A (log-odds
per SD), with the points carried in the text.

**Delta vs previous:**
- Introduction: opens with Josh Naylor (24.4 ft/s, slower than 97% of 579 players timed; 30 of 32 in 2025; ground gained
  in the top 1.3% of runner-seasons in 2025 and the most of any in 2026).
- §2.2 (new): the race (Figure 1) and a definition-and-reason paragraph for every input; ground gained gets its own.
- §3.1: why RE24, computed from 2023–26; a worked break-even example; why not Savant's flat run value.
- §3.4: the success model is scored with each 2026 attempt's measured ground gained and pitch type; the three metrics
  defined. §3.5 (new): why and how delivery time was measured from video.
- §4.4 (new): log-odds weight of every input per SD, before the pitch and given the jump (Figure 5).
- §4.6: Table 3 (stolen bases and caught stealing by speed bin and season, runner-seasons, attempts per runner-season)
  and Figure 8 (runner-season success vs speed; noise-corrected r).
- §4.9: Table 4 rebuilt as three calls (green light 87, battery-dependent 90, hold 11) with points above the bar.
- Steals of third: pop time to third (Table C3); observed success by state now counts every tracked attempt.
- Headline after the re-run: success GLMM AUROC 0.791 (95% CI 0.772–0.808) against v16 tuned 0.782 (paired gain +0.010,
  95% CI +0.001 to +0.018), on 3,145 attempts of 2026.

**Code delta:**
- files: `stealiq.py`, `report.py`
- previous behavior: steals of third used pop time to second; observed success by state counted only attempts with every
  model input; effects shown in probability points only; no race diagram, runner-level view or speed table.
- new behavior: dec_rows uses pop time to the base being stolen; state_rows for observed success; fig_anatomy,
  fig_influence, fig_runners, speed_bins, pop_choice; fig_speed_2b stacked; fig_effects removed (its PNG moved to the Trash).
- unchanged: RE24, break-even, the steal-of-second models (reproduced to the third decimal), the forward-test design.
- evidence: `stealiq.py decide success popchoice paperfigs` → exit 0 in 384 s; `report.py` → 21-page paper, 10-page
  discussion; `analysis.ipynb` executed, 62 cells, 0 errors.

**Board moves:**
- Tried: 1a–1i of the v003 critique list; pop time to third for steals of third (deviance 6.2 lower).
- Ruled out: the attempt-weighted line in the runner figure (two Naylor seasons drive its slope); pop time to second for
  steals of third (fits worse, Table C3).

**Not done:** items 2–4 below remain.

### v004 — 2026-10-07

**XYZ:** Achieved the v003 paper rebuilt from the reorganized code with its content unchanged, as measured by a text diff
against v003 (4 lines differ, all the reproduce command in Appendix E; 21 pages) and the discussion (10 pages, 0 lines
differ), by moving the renderer from `report.py` to `build_paper.py` and splitting `stealiq.py` into modules.

**Follows:** v003, all text, figures and numbers kept.

**Delta vs previous:**
- Appendix E: `python3 2-Data-Analysis/report.py` → `python3 2-Data-Analysis/build_paper.py`.

**Code delta:**
- files: `stealiq.py` → `core.py` (paths, loaders, shared models), `paper.py` (§13, the paper), `research.py` (§2–12),
  `5-Live-Webapp/webapp.py` (§1, the site); `stealiq.py` is now the 46-line command line that imports them.
  `report.py` → `build_paper.py` (the PDFs) and `build_catalog.py` (CATALOG.md, `0-Catalog.pdf`, folder READMEs);
  `make_notebook.py` → `build_notebook.py`. Shelves renumbered: `4-Engines-v12-v13` → `3-…`, `5-Studies` → `4-…`,
  `6-Public-Baselines` → `5-…`; the site's outputs moved from `3-v15-Site-Calculator` to `5-Live-Webapp/results|figures`.
- previous behavior: one 4,716-line `stealiq.py`; `report.py` rendered the PDFs and CATALOG.md.
- new behavior: the same outputs. `build_paper.table` gains `keep=` (long catalog tables break across pages) and keeps a
  group label with its first row; the paper is unaffected (every page's text identical before and after).
- unchanged: every model, number and figure.
- evidence: `stealiq.py decide success popchoice paperfigs speed baselines jump prior` → exit 0; 39 regenerated result files
  byte-identical to the pre-split snapshot; the site's outputs equal those of the original `stealiq.py` run in a sandbox
  (scratchpad `compare_reorg.py`). `build_paper.py` → exit 0 in 7.7 s; `build_notebook.py` → 62 cells, 0 errors.

**Board moves:** none (maintenance).

**Not done:** Not tried items 2–4 remain.
