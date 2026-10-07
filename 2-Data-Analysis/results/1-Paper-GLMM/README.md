# 2-Data-Analysis/results/1-Paper-GLMM: The paper (StealIQ_Paper.pdf, StealIQ_Figure_Discussion.pdf)

Per-base logistic mixed models (GLMMs) with runner, pitcher and catcher random intercepts. The success model (inputs through the release of the pitch) is the best predictor of a steal; the decision model (pre-pitch inputs only) makes the go / hold call against the break-even rate. Written by `stealiq.py decide success popchoice paperfigs` (and `glmmcheck`), code in paper.py.

**Headline.** Success GLMM: AUROC 0.791 (95% CI 0.772-0.808) on 3,145 attempts of 2026, fit on 2023-25; decision GLMM 0.629; v16 tuned 0.782 on the same attempts (DF_paper_forward_all.csv).

| File | What it is |
|---|---|
| `DF_paper_anatomy.csv` | The median steal of second: lead, ground gained, sprint speed, delivery, pitch flight, pop time (paper, Fig 1). |
| `DF_paper_breakeven.csv` | Break-even success rate for steals of second and third by outs (paper, Fig 2, Table A2). |
| `DF_paper_calibration_all.csv` | 2026 calibration bins of every model (paper, Fig 4B). |
| `DF_paper_decision_calibration.csv` | 2026 calibration bins of the decision GLMM by base. |
| `DF_paper_decision_fixed.csv` | Decision GLMM fixed effects by base (paper, Fig 5, Table B2); rows with model = evaluation are the superseded decision-plus-ground-gained model. |
| `DF_paper_decision_forward.csv` | Decision GLMM forecast test by base against simpler pre-pitch models (calibration quoted in paper §4.3). |
| `DF_paper_decision_players.csv` | Every player's shrunk effect in the decision model (paper, Table 5). |
| `DF_paper_decision_value.csv` | 2026 runs per attempt by the decision model's margin over break-even, and go / hold totals (paper, Fig 10). |
| `DF_paper_decision_variance.csv` | Decision GLMM random-intercept SDs with likelihood-ratio intervals (paper, Fig 9A, Table B3). |
| `DF_paper_forward_all.csv` | THE HEADLINE: every model fit on 2023-25 and scored on the same 2026 attempts: AUROC (clustered CI, by base), log-loss, Brier, calibration, paired difference from v16 (paper, Table 2, Fig 4). |
| `DF_paper_glmm_check.csv` | GLMM recovery on 20 simulated data sets and the statsmodels cross-check (paper, Table C1). |
| `DF_paper_greenlight.csv` | 2027 green-light list: each recent runner's predicted success by outs, average and tough battery (paper, Fig 11, Table 4). |
| `DF_paper_ground.csv` | Ground needed by release to clear each break-even rate, by base, outs and catcher, and the share of attempts reaching it (paper, Fig 6). |
| `DF_paper_influence.csv` | Each input's log-odds weight per SD (indicators 0 to 1), decision and success models, by base (paper, Fig 5). |
| `DF_paper_pop_choice.csv` | Steals of third: pop time to third vs to second in the decision model, same attempts (paper, Table C3). |
| `DF_paper_re24.csv` | Runs to the end of the half-inning by season and base-out state: the run-expectancy table (paper, Appendix A). |
| `DF_paper_roc.csv` | ROC curves of every model on 2026 (paper, Fig 4A). |
| `DF_paper_runner_speed.csv` | Runner-season speed vs success: correlation with and without Josh Naylor, binomial noise, reliability, true SD, corrected r (paper, Fig 8). |
| `DF_paper_speed_2b.csv` | Sprint speed and steals of second by season: slope, p, swing, AUROC, runner-level r (paper, Fig 7). |
| `DF_paper_speed_bins.csv` | Steals of second by sprint-speed bin: stolen bases and caught stealing by season, success, runner-seasons, attempts per runner-season (paper, Table 3). |
| `DF_paper_success_cv.csv` | Choosing the success model: three specifications, five-fold CV on 2023-25 only, per base (paper, Table C2). |
| `DF_paper_success_fixed.csv` | Success GLMM fixed effects by base, 2023-26, with 10th/90th percentiles (paper, Table B1). |
| `DF_paper_success_players.csv` | Every player's shrunk effect in the success model: skill net of the jump and the pitch. |
| `DF_paper_success_variance.csv` | Success GLMM random-intercept SDs with likelihood-ratio intervals (paper, Fig 9B, Table B3). |
