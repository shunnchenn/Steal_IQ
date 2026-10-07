# 5-Live-Webapp/results: The live site: calculator, leaderboard and season metrics

What the public site shows: the steal-odds calculator, the season metrics (Steal+, Burst) and the payload written into docs/index.html. Written by `python3 5-Live-Webapp/webapp.py`.

**Headline.** Calculator: the success GLMM's fixed part, AUROC 0.789 on the 2026 forward test (DF_calculator.csv).

| File | What it is |
|---|---|
| `DF_calculator.csv` | THE CALCULATOR: the success GLMM's fixed part per base, term by term as the page runs it, and its 2026 forward-test AUROC. |
| `DF_perattempt_AUC.csv` | Per-attempt XGBoost AUROC ladder. |
| `DF_perattempt_Importance.csv` | Per-attempt XGBoost feature importance. |
| `DF_success_model.csv` | v15 four-input logistic on 2023-26 (intercept, 4 coefficients, CV AUROC): its lead weights define Ground and Burst. The 2015-22 calculator runs the same four inputs, fit on that era. |
| `DF_v15_leaderboard.csv` | Published leaderboard: Steal+, Burst, percentiles. |
| `DF_v15_reliability.csv` | How much of one season's Steal+ is signal vs chance. |
| `DF_v15_reliability_top.csv` | Top runner-seasons with chance bands. |
| `DF_v15_validation.csv` | Validation behind every season-metric claim. |
| `v15_players.json` | The site payload (league fits, validation, players), synced into docs/index.html. |
