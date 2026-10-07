# 2-Data-Analysis/results/2-v16-Logistic: v16 pooled logistic regression, the previous best

One logistic regression for both bases on sprint speed, lead at first move, ground gained, pop time, pitch class and base; the tuned version adds splines and pairwise interactions chosen by Optuna in nested cross-validation. Superseded by the success GLMM (paper, Table 2). Written by `stealiq.py v16 v16checks` (research.py).

**Headline.** v16: AUROC 0.789 cross-validated on 2023-26 (DF_v16_models.csv); 0.780 default and 0.783 tuned, fit on 2023-25 and scored on 2026 (DF_v16_forward.csv).

| File | What it is |
|---|---|
| `DF_v16_base.csv` | Base-stolen test (technical report §3.3). |
| `DF_v16_base_slopes.csv` | Separate slopes for 2nd vs 3rd (rejected). |
| `DF_v16_engine_leak_fix.csv` | One-off record: v12/v13 before vs after the score-leak fix. |
| `DF_v16_final_trials.csv` | Every trial of the final Optuna study. |
| `DF_v16_forward.csv` | v16 forward test: train 2023–25, test 2026. |
| `DF_v16_leakage_audit.csv` | When each candidate input is known (technical report §2.6). |
| `DF_v16_meta.json` | v16 run summary: final params, importances, MLE coefficients, delivery beta. |
| `DF_v16_models.csv` | v16 HEADLINE: shipped → + pitch type → + base, tuned, isotonic (technical report §3.2). |
| `DF_v16_nested_choices.csv` | Configuration chosen in each outer fold. |
| `DF_v16_pop_prior.csv` | Same- vs prior-season pop time. |
| `DF_v16_recal.csv` | Drift recalibration methods (technical report §2.5). |
| `DF_v16_v12_score_audit.csv` | One-off audit: v12 with no / end-of-PA / clean score. |
| `DF_v16_v13_score_audit.csv` | One-off audit: v13 with no / end-of-PA / clean score. |
| `DF_v16_weights.csv` | v16 input weights on one scale (technical report §3.4). |
