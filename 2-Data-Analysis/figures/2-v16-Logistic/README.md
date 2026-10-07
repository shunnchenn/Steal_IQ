# 2-Data-Analysis/figures/2-v16-Logistic: v16 pooled logistic regression, the previous best

One logistic regression for both bases on sprint speed, lead at first move, ground gained, pop time, pitch class and base; the tuned version adds splines and pairwise interactions chosen by Optuna in nested cross-validation. Superseded by the success GLMM (paper, Table 2). Written by `stealiq.py v16 v16checks` (research.py).

**Headline.** v16: AUROC 0.789 cross-validated on 2023-26 (DF_v16_models.csv); 0.780 default and 0.783 tuned, fit on 2023-25 and scored on 2026 (DF_v16_forward.csv).

| Figure | Used in |
|---|---|
| `Fig_v16_tuning.png` | notebook; technical report (archived) |
| `Fig_v16_weights.png` | notebook; technical report (archived) |
