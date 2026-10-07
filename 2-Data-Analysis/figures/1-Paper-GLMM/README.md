# 2-Data-Analysis/figures/1-Paper-GLMM: The paper (StealIQ_Paper.pdf, StealIQ_Figure_Discussion.pdf)

Per-base logistic mixed models (GLMMs) with runner, pitcher and catcher random intercepts. The success model (inputs through the release of the pitch) is the best predictor of a steal; the decision model (pre-pitch inputs only) makes the go / hold call against the break-even rate. Written by `stealiq.py decide success popchoice paperfigs` (and `glmmcheck`), code in paper.py.

**Headline.** Success GLMM: AUROC 0.791 (95% CI 0.772-0.808) on 3,145 attempts of 2026, fit on 2023-25; decision GLMM 0.629; v16 tuned 0.782 on the same attempts (DF_paper_forward_all.csv).

| Figure | Used in |
|---|---|
| `Fig_paper_anatomy.png` | paper, Figure 1; notebook |
| `Fig_paper_breakeven.png` | paper, Figure 2; notebook |
| `Fig_paper_decision_value.png` | paper, Figure 10; notebook |
| `Fig_paper_drivers.png` | paper, Figure 3; notebook; README |
| `Fig_paper_greenlight.png` | paper, Figure 11; notebook |
| `Fig_paper_ground.png` | paper, Figure 6; notebook |
| `Fig_paper_influence.png` | paper, Figure 5; notebook |
| `Fig_paper_prediction.png` | paper, Figure 4; notebook |
| `Fig_paper_runners.png` | paper, Figure 8; notebook |
| `Fig_paper_speed.png` | paper, Figure 7; notebook |
| `Fig_paper_variance.png` | paper, Figure 9; notebook |
