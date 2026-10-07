# 5 · Live Webapp

The code behind the public site, **[shunnchenn.github.io/StealIQ](https://shunnchenn.github.io/StealIQ/)**: the leaderboard
(Steal+, Burst), the season metrics and the steal-odds calculator. `webapp.py` computes them and writes them into the
payload embedded in `docs/index.html`. The page itself stays in `docs/` at the repository root, because GitHub Pages
serves only the root or `docs/`.

```
5-Live-Webapp/
├── webapp.py      refit the calculator and leaderboard, write results/ and figures/, sync docs/index.html
├── results/       leaderboard, validation, calculator coefficients, the site payload (generated; not committed)
└── figures/       the site's figures (generated; not committed)
```

## Update the site

```bash
python3 1-Data-Ingestion/ingest.py meta     # only if the data changed: rebuild the per-season tables
python3 5-Live-Webapp/webapp.py             # refit, rewrite results/ and figures/, sync docs/index.html
```

Then commit `docs/index.html` and push; Pages republishes within a few minutes. `python3 2-Data-Analysis/stealiq.py v15`
runs the same step.

## The calculator

The 2023–26 calculator runs the best predictor in the project: the paper's per-base success GLMM (AUROC 0.791 on the
2026 attempts, against 0.782 for v16; `2-Data-Analysis/results/1-Paper-GLMM/DF_paper_forward_all.csv`). A browser cannot
know who is running, pitching or catching, so it runs the model's fixed part, every player's own effect at average. Fit
on 2023–25 and scored on the same 2026 attempts, that part reaches AUROC 0.789 (`results/DF_calculator.csv`). The page
sets each attempt's chance of being safe against the break-even rate for its base and outs.

`webapp.py` (`run_calculator`, also `python3 2-Data-Analysis/stealiq.py calc`) does it in four steps on every run:

1. It reads the specification that won the paper's cross-validation (`DF_paper_success_cv.csv`) and refits it on every
   2023–26 attempt, per base.
2. It exports the fit term by term (coefficient, input, centring) with the slider ranges, the defaults and the break-even
   rates, into `CALC_MODEL` in `docs/index.html`.
3. Before writing, it checks the formula twice on every attempt: in Python against the model's own design matrix, then
   the page's own JavaScript (`calcEta`, run in node). Either gap above 1e-9 stops the run.
4. It scores the same model fit on 2023–25 on the paper's 2026 forward-test attempts; the page quotes that number.

When the paper's model changes, rerun `stealiq.py success`, then `stealiq.py calc`: the calculator follows. A new term
in the specification needs one line in `CALC_TERMS` (how the page computes it from the inputs) and nothing else; if the
paper's best model stops being the success GLMM, the step says so. The 2015–22 era keeps the four-input v15 model, fit
on its own attempts; the same model fit on 2023–26 (`results/DF_success_model.csv`) still supplies the two lead weights
behind the leaderboard's Ground and Burst.
