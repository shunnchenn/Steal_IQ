# 2 · Data Analysis

The models, every test behind them, and the paper. Reads the meta tables and season tables from `1-Data-Ingestion/data/`;
writes results and figures here (the live site's go to `5-Live-Webapp/`).

```
2-Data-Analysis/
├── stealiq.py               the command line: python3 stealiq.py <step> ...  (python3 stealiq.py -h lists the steps)
├── core.py                  shared: paths, output folders, loaders, and the models and features several parts use
├── paper.py                 the paper's models: RE24, break-even, per-base GLMMs, 2026 forward test, calls, 2027 list
├── research.py              the archived technical report's studies, still runnable
├── build_paper.py           → StealIQ_Paper.pdf and StealIQ_Figure_Discussion.pdf, then the catalog
├── build_catalog.py         → CATALOG.md and 0-Catalog.pdf (repository root), a README.md in every results/figures folder
├── build_notebook.py        → analysis.ipynb, the guided walkthrough of the paper (written, then executed)
├── StealIQ_Paper.ledger.md  the paper's version history: what changed in each version, what was tried and ruled out
├── literature.py            published stolen-base models refit on our data and set against ours (prints; ~3 min)
├── LITERATURE.md            the field: every published model, their weights on our data, what we are missing
├── NOTES.md                 designs not built yet (a speed-vs-success panel for the web app, with power arithmetic)
├── results/<shelf>/         tables (CSV / JSON)
└── figures/<shelf>/         figures (PNG)
```

Generated files stay on this machine: `results/` and `figures/` (except each folder's README.md), `analysis.ipynb`, the
two paper PDFs and their snapshots in `.StealIQ_Paper.versions/`. The commands below rebuild all of them.

**Shelves** (a file's name prefix picks its shelf, so a new output lands in the right place automatically; each shelf's
README.md gives its headline number and describes every file). Files without a listed prefix belong to the live site
and go to `5-Live-Webapp/results/` and `figures/`.

| Shelf | What | Prefix | Headline |
|---|---|---|---|
| `1-Paper-GLMM/` | the paper: per-base success and decision GLMMs, break-even rates, calls, 2027 list | `DF_paper_*`, `Fig_paper_*` | success GLMM AUROC 0.791 on 2026 (fit 2023–25) |
| `2-v16-Logistic/` | v16, the previous best: one pooled logistic regression, Optuna-tuned | `DF_v16_*`, `Fig_v16_*` | AUROC 0.789 cross-validated; 0.780 / 0.783 on 2026 |
| `3-Engines-v12-v13/` | v12 success and v13 decision gradient-boosting engines | `DF_v12_*`, `DF_v13_*` | |
| `4-Studies/` | Powers et al., delivery time, pre-pitch, jump speed | `DF_add_*`, `Fig_add_*` | |
| `5-Public-Baselines/` | the published models vs this project; speed by season | `DF_cmp_*`, `Fig_cmp_*` | |

`CATALOG.md` and `0-Catalog.pdf` at the repository root describe every file.

## The code, module by module

Section numbers match the archived technical report and the section banners inside the code.

| Module | § | What | Run (`stealiq.py <step>`) | Writes |
|---|---|---|---|---|
| `../5-Live-Webapp/webapp.py` | 1 | Season metrics (Steal+, Burst), the leaderboard, the odds calculator (the paper's success model, fixed part; the 2015–22 era keeps the four-input model), the site payload | `v15` (all of it), `calc` (the calculator only), or `python3 5-Live-Webapp/webapp.py` | `5-Live-Webapp/results/`, `docs/index.html` |
| `research.py` | 2 | Per-event engines: v12 success (XGBoost, 18 inputs), v13 attempt decision (617k pitches) | `v12`, `v13` | `results/3-Engines-v12-v13/` |
| `research.py` | 3 | Powers et al.: speed suppression M0–M3, mediation, mixed-effects fits; delivery time as a beta | `powers` | `DF_add_speed_suppression`, `DF_add_delivery_beta` |
| `research.py` | 4–6 | Before the pitch: delivery profile, what moves P(safe), game context, pitch type × state | `prepitch`, `needle`, `context` | `DF_add_prepitch`, `_needle`, `_context_prepitch`, `_pitchtype*` |
| `research.py` | 7 | Learner choice: logistic vs splines vs nested-tuned XGBoost / LightGBM | `algo` | `DF_add_algo` |
| `research.py` | 8–9 | v16: + pitch type + base; leakage audit, nested Optuna tuning, weights, forward test; base, drift, slopes | `v16`, `v16checks` | `results/2-v16-Logistic/` |
| `research.py` | 10 | The public standard (Powers et al., The Pitcher's Dilemma) vs this model; sprint speed vs success season by season (significant, and how small) | `baselines`, `speed` | `results/5-Public-Baselines/` |
| `research.py` | 11–12 | Runner jump speed vs pitcher time; the pitcher's earlier delivery (option b) | `jump`, `prior` | `DF_add_jump_*`, `DF_add_prior_delivery` |
| `paper.py` | 13 | **The paper:** run expectancy and break-even rates; per-base GLMMs (runner, pitcher, catcher random intercepts; Laplace maximum likelihood; likelihood-ratio intervals for the SDs): the decision model (pre-pitch inputs) with its 2026 calls in runs and the 2027 green-light list, and the success model (inputs through release; specification chosen by CV on 2023–25) in the 2026 forward test against v16 and the published specifications; the ground needed to be safe; the paper's figures; the GLMM recovery check | `decide`, `success` (`successfit` refits only), `popchoice`, `paperfigs`, `glmmcheck` | `results/1-Paper-GLMM/`, `figures/1-Paper-GLMM/` |
| `core.py` | – | Paths, shelves and loaders; the pieces several modules use: the v16 pipeline, the published models' inputs, the pitch classes, the v15 feature lists, calibration error | – | – |

```bash
python3 2-Data-Analysis/stealiq.py all          # everything, in order (decide ~2 min, success ~4 min, glmmcheck up to 1 h)
python3 2-Data-Analysis/stealiq.py baselines jump
python3 2-Data-Analysis/stealiq.py v16 60 200   # outer / final Optuna trials
python3 2-Data-Analysis/stealiq.py decide success popchoice paperfigs   # the paper's models, forward test, figures (~7 min)
python3 2-Data-Analysis/build_paper.py         # the paper and its figure discussion, then the catalog
python3 2-Data-Analysis/build_notebook.py      # the walkthrough notebook, written and executed
```

`analysis.ipynb` runs top to bottom in about a minute from the stored results (`RERUN = False`), or recomputes every step
first (`RERUN = True`). Guards inside the code fail the run rather than warn: VIF < 10 on every specification, banned
(post-outcome) columns asserted absent, identical rows asserted across compared models, and the shipped coefficients
reproduced before any study builds on them.

**Reproducibility notes.** Cross-validation folds depend on row order, so the meta tables carry `attempt_idx` and the
loaders restore it. The v16 Optuna search runs two parallel workers, so its chosen settings can differ slightly between
runs (the conclusion that tuning adds nothing beyond noise has held across re-runs); the variational-Bayes mixed-effects
fits in §3 vary in the sixth decimal. Every other output is deterministic. `results/2-v16-Logistic/DF_v16_engine_leak_fix.csv`
and the two `*_score_audit.csv` files are one-off records of the score-leak fix, kept as evidence.
