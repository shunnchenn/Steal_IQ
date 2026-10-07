# StealIQ

## ▶ **[Try the steal-odds calculator](https://shunnchenn.github.io/StealIQ/#calc)** &nbsp;·&nbsp; [explore the leaderboard](https://shunnchenn.github.io/StealIQ/)

Josh Naylor stole 30 bases at 94% in 2025 while running slower than 97% of the players Statcast timed. **What decides a stolen base, if
not speed?** This project measures it on every tracked steal attempt of 2023–2026 and finds the answer is the ground the
runner takes *while the pitcher delivers* (the secondary lead), not the sprint. The paper models steals of second and
third separately, with mixed models that rate every runner, pitcher and catcher: a **success model** that predicts the
outcome from everything measured through the release of the pitch (the most accurate of the models tested), and a
**decision model** that uses only what is known before the pitch and sets each attempt against its break-even rate.

| Question | Answer (2023–2026, steals of second unless noted) | Source |
|---|---|---|
| How well can a steal be predicted? | Fit on 2023–25 and scored on 2026, the success model reaches **AUROC 0.791** (95% CI 0.772–0.808), ahead of v16 (0.782 tuned, 0.779 default) and the published specifications refit to the same data (0.602–0.612); sprint speed alone 0.551. From pre-pitch inputs only, the decision model reaches 0.629. | `results/1-Paper-GLMM/DF_paper_forward_all.csv` |
| What is the bar? | Break-even success **69–71%** on steals of second (teams make 78–81%); on steals of third **69%** with one out but **91%** with two. | `results/1-Paper-GLMM/DF_paper_breakeven.csv` |
| What decides a steal? | The ground gained during the delivery: success **61% → 96%** across its fifths, against **76% → 83%** across sprint speed. Speed decides *who runs*: the fastest fifth attempts 7× as often. | `figures/1-Paper-GLMM/Fig_paper_drivers.png` |
| What does it take to be safe? | With one out, about **9.4 ft** of ground by release against an average catcher and **10.7 ft** against a quick one; 68% and 45% of attempts reach those. | `results/1-Paper-GLMM/DF_paper_ground.csv` |
| Is speed significant? | Yes (3 of 4 seasons), and small: **+4.0 to +6.5 points** from slow to fast runners, AUROC 0.53–0.55. | `results/1-Paper-GLMM/DF_paper_speed_2b.csv` |
| Who controls the running game? | Before the pitch, one SD of runner skill is worth **+4.4 points**, pitcher **+3.6**, catcher **+2.5** (pitcher and catcher together about as much as the runner); pitchers act mostly through the ground they allow and the pitch they throw. On steals of third no group's differences can yet be told apart from chance (every 95% interval reaches zero). | `results/1-Paper-GLMM/DF_paper_decision_variance.csv` |
| Do the calls pay? | The decision model is calibrated on 2026 (slope **0.93**). The 18% of 2026 steals of second it would have held **lost 13 runs**; the attempts it would have sent **gained 116**. | `results/1-Paper-GLMM/DF_paper_decision_value.csv` |
| Who to send in 2027? | **177 of 188** regular base stealers clear the one-out bar against an average battery, **87** against a battery one SD tougher. | `results/1-Paper-GLMM/DF_paper_greenlight.csv` |

The full argument is the paper, `StealIQ_Paper.pdf`, with a figure-by-figure discussion, and a walkthrough notebook,
`analysis.ipynb`. All three are built locally from the code (`2-Data-Analysis/build_paper.py`, `build_notebook.py`) and
are not committed yet. **[`0-Catalog.pdf`](0-Catalog.pdf)** is the map of every folder and file.

## The repository is the research funnel

```
StealIQ/
├── 1-Data-Ingestion/            scrape → raw tables → ONE META TABLE PER SEASON
│   ├── ingest.py                  the one scraper and builder (Savant + MLB StatsAPI)
│   ├── vision/                    delivery.py: pitcher delivery time from broadcast video (+ gold labels, REPORT.md)
│   └── data/
│       ├── raw/                   committed source tables
│       ├── meta/                  meta_2023.csv … meta_2026.csv + data_dictionary.csv   ← every field, video link, QA flags
│       ├── delivery/              delivery_<season>.csv (computer-vision timing runs)
│       └── cache/                 request caches (gitignored)
├── 2-Data-Analysis/             models, tests, the paper
│   ├── stealiq.py                 the command line (core.py, paper.py, research.py hold the code)
│   ├── build_paper.py             → StealIQ_Paper.pdf + figure discussion (local), then the catalog
│   ├── build_notebook.py          → analysis.ipynb, the walkthrough (local)
│   ├── StealIQ_Paper.ledger.md    the paper's version history
│   ├── results/<shelf>/           tables, a README in each: 1-Paper-GLMM (the paper) · 2-v16-Logistic ·
│   │                              3-Engines-v12-v13 · 4-Studies · 5-Public-Baselines (generated; not committed)
│   └── figures/<shelf>/           figures, same shelves
├── 3-Presentations/             SABR Seminar prototype (the deck embeds video and stays local)
├── 4-Archive/                   superseded code, reports and outputs (local only, gitignored)
├── 5-Live-Webapp/               webapp.py builds the live site's calculator and leaderboard into docs/
├── docs/                        the live site (GitHub Pages serves this folder)
├── 0-Catalog.pdf                every folder, data file, result and figure: what it is, which code uses it
└── CATALOG.md                   the same catalog, readable on GitHub
```

Each stage reads only what the stage before it wrote. A new season, or another season of video timing, slots into the
same folders: `ingest.py` and `delivery.py` take the season as an argument and write `meta_<season>.csv` /
`delivery_<season>.csv` next to the others; the analysis picks them up. Folder READMEs:
[`1-Data-Ingestion`](1-Data-Ingestion/README.md) (sources, joins, QA flags, adding a season) ·
[`2-Data-Analysis`](2-Data-Analysis/README.md) (what each module does) ·
[`5-Live-Webapp`](5-Live-Webapp/README.md) (updating the site).

## Run it

```bash
pip install -r requirements.txt
python3 1-Data-Ingestion/ingest.py meta      # the per-season meta tables, from the committed raw tables (offline)
python3 2-Data-Analysis/stealiq.py all        # every model and test (the paper's GLMM steps take ~7 min; glmmcheck up to 1 h)
python3 2-Data-Analysis/build_paper.py       # the paper, its figure discussion, then CATALOG.md and 0-Catalog.pdf
python3 2-Data-Analysis/build_notebook.py    # the walkthrough notebook, written and executed
python3 5-Live-Webapp/webapp.py              # refit the calculator and leaderboard, sync docs/index.html
```

The site updates when `docs/` is pushed. The video pipeline has its own environment (`1-Data-Ingestion/vision/.venv`); see the ingestion README.

## What ships

- **The calculator** (`docs/`, built by `5-Live-Webapp/webapp.py`): the paper's per-base success model. P(safe) for a steal
  of second or third from sprint speed, the lead at the pitcher's first move, the ground gained by release, the catcher's
  pop time, the pitch, the count, the outs and both hands, set against the break-even rate for the base and outs. It runs
  the model's fixed part, every player's own effect at average: fit on 2023–25 and scored on the 2026 attempts every
  model can score, AUROC 0.789, against 0.791 with player effects and 0.782 for v16 tuned
  (`5-Live-Webapp/results/DF_calculator.csv`, `DF_paper_forward_all.csv`). The 2015–22 era keeps the earlier four-input
  logistic regression.
- **The leaderboard**: **Steal+**, net bases above what an average runner of the same sprint speed would produce over the
  same attempts (speed-neutral; the best single answer to *who steals well*), and **Burst**, ground gained by release
  above what the runner's speed predicts (the most repeatable technique number).

## References

- Powers, Ramani, Hahn & Schaefer (2026). *Strategies under a pickoff limit in baseball: a zero-sum sequential game with
  multilevel models.* arXiv 2601.15608 (v1 22 Jan 2026, v2 16 Aug 2026).
- 42 Analytics. *The Pitcher's Dilemma.*
- Data: Baseball Savant (Statcast search, base-stealing running game, sprint speed, pop time, video) and MLB StatsAPI.
- Computer vision: BaseballCV (github.com/dylandru/BaseballCV) detectors; RTMPose (rtmlib) keypoints.
