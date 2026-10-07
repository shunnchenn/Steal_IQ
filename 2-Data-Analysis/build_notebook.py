#!/usr/bin/env python3
"""
build_notebook.py — writes analysis.ipynb, the guided walkthrough of the paper (the cells below), then executes it
so every table and figure is stored in the notebook. The notebook is generated, never edited by hand, so it cannot drift
from the paper: it reads the same result files, and its summary prints build_paper.findings, the function that writes
the paper's discussion.

Usage:  python3 build_notebook.py
"""
import subprocess, sys
import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).resolve().parent / "analysis.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# StealIQ — analysis notebook

**When should a runner steal?** This notebook walks the research behind `StealIQ_Paper.pdf` in the paper's order (Part A:
the race and its measurements, the bar, what decides a steal, how well it can be predicted, what moves the odds, what it
takes to be safe, speed, who controls the running game, the calls, the 2027 list), with the reason for every modelling
choice next to the evidence for it; any number in the paper can be found in a cell below.
Part B keeps the analyses of the earlier technical report (archived), which pooled steals of second and third.

**How it runs.** All computation lives in the modules behind `stealiq.py` (core, paper, research; the live site's in
`5-Live-Webapp/webapp.py`) and reads the per-season meta tables built by `1-Data-Ingestion/ingest.py`. With
`RERUN = False` (default) each section loads the tables `stealiq.py` already wrote and
the notebook executes in about a minute. With `RERUN = True` every step is recomputed first (hours; the paper's mixed-model
steps alone take about 45 minutes). Cells marked *live* always compute from the data.
""")
code(r"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown

import stealiq as IQ                    # 2-Data-Analysis/stealiq.py
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_colwidth", 70)
RERUN = False                           # True: recompute every step before showing it

def show(name, cols=None, query=None, n=None, nd=4):
    # a results table by name (its shelf is found from the prefix)
    p = IQ.res_path(name)
    d = pd.read_json(p) if p.suffix == ".json" else pd.read_csv(p)
    d = d.query(query) if query else d
    d = d[cols] if cols else d
    display(d.head(n).round(nd) if n else d.round(nd))
    return d

def fig(name, width=900):
    display(Image(filename=str(IQ.fig_path(name)), width=width))

print("meta tables:", sorted(p.name for p in IQ.META.glob("meta_*.csv")))
""")

# ── 1 data ──
md(r"""
## 1 · The data

One table per season (`1-Data-Ingestion/data/meta/meta_<season>.csv`): one row per tracked running event from Savant's
base-stealing drawer, joined to every Statcast field of that pitch, the MLB feed's context, season pop time and sprint
speed, the computer-vision delivery time, links to the video, and a `qa_flags` column naming any disagreement between
sources. The models read these tables through `IQ.load_attempts()` / `IQ.load_context()`, so what you QA is what the models see.

*Live:* coverage by season, and the cross-source checks.
""")
code(r"""
meta = IQ.load_meta()
sbcs = meta[meta.result.isin(["SB", "CS"])]
cov = sbcs.groupby("season").agg(attempts=("play_id", "size"), safe_rate=("y", "mean"),
                                  statcast_row=("at_bat_number", lambda s: s.notna().mean()),
                                  feed_context=("feed_is_lhp", lambda s: s.notna().mean()),
                                  cv_processed=("cv_qa", lambda s: s.notna().sum()),
                                  cv_timed=("cv_qa", lambda s: (s == "PASS").sum()),
                                  clean_rows=("qa_flags", lambda s: (s.fillna("ok") == "ok").mean()))
display(cov.round(3))
qf = sbcs.qa_flags.fillna("ok")
print("flags raised on SB/CS rows:", qf[qf != "ok"].str.split(";").explode().value_counts().to_dict())
""")
code(r"""
# one row, the way a QA reader sees it: identity and link first, then the game state, the runner, pitcher and catcher
cols = ["season", "date", "video_url", "qa_flags", "inning", "outs_when_up", "balls", "strikes", "runner_name", "base",
        "result", "lead_at_firstmove_ft", "gain_to_release_ft", "runner_sprint_speed", "pitcher_name", "p_throws",
        "cv_delivery_s", "catcher_name", "catcher_pop_2b", "batter_name", "pitch_name", "release_speed", "description"]
display(sbcs[cols].dropna(subset=["cv_delivery_s"]).head(3).T)
dd = pd.read_csv(IQ.META / "data_dictionary.csv")
print(len(dd), "columns documented in data_dictionary.csv; blocks:", dd.block.value_counts().to_dict())
""")
md(r"""
**The leakage rule.** Every input must be known before the tag. The MLB feed's count is recorded *after* the pitch and its
score at the *end* of the plate appearance, so neither is ever an input (§2.6 of the report); Statcast's pre-pitch count
and score sit beside them in the meta table.
""")
code(r"""
fig("Fig_cmp_timeline.png")
lg = show("DF_cmp_seasons.csv", cols=["season", "SB", "CS", "success_pct", "attempts_per_game", "tracked_success"], query="season >= 2021")
fig("Fig_cmp_league_rates.png")
""")

# ── A2 the race ──
md(r"""
## 2 · The race, and what is measured (paper §2.2)

A steal is a race between two clocks: the runner's lead at the pitcher's first move, the ground he gains by the release
and his sprint, against the pitcher's delivery, the pitch's flight and the catcher's pop time. The figure is drawn at the
median of every input (steals of second); ground gained is Savant's *lead distance gained*, the lead at release minus the
lead at the first move.
""")
code(r"""
if RERUN: IQ.run_decide(); IQ.run_success(); IQ.pop_choice(); IQ.run_paperfigs()
show("DF_paper_anatomy.csv")
fig("Fig_paper_anatomy.png", 760)
""")
md(r"""
## 3 · The bar each steal must clear (paper §3.1, §4.1)

A steal pays when its probability of success exceeds the break-even rate of its base-out state,
p* = [RE(now) − RE(caught)] / [RE(safe) − RE(caught)], with RE24 computed from every plate appearance of 2023–26 in
games with a tracked attempt. Savant's run value is the same for every steal (+0.2 / −0.45 runs), so it cannot give a
state-specific bar. *Live:* the RE24 table.
""")
code(r"""
RE = IQ.re24(range(2023, 2027))
display(pd.Series(RE).unstack(1).reindex(["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]).round(3))
show("DF_paper_breakeven.csv")
fig("Fig_paper_breakeven.png", 760)
""")
md(r"""
## 4 · The jump decides; speed decides who runs (paper §4.2)
""")
code(r"""
fig("Fig_paper_drivers.png", 760)
""")
md(r"""
## 5 · How well a steal can be predicted (paper §3.2–3.4, §4.3)

Two logistic GLMMs per base with runner, pitcher and catcher random intercepts: the *success model* (inputs through the
release, including ground gained and pitch type) and the *decision model* (pre-pitch inputs only). The success model's
specification is chosen by five-fold CV on 2023–25 (first table). Every model is then fit on 2023–25 and applied once to
the same 2026 attempts; the success model uses each 2026 attempt's measured ground gained and pitch type.
""")
code(r"""
show("DF_paper_success_cv.csv")
show("DF_paper_forward_all.csv", cols=["model", "n", "auroc", "auroc_lo", "auroc_hi", "auroc_2b", "auroc_3b", "log_loss",
                                       "cal_slope", "d_auroc_vs_v16_tuned", "d_lo", "d_hi"])
fig("Fig_paper_prediction.png", 760)
""")
md(r"""
## 6 · What moves the odds (paper §4.4, Appendix B–C)

Each input's weight in log-odds per standard deviation (indicators 0 to 1), before the pitch and given the jump. Steals of
third use the catcher's pop time to third, which fits them better than pop time to second (last table).
""")
code(r"""
show("DF_paper_influence.csv", cols=["model", "base", "term", "sd", "beta", "per_sd", "lo", "hi"])
fig("Fig_paper_influence.png", 760)
show("DF_paper_success_fixed.csv", cols=["base", "term", "beta", "se", "p", "p10", "p90"])
show("DF_paper_decision_fixed.csv", cols=["base", "model", "term", "beta", "se", "p", "p10", "p90"], query="model == 'decision'")
show("DF_paper_pop_choice.csv")
""")
md(r"""
## 7 · What it takes to be safe (paper §4.5)
""")
code(r"""
show("DF_paper_ground.csv")
fig("Fig_paper_ground.png", 760)
""")
md(r"""
## 8 · Speed is significant, and small (paper §4.6)

Season by season (attempts pooled in 0.5 ft/s bins), the counts behind the bins, and runner by runner: each runner-season's
success rate against his speed, with the correlation corrected for the binomial noise in a season's success rate.
""")
code(r"""
show("DF_paper_speed_2b.csv")
fig("Fig_paper_speed.png", 760)
show("DF_paper_speed_bins.csv")
show("DF_paper_runner_speed.csv")
fig("Fig_paper_runners.png", 760)
""")
md(r"""
## 9 · Who controls the running game (paper §4.7, Appendix C)

Random-intercept SDs with 95% likelihood-ratio intervals; the fitting code is checked on 20 simulated data sets (last table).
""")
code(r"""
show("DF_paper_decision_variance.csv", cols=["base", "model", "group", "levels", "sigma", "lo", "hi", "pp_plus_1sd"], query="model == 'decision'")
show("DF_paper_success_variance.csv", cols=["base", "group", "levels", "sigma", "lo", "hi", "pp_plus_1sd"])
fig("Fig_paper_variance.png", 760)
show("DF_paper_glmm_check.csv")
""")
md(r"""
## 10 · The calls, scored on 2026 (paper §4.8)
""")
code(r"""
show("DF_paper_decision_forward.csv", cols=["base", "model", "log_loss", "auroc", "auroc_lo", "auroc_hi", "cal_intercept", "cal_slope"])
show("DF_paper_decision_value.csv", cols=["base", "margin_pts", "n", "success", "mean_p", "mean_be", "runs_per_attempt", "runs_se", "total_runs"])
fig("Fig_paper_decision_value.png", 760)
""")
md(r"""
## 11 · Who to send in 2027 (paper §4.9)

Green light: clears the one-out break-even rate even against a battery one SD tougher; battery-dependent: only against an
average battery; hold: below it even against an average battery.
""")
code(r"""
G = pd.read_csv(IQ.res_path("DF_paper_greenlight.csv")); G = G[G.base == "2B"].copy(); be = G.be_1out.iloc[0]
G["call"] = np.where(G.p_tough_1out >= be, "green light", np.where(G.p_1out >= be, "battery-dependent", "hold"))
display(G.call.value_counts())
display(G.sort_values("p_1out", ascending=False)[["name", "attempts_2025_26", "speed", "p_1out", "p_tough_1out", "call"]].head(15).round(3))
fig("Fig_paper_greenlight.png", 760)
""")
md(r"""
# Part B · Earlier analyses (the archived technical report)

The analyses of the earlier technical report (`4-Archive/reports/StealIQ_Technical_Report_2026-10-06.pdf`): season
metrics, the learner choice, the input ladder, Powers et al.'s speed models, the public models and the video delivery
time. They pool steals of second and third and use inputs measured during the pitch; for decisions, Part A supersedes them.
""")

# ── 2 season metrics ──
md(r"""
## B1 · Rating base-stealers over a season: Steal+ and Burst (technical report §7)

`fit_league()` learns the league constants once (the speed → success line, the speed → ground line, the percentile
references); `score()` is then a pure function of those constants, so scoring 1, 5 or every runner-season gives identical
rows. *Live:* the n = 1 → n = 5 → full test, asserting the decomposition `net bases = NetSpeed + Steal+` closes exactly and
the small runs match the full run byte for byte.
""")
code(r"""
era = IQ.load_era(); fit = IQ.fit_league(era)
one = IQ.score(era[(era.runner_id == 647304) & (era.season == 2025)].copy(), fit)          # Josh Naylor 2025
five = IQ.score(era[era.runner_id.isin(IQ.TEST_IDS)].sort_values(["runner_id", "season"]).head(5).copy(), fit)
full = IQ.score(era.copy(), fit)
for d in (one, five, full):
    IQ.check_invariants(d)
cols = ["steal_plus", "burst_ft", "netspeed", "surplus", "steal_plus_pct", "burst_pct"]
chk = five.merge(full[["runner_id", "season"] + cols], on=["runner_id", "season"], suffixes=("_5", "_full"))
print("max |n=5 - full| over every metric:", max((chk[c + "_5"] - chk[c + "_full"]).abs().max() for c in cols))
print(f"{len(full)} runner-seasons | p_speed = {fit['a0']:.3f} + {fit['a1']:.4f}*speed | ground = {fit['b0']:.2f} + {fit['b1']:.3f}*speed")
display(one[["player_name", "season", "sprint_speed", "sb_attempts", "net_sb", "netspeed", "steal_plus", "burst_ft"]].round(2))
""")
code(r"""
if RERUN: IQ.run_v15()          # leaderboard, validation, reliability, the shipped calculator, the site payload
show("DF_v15_validation.csv")
show("DF_v15_reliability.csv")
fig("Fig_v15_Reliability.png")
""")

# ── 3 the calculator and why LR ──
md(r"""
## B2 · Why a logistic regression (technical report §2)

The calculator's job is a probability a coach reads as a chance, from a few measurable inputs, with effects he can reason
about, on about 12,000 attempts. A logistic regression gives one coefficient per input (log-odds per unit), runs as one
formula in the browser, and is nearly calibrated before any correction. The question is whether a more flexible learner
would rank attempts better. Same rows, same outer folds; boosted trees are tuned *inside* each training fold (nested), and
confidence intervals resample runners (post-pitch) or pitchers (pre-pitch).
""")
code(r"""
if RERUN: IQ.run_algo()
show("DF_add_algo.csv", cols=["features", "learner", "n", "auroc", "brier", "auroc_vs_logistic", "ci_lo", "ci_hi"])
""")
md(r"""
**More inputs in a boosted engine vs the right inputs in a logistic.** The earlier v12 engine is XGBoost on 18 inputs
(leads, runner season skill, hands, pitch class, outs, inning, catcher arm, catcher tendency). Compare its random and
forward-in-time AUROC with the 7-input v16 logistic in §4.
""")
code(r"""
if RERUN: IQ.run_v12(); IQ.run_v13()
show("DF_v12_AUC.csv"); show("DF_v12_Validation.csv")
fig("Fig_v12_AUC.png", 700)
""")
md(r"""
**The shipped calculator (v15)** — speed, lead at first move, ground gained to release, catcher pop time — and its
calibration. The raw logistic's calibration error is a *wave* across deciles whose signed errors cancel, so one intercept
or slope correction (Platt) cannot fix it; isotonic regression bends with it. Scored without the map ever seeing the rows
(nested), it removes the off-target deciles at a cost of a few thousandths of AUROC.
""")
code(r"""
sm = show("DF_success_model.csv")
h = IQ.SITE.read_text(encoding="utf-8")
payload = json.loads(h.split("/*__SUCCESS_MODEL__*/", 1)[1].split("/*__END_SUCCESS_MODEL__*/", 1)[0])
print({k: payload[k] for k in ["n", "auc", "auc_cal", "brier", "brier_cal", "ece_raw", "ece_cal", "bad_deciles_raw", "bad_deciles_cal"]})
print(f"isotonic map shipped to the browser: {len(payload['calibration']['x'])} breakpoints")
fig("Fig_ROC_Calculator.png", 520)
""")

# ── 4 inputs and v16 ──
md(r"""
## B3 · The inputs, and what each adds (technical report §3)

Each input is added in turn on the same attempts with the same learner. The three inputs public models already use come
first; then the one public data lacks — **ground gained between the pitcher's first move and release** (the secondary
lead) — then the pitch thrown and the base.
""")
code(r"""
if RERUN: IQ.run_baselines()
show("DF_cmp_ladder.csv", cols=["step", "why", "auroc", "gain", "log_loss"])
fig("Fig_cmp_ladder.png")
""")
md(r"""
**v16 = the shipped 4 inputs + pitch type + base**, with a leakage audit, a nested Optuna search over the logistic's design
(penalty, splines, interactions, class weight; objective = log-loss, a proper scoring rule, because the output is a
probability), standardized weights and a forward test (train 2023–25, test 2026, tuned on 2023–25 only).
The search uses two parallel workers, so the chosen settings can differ slightly between runs; the conclusion has not.
""")
code(r"""
if RERUN: IQ.run_v16(60, 200); IQ.run_v16_checks()    # ~40 min
show("DF_v16_models.csv", cols=["model", "auroc", "log_loss", "brier", "ece", "bad_deciles", "d_auc_vs_shipped",
                                "d_auc1_lo", "d_auc1_hi", "d_auc_vs_v16default", "d_auc2_lo", "d_auc2_hi"])
show("DF_v16_forward.csv", cols=["model", "auroc", "log_loss", "ece", "n", "test"])
fig("Fig_v16_tuning.png")
""")
code(r"""
show("DF_v16_leakage_audit.csv")
show("DF_v16_base.csv", cols=["n", "n_3b", "safe_2b", "safe_3b", "auroc_v16", "auroc_v16_plus_base", "d_auroc", "d_auroc_lo",
                              "d_auroc_hi", "or_3b", "or_3b_lo", "or_3b_hi", "mean_lead_2b", "mean_lead_3b"])
show("DF_v16_weights.csv", cols=["variable", "unit", "logodds_per_unit", "or_per_unit", "logodds_per_sd", "or_per_sd",
                                 "share_of_weight", "swing_pp", "p"])
fig("Fig_v16_weights.png", 800)
""")
md(r"""
**Calibration drift.** League success falls every season since the 2023 rules, so a model fit on past seasons
over-predicts the next. Every recalibrator below is fit on training seasons' out-of-fold predictions only.
""")
code(r"""
show("DF_v16_recal.csv", cols=["test_season", "method", "observed", "mean_pred", "gap_pp", "auroc", "ece", "bad_deciles"])
""")

# ── 5 Powers ──
md(r"""
## B4 · Does speed matter? Powers et al. and mixed-effects models (technical report §4)

**Why a mixed-effects model here, and not in the calculator.** For *inference* — how big is speed's effect, and how sure
are we — repeated attempts by the same runner and against the same pitcher are not independent; random intercepts per
runner and pitcher (partial pooling) and runner-clustered errors account for that, which is why Powers et al. use a GLMM.
For *prediction*, a random intercept only helps for a player seen often, needs player identities the calculator does not
take, and complicates calibration; so the calculator is a plain logistic and clustering is respected in its validation.

**The test.** Faster runners gain *less* ground before release, and ground helps; leave ground gained out (as public data
must) and that path is folded into speed's coefficient. Identical rows for M0–M3 (asserted); mediation by a 1,000-resample
runner-clustered bootstrap; random intercepts via statsmodels' variational-Bayes binomial GLMM.
""")
code(r"""
if RERUN: IQ.run_powers(1000)    # ~25 min
S = pd.read_csv(IQ.res_path("DF_add_speed_suppression.csv"))
display(S[S.section == "model"][["model", "terms", "n", "beta_speed", "se", "ci_lo_cluster", "ci_hi_cluster", "auroc_oof", "max_vif"]].round(4))
display(S[S.section == "mediation"][["model", "estimate", "ci_lo", "ci_hi"]].round(4))
display(S[S.section == "random_effects"][["model", "beta_speed", "se", "re_sd_runner", "re_sd_pitcher"]].round(4))
print("verdict:", json.loads(IQ.res_path("DF_add_meta.json").read_text())["verdict"])
fig("Fig_add_speed_suppression.png", 760)
""")
md(r"""
### "Speed is significant": true, and small, in every season (technical report §4.4)

Significance says an effect can be told from zero with this many attempts; it does not say the effect is large. Season by
season on the shipped calculator's rows: stolen bases and caught stealing per 0.5 ft/s of sprint speed (with the runners in
each bin), the share safe per bin with Wilson 95% intervals, speed alone as a logistic (runner-clustered p) next to its
size (10th→90th percentile swing in points, AUROC), the same for ground gained, and the runner-level correlation of speed
with success rate. Then the power arithmetic: how large a gap a given number of attempts can detect.
""")
code(r"""
if RERUN: IQ.run_speed()
SP = show("DF_cmp_speed_by_season.csv", cols=["season", "attempts", "success", "speed_beta", "speed_p", "speed_swing_pp", "speed_auroc",
                                              "ground_swing_pp", "ground_auroc", "runner_seasons", "r_runner", "r_lo", "r_hi"]).set_index("season")
fig("Fig_cmp_speed_by_season.png", 760)
from scipy.stats import norm
z, p0 = norm.ppf(0.975) + norm.ppf(0.8), SP.loc["2023-26", "success"]
for n in (30, 100, 500, 1000, 3000):
    print(f"{n:>5,} attempts per group: smallest detectable gap {100 * z * np.sqrt(2 * p0 * (1 - p0) / n):4.1f} pts "
          f"(80% power, alpha 0.05); one group's rate ±{100 * norm.ppf(0.975) * np.sqrt(p0 * (1 - p0) / n):.1f} pts")
""")

# ── 6 public ──
md(r"""
## B5 · The public standard vs this model (technical report §5)

Powers et al. (v1, Table 1) and *The Pitcher's Dilemma* applied exactly as published (intercept re-fit on training folds
for a fair calibration test), their specifications refit here as their best case, and this project's v15 and v16 — on
identical attempts, with runner-clustered confidence intervals.
""")
code(r"""
show("DF_cmp_baselines.csv", cols=["family", "model", "auroc", "auroc_lo", "auroc_hi", "log_loss", "ece", "bad_deciles"])
fig("Fig_cmp_baselines.png")
fig("Fig_cmp_speed_vs_ground.png")
show("DF_cmp_runner_level.csv")
""")

# ── 7 delivery ──
md(r"""
## B6 · The pitcher's delivery time, from video (technical report §6)

Lead-foot lift-off to release, measured on each attempt's broadcast clip by `1-Data-Ingestion/vision/delivery.py`, with a
camera gate that refuses clips that cut away during the delivery. *Live:* accuracy against the 50 hand-labelled clips.
""")
code(r"""
g = pd.read_csv(IQ.DELIVERY / "delivery_gold.csv"); ok = g[g.qa == "PASS"]
e = ((ok.release_frame - ok.lift_frame) - (ok.gold_release - ok.gold_lift)) / ok.fps
print(f"gold clips {len(g)}, pass QA {len(ok)}: delivery MAE {1000 * e.abs().mean():.0f} ms, CV short by {1000 * -e.mean():.0f} ms on average")
for s in IQ.timed_seasons(0.0):
    c = IQ.load_delivery(s); print(s, f"{len(c):,} processed, {(c.qa == 'PASS').sum():,} timed, confidence {c.confidence.value_counts().to_dict()}")
""")
md(r"""
**After the pitch, delivery time is not a new lever:** with the shipped weights frozen as an offset, it earns no coefficient,
because each extra 0.1 s of delivery buys the runner more ground — which is already an input.
""")
code(r"""
if RERUN: IQ.run_prepitch(); IQ.run_needle(); IQ.run_context()
d = show("DF_add_delivery_beta.csv", cols=["model", "n", "beta_per_0p1s", "ci_lo", "ci_hi", "odds_mult_per_0p1s", "p"], query="model != '_summary'")
ext = json.loads(IQ.res_path("DF_add_meta.json").read_text())["delivery"]
print(f"ground gained per +0.1 s of delivery: {ext['gain_per_0p1s']:+.2f} ft (SE {ext['gain_per_0p1s_se']:.2f})")
fig("Fig_add_delivery_beta.png")
""")
md(r"""
**Before the pitch,** a runner has his speed, his lead at the first move, the catcher, and what he knows about the pitcher.
Profiles from 2023 are scored on 2024–26 (seasons they never saw).
""")
code(r"""
show("DF_add_prepitch.csv", cols=["sample", "model", "n", "auroc_oof", "auroc_gain", "gain_ci_lo", "gain_ci_hi"])
show("DF_add_needle.csv", cols=["input", "alone_or_per_sd", "alone_auroc", "with_speed_gain", "all6_or_per_sd", "all6_swing_pp", "drop_one_loss"],
     query="input != '_summary'")
fig("Fig_add_needle.png")
""")
md(r"""
**Who supplies the ground?** Jump speed = ground gained ÷ delivery time splits the runner's part from the pitcher's. Jump
speed contains ground gained, so it is only ever tested on a runner's *other* attempts.
""")
code(r"""
if RERUN: IQ.run_jump(); IQ.run_prior()
show("DF_add_jump_tests.csv", cols=["test", "item", "value", "lo", "hi", "n", "gain"])
fig("Fig_add_jump_tests.png")
show("DF_add_prior_delivery.csv", cols=["sample", "model", "n", "auroc", "gain", "lo", "hi"])
fig("Fig_add_prior_delivery.png")
""")

# ── 8 decision ──
md(r"""
## B7 · Who chooses to run (technical report §8)

Every attempt above is one a runner chose to make. The decision model scores every pitch with a runner on 1st and 2nd empty.
""")
code(r"""
show("DF_v13_Attempt.csv")
show("DF_v13_SelectionEffect.csv")
fig("Fig_v13_Attempt.png", 640)
""")

# ── summary ──
md(r"""
## Summary — the paper's findings, figure by figure

Printed by the same function that writes the paper's discussion (`build_paper.findings`), so the two cannot disagree.
""")
code(r"""
import build_paper
for f in build_paper.findings(build_paper.Data()):
    display(Markdown(f"**Figure {f['n']}. {f['title']}.** {f['bluf']} *Use:* {f['use']}"))
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells); executing")
subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=900", str(OUT)], check=True)
print(f"executed {OUT}")
