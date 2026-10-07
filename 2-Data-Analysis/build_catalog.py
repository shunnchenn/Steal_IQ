#!/usr/bin/env python3
"""
build_catalog.py — writes the repository's catalog: CATALOG.md and 0-Catalog.pdf at the root (every folder, data file,
result and figure: what it is and which code uses it, found by scanning the code) and a README.md in every results/
and figures/ folder with that folder's headline number. build_paper.py runs it after the PDFs.

Usage:  python3 build_catalog.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stealiq as IQ  # noqa: E402
from stealiq import res_path  # noqa: E402

REPO = IQ.ROOT.parent
CODE = {"1-Data-Ingestion/ingest.py": "ingest.py", "1-Data-Ingestion/vision/delivery.py": "delivery.py",
        "2-Data-Analysis/core.py": "core.py", "2-Data-Analysis/paper.py": "paper.py", "2-Data-Analysis/research.py": "research.py",
        "5-Live-Webapp/webapp.py": "webapp.py", "2-Data-Analysis/build_paper.py": "build_paper.py",
        "2-Data-Analysis/build_notebook.py": "build_notebook.py", "2-Data-Analysis/literature.py": "literature.py"}
DESC = {
    "Raw_Attempts.csv": ("play_id", "Every tracked running event 2023+ from Savant's drawer (SB/CS + PK/BK/FB): runner, pitcher, catcher, base, result, lead at first move, ground gained to release. The canonical attempt order."),
    "Raw_Attempt_Context.csv": ("play_id", "MLB-feed context per attempt: hand, batter side, pitch code, outs, inning. balls/strikes are post-pitch and score_diff end-of-PA: never inputs."),
    "Raw_Statcast_Pitches.csv.gz": ("play_id", "The Statcast row (all columns) of every attempt pitch, matched via the MLB feed's (at_bat_number, pitch_number)."),
    "Raw_Season.csv": ("runner_id, season", "Runner-season table: sprint speed and running splits, SB, CS, attempts, team (v7 features + live overlay)."),
    "Raw_Opportunities.csv.gz": ("play_id", "Every pitch with a runner on 1st and 2nd empty: the decision-model denominator."),
    "poptime.csv": ("catcher_id, season", "Savant catcher pop time, arm strength and exchange."),
    "sprint_speed.csv": ("runner_id, season", "Statcast sprint speed for every runner."),
    "team_map.csv": ("runner_id, season", "Runner's team by season (site display)."),
    "league_sb_rates.csv": ("season", "MLB regular-season SB, CS, games, success % and attempts per game (StatsAPI, all 30 teams)."),
    "people.csv": ("player_id", "MLB names for every runner and batter id (StatsAPI)."),
    "statcast_csv_docs.csv": ("field", "Savant's own definition of each Statcast CSV field (baseballsavant.mlb.com/csv-docs)."),
    "DF_v7_SSSI.csv": ("runner_id, season", "Legacy v7 season features (runner skill): an input to Raw_Season and the engines."),
    "DF_v7_xSB_Outcome.csv": ("runner_id, season", "Legacy v7 expected-SB outcomes: an input to Raw_Season."),
    "data_dictionary.csv": ("column", "Every meta-table column: block, source, description, % non-null; dropped columns listed."),
    "delivery_gold.csv": ("play_id", "The CV scored on the 50 hand-labelled gold clips (accuracy check)."),
}
for s in range(2023, 2040):
    DESC[f"meta_{s}.csv"] = ("play_id + runner_id", f"THE {s} TABLE: one row per tracked running event, every Statcast field, MLB-feed context, season pop time and sprint speed, CV delivery time, video links, qa_flags.")
    DESC[f"delivery_{s}.csv"] = ("play_id", f"CV delivery time for every {s} SB/CS attempt: QA verdict, frames, delivery_s, confidence.")
RES = {"DF_calculator.csv": "THE CALCULATOR: the success GLMM's fixed part per base, term by term as the page runs it, and its 2026 forward-test AUROC.",
       "DF_success_model.csv": "v15 four-input logistic on 2023-26 (intercept, 4 coefficients, CV AUROC): its lead weights define Ground and Burst. The 2015-22 calculator runs the same four inputs, fit on that era.",
       "DF_v15_leaderboard.csv": "Published leaderboard: Steal+, Burst, percentiles.", "DF_v15_validation.csv": "Validation behind every season-metric claim.",
       "DF_v15_reliability.csv": "How much of one season's Steal+ is signal vs chance.", "DF_v15_reliability_top.csv": "Top runner-seasons with chance bands.",
       "DF_perattempt_AUC.csv": "Per-attempt XGBoost AUROC ladder.", "DF_perattempt_Importance.csv": "Per-attempt XGBoost feature importance.",
       "v15_players.json": "The site payload (league fits, validation, players), synced into docs/index.html.",
       "DF_v12_AUC.csv": "v12 success engine: AUROC by feature block.", "DF_v12_Validation.csv": "v12 under random / grouped / forward splits.",
       "DF_v12_Calibration.csv": "v12 calibration by decile.", "DF_v13_Attempt.csv": "v13 decision engine: AUROC/AUPRC by block and split.",
       "DF_v13_Calibration.csv": "v13 forward calibration.", "DF_v13_SelectionEffect.csv": "Attempt rate by sprint-speed quintile.",
       "DF_v13_SpeedPercentiles.csv": "Sprint-speed percentiles in the opportunity population.",
       "DF_add_speed_suppression.csv": "Speed coefficient M0–M3, mediation, random-effect fits, speed bands (technical report §4).",
       "DF_add_delivery_beta.csv": "Delivery-time coefficient five ways (technical report §6.2).", "DF_add_meta.json": "Verdict, reproduced coefficients, delivery summary.",
       "DF_add_prepitch.csv": "2023 delivery profile in a pre-pitch model (technical report §6.3).", "DF_add_needle.csv": "Six pre-pitch inputs on one scale (technical report §6.3).",
       "DF_add_algo.csv": "Logistic vs splines vs XGBoost vs LightGBM (technical report §2.2).", "DF_add_context_prepitch.csv": "Pre-pitch model with game context.",
       "DF_add_pitchtype.csv": "Safe rate and observed − expected by pitch type (technical report §3.3).", "DF_add_pitchtype_state.csv": "Pitch type × game state cells and interaction tests.",
       "DF_add_jump_metric.csv": "Per-attempt jump speed (technical report §6.4).", "DF_add_jump_tests.csv": "Jump-speed tests: decomposition, reliability, out-of-sample value.",
       "DF_add_jump_runners.csv": "Runners' ground split into jump-speed and pitcher-time parts.", "DF_add_prior_delivery.csv": "Earlier delivery as a pre-pitch input (technical report §6.5).",
       "DF_v16_models.csv": "v16 HEADLINE: shipped → + pitch type → + base, tuned, isotonic (technical report §3.2).", "DF_v16_weights.csv": "v16 input weights on one scale (technical report §3.4).",
       "DF_v16_forward.csv": "v16 forward test: train 2023–25, test 2026.", "DF_v16_leakage_audit.csv": "When each candidate input is known (technical report §2.6).",
       "DF_v16_nested_choices.csv": "Configuration chosen in each outer fold.", "DF_v16_final_trials.csv": "Every trial of the final Optuna study.",
       "DF_v16_meta.json": "v16 run summary: final params, importances, MLE coefficients, delivery beta.", "DF_v16_pop_prior.csv": "Same- vs prior-season pop time.",
       "DF_v16_base.csv": "Base-stolen test (technical report §3.3).", "DF_v16_base_slopes.csv": "Separate slopes for 2nd vs 3rd (rejected).",
       "DF_v16_recal.csv": "Drift recalibration methods (technical report §2.5).", "DF_v16_engine_leak_fix.csv": "One-off record: v12/v13 before vs after the score-leak fix.",
       "DF_v16_v12_score_audit.csv": "One-off audit: v12 with no / end-of-PA / clean score.", "DF_v16_v13_score_audit.csv": "One-off audit: v13 with no / end-of-PA / clean score.",
       "DF_cmp_baselines.csv": "Public baselines vs this model (technical report §5.2).", "DF_cmp_ladder.csv": "What each input adds, and why (technical report §3.1).",
       "DF_cmp_seasons.csv": "League success and attempts per game by season (technical report §1.4).", "DF_cmp_runner_level.csv": "Runner-season check of each model (technical report §5.4).",
       "DF_cmp_scenarios.csv": "Four real runner-seasons under each model.",
       "DF_cmp_speed_by_season.csv": "Sprint speed vs success by season and pooled: speed-only logit (clustered p), 10th→90th swing, AUROC, the same for ground gained, runner-level r (technical report §4.4).",
       "DF_cmp_speed_bins.csv": "Per season × 0.5 ft/s speed bin: attempts, SB, CS, runners, share safe with Wilson CI (technical report §4.4).",
       "DF_paper_re24.csv": "Runs to the end of the half-inning by season and base-out state: the run-expectancy table (paper, Appendix A).",
       "DF_paper_breakeven.csv": "Break-even success rate for steals of second and third by outs (paper, Fig 2, Table A2).",
       "DF_paper_success_cv.csv": "Choosing the success model: three specifications, five-fold CV on 2023-25 only, per base (paper, Table C2).",
       "DF_paper_forward_all.csv": "THE HEADLINE: every model fit on 2023-25 and scored on the same 2026 attempts: AUROC (clustered CI, by base), log-loss, Brier, calibration, paired difference from v16 (paper, Table 2, Fig 4).",
       "DF_paper_roc.csv": "ROC curves of every model on 2026 (paper, Fig 4A).",
       "DF_paper_calibration_all.csv": "2026 calibration bins of every model (paper, Fig 4B).",
       "DF_paper_success_fixed.csv": "Success GLMM fixed effects by base, 2023-26, with 10th/90th percentiles (paper, Table B1).",
       "DF_paper_success_variance.csv": "Success GLMM random-intercept SDs with likelihood-ratio intervals (paper, Fig 9B, Table B3).",
       "DF_paper_success_players.csv": "Every player's shrunk effect in the success model: skill net of the jump and the pitch.",
       "DF_paper_ground.csv": "Ground needed by release to clear each break-even rate, by base, outs and catcher, and the share of attempts reaching it (paper, Fig 6).",
       "DF_paper_decision_fixed.csv": "Decision GLMM fixed effects by base (paper, Fig 5, Table B2); rows with model = evaluation are the superseded decision-plus-ground-gained model.",
       "DF_paper_decision_variance.csv": "Decision GLMM random-intercept SDs with likelihood-ratio intervals (paper, Fig 9A, Table B3).",
       "DF_paper_decision_players.csv": "Every player's shrunk effect in the decision model (paper, Table 5).",
       "DF_paper_decision_forward.csv": "Decision GLMM forecast test by base against simpler pre-pitch models (calibration quoted in paper §4.3).",
       "DF_paper_decision_calibration.csv": "2026 calibration bins of the decision GLMM by base.",
       "DF_paper_decision_value.csv": "2026 runs per attempt by the decision model's margin over break-even, and go / hold totals (paper, Fig 10).",
       "DF_paper_greenlight.csv": "2027 green-light list: each recent runner's predicted success by outs, average and tough battery (paper, Fig 11, Table 4).",
       "DF_paper_glmm_check.csv": "GLMM recovery on 20 simulated data sets and the statsmodels cross-check (paper, Table C1).",
       "DF_paper_speed_2b.csv": "Sprint speed and steals of second by season: slope, p, swing, AUROC, runner-level r (paper, Fig 7).",
       "DF_paper_speed_bins.csv": "Steals of second by sprint-speed bin: stolen bases and caught stealing by season, success, runner-seasons, attempts per runner-season (paper, Table 3).",
       "DF_paper_runner_speed.csv": "Runner-season speed vs success: correlation with and without Josh Naylor, binomial noise, reliability, true SD, corrected r (paper, Fig 8).",
       "DF_paper_influence.csv": "Each input's log-odds weight per SD (indicators 0 to 1), decision and success models, by base (paper, Fig 5).",
       "DF_paper_anatomy.csv": "The median steal of second: lead, ground gained, sprint speed, delivery, pitch flight, pop time (paper, Fig 1).",
       "DF_paper_pop_choice.csv": "Steals of third: pop time to third vs to second in the decision model, same attempts (paper, Table C3)."}


# one README per results/ and figures/ folder: what the folder is and, where it has one, its headline number
SHELF_NOTE = {
    "1-Paper-GLMM": ("The paper (StealIQ_Paper.pdf, StealIQ_Figure_Discussion.pdf)",
                     "Per-base logistic mixed models (GLMMs) with runner, pitcher and catcher random intercepts. The success model "
                     "(inputs through the release of the pitch) is the best predictor of a steal; the decision model (pre-pitch "
                     "inputs only) makes the go / hold call against the break-even rate. Written by `stealiq.py decide success "
                     "popchoice paperfigs` (and `glmmcheck`), code in paper.py."),
    "2-v16-Logistic": ("v16 pooled logistic regression, the previous best",
                       "One logistic regression for both bases on sprint speed, lead at first move, ground gained, pop time, pitch "
                       "class and base; the tuned version adds splines and pairwise interactions chosen by Optuna in nested "
                       "cross-validation. Superseded by the success GLMM (paper, Table 2). Written by `stealiq.py v16 v16checks` "
                       "(research.py)."),
    "3-Engines-v12-v13": ("v12 and v13 gradient-boosting engines, kept for the record",
                          "v12 (success) and v13 (attempt decision): AUROC by feature block, validation splits, calibration. "
                          "Written by `stealiq.py v12 v13` (research.py)."),
    "4-Studies": ("Studies behind the archived technical report",
                  "Powers et al. replication and speed suppression, delivery time from video, pre-pitch inputs, game context, "
                  "learner choice, pitch type, jump speed. Written by `stealiq.py powers prepitch needle context algo jump prior` "
                  "(research.py)."),
    "5-Public-Baselines": ("Published models and league context",
                           "Published stolen-base models against this project, the input ladder, sprint speed season by season. "
                           "Written by `stealiq.py baselines speed` (research.py)."),
    "5-Live-Webapp": ("The live site: calculator, leaderboard and season metrics",
                      "What the public site shows: the steal-odds calculator, the season metrics (Steal+, Burst) and the "
                      "payload written into docs/index.html. Written by `python3 5-Live-Webapp/webapp.py`."),
}

FOLDERS = [   # the map that opens 0-Catalog.pdf
    ("1-Data-Ingestion", "Scrapes Baseball Savant and the MLB StatsAPI into committed raw tables and one meta table per season; "
     "times the pitcher's delivery from broadcast video (vision/).", "ingest.py, vision/delivery.py, data/meta/"),
    ("2-Data-Analysis", "The models and the paper: core.py (shared paths, loaders and models), paper.py (the paper's mixed "
     "models), research.py (the archived report's studies), stealiq.py (command line), and the build_*.py scripts that write "
     "the paper, this catalog and the analysis notebook.", "stealiq.py, StealIQ_Paper.ledger.md"),
    ("3-Presentations", "The SABR seminar prototype; the deck itself embeds broadcast video and stays on this machine.", "Saber_Prototype.pdf"),
    ("4-Archive", "Superseded code, reports, video-timing logs and outputs; on this machine only (gitignored).", "README.md"),
    ("5-Live-Webapp", "Builds the live site's calculator and leaderboard and writes them into docs/index.html.", "webapp.py, README.md"),
    ("docs", "The live site, served by GitHub Pages from this folder.", "index.html"),
]


def headline(shelf):
    """The folder's headline number, read from its own result files (empty when the folder has none)."""
    try:
        if shelf == "1-Paper-GLMM":
            F = pd.read_csv(res_path("DF_paper_forward_all.csv")).set_index("model")
            s = F.loc[[m for m in F.index if m.startswith("Success GLMM")][0]]
            return (f"Success GLMM: AUROC {s.auroc:.3f} (95% CI {s.auroc_lo:.3f}-{s.auroc_hi:.3f}) on {int(s.n):,} attempts of 2026, "
                    f"fit on 2023-25; decision GLMM {F.loc['Decision GLMM: before the pitch only', 'auroc']:.3f}; v16 tuned "
                    f"{F.loc['v16 logistic (pooled, tuned on 2023-25)', 'auroc']:.3f} on the same attempts (DF_paper_forward_all.csv).")
        if shelf == "2-v16-Logistic":
            M, W = pd.read_csv(res_path("DF_v16_models.csv")).set_index("model"), pd.read_csv(res_path("DF_v16_forward.csv")).set_index("model")
            return (f"v16: AUROC {M.loc['v16: Optuna-tuned (nested)', 'auroc']:.3f} cross-validated on 2023-26 (DF_v16_models.csv); "
                    f"{W.loc['v16: + pitch type + base (default LR)', 'auroc']:.3f} default and "
                    f"{W.loc['v16 tuned on 2023-25 only', 'auroc']:.3f} tuned, fit on 2023-25 and scored on 2026 (DF_v16_forward.csv).")
        if shelf == "5-Live-Webapp":
            C = pd.read_csv(res_path("DF_calculator.csv")).set_index("term")
            return (f"Calculator: the success GLMM's fixed part, AUROC {C.loc['AUROC (fixed part)', 'beta']:.3f} on the 2026 forward test "
                    "(DF_calculator.csv).")
    except (FileNotFoundError, KeyError, IndexError) as e:
        return f"(headline unavailable: {e!r})"
    return ""


def outputs(kind):
    """Every generated table (kind="results") or figure (kind="figures") on disk: the analysis shelves and the live site."""
    roots = [IQ.RESULTS if kind == "results" else IQ.FIGURES, IQ.WEBAPP / kind]
    return sorted(p for r in roots if r.exists() for p in r.rglob("*")
                  if p.is_file() and not p.name.startswith(".") and p.name != "README.md")


def figure_numbers():
    """Where each figure is shown: its number in the paper (read from build_paper.py), the notebook, the archived technical
    report (on this machine only) or the root README; "—" when nothing shows it."""
    rep = (REPO / "2-Data-Analysis/build_paper.py").read_text()
    used = {n: f"paper, Figure {k}" for n, k in re.findall(r'figure\("(Fig_[^"]+\.png)", (\d+)', rep)}
    shown = [(label, (REPO / f).read_text()) for label, f in (("notebook", "2-Data-Analysis/build_notebook.py"),
             ("technical report (archived)", "4-Archive/code/report_technical_2026-10-06.py"), ("README", "README.md")) if (REPO / f).exists()]
    return {p.name: "; ".join(([used[p.name]] if p.name in used else []) + [label for label, t in shown if p.name in t]) or "—"
            for p in outputs("figures")}


def code_sources():
    """The code scanned for 'Used by', plus the notebook generator's output."""
    src = {k: (REPO / k).read_text() for k in CODE if (REPO / k).exists()}
    nb = REPO / "2-Data-Analysis" / "analysis.ipynb"
    return src, (nb.read_text() if nb.exists() else "")


def users(name, src, nbt):
    stem = name.split(".")[0]
    key = stem[:-2] if stem[-4:].isdigit() else stem          # meta_2023 -> 'meta_20': read through a glob
    u = [CODE[k] for k, s in src.items() if key in s]
    return ", ".join(sorted(set(u + (["analysis.ipynb"] if key in nbt else [])))) or "—"


def shape(p):
    try:
        if p.suffix == ".csv" or p.name.endswith(".csv.gz"):
            return f"{len(pd.read_csv(p, usecols=[0])):,} × {len(pd.read_csv(p, nrows=0).columns)}"
    except Exception:
        pass
    return "—"


DATA = [("1-Data-Ingestion/data/raw", "committed source tables"), ("1-Data-Ingestion/data/meta", "the per-season tables"),
        ("1-Data-Ingestion/data/delivery", "video delivery times")]


def shelf_readmes(fig_numbers):
    """README.md in every results/ and figures/ folder: what it is, its headline, and each file with its description."""
    for kind, root in (("results", IQ.RESULTS), ("figures", IQ.FIGURES)):
        for d in sorted(p for p in root.iterdir() if p.is_dir()) + [IQ.WEBAPP / kind]:
            if not d.exists():
                continue
            key = "5-Live-Webapp" if d.parent == IQ.WEBAPP else d.name
            title, what = SHELF_NOTE.get(key, (key, ""))
            L = [f"# {d.relative_to(REPO).as_posix()}: {title}", "", what, ""]
            h = headline(key)
            L += [f"**Headline.** {h}", ""] if h else []
            files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith(".") and p.name != "README.md")
            if kind == "results":
                L += ["| File | What it is |", "|---|---|"] + [f"| `{p.name}` | {RES.get(p.name, '')} |" for p in files]
            else:
                L += ["| Figure | Used in |", "|---|---|"] + [f"| `{p.name}` | {fig_numbers.get(p.name, '')} |" for p in files]
            (d / "README.md").write_text("\n".join(L) + "\n")
    print("wrote README.md in every results/ and figures/ folder")


def catalog():
    """CATALOG.md at the repository root (the same content as 0-Catalog.pdf, readable on GitHub)."""
    src, nbt = code_sources()
    L = ["# Catalog", "", "Every folder, data file, result and figure in the repository: what it is and which code uses it (found by "
         "scanning the code). Generated by `python3 2-Data-Analysis/build_catalog.py`; the same content as 0-Catalog.pdf.", "",
         "## The repository", "", "| Folder | What it holds | Start with |", "|---|---|---|"]
    L += [f"| `{a}/` | {b} | {c} |" for a, b, c in FOLDERS] + [""]
    TH = ["| File | Key | Rows × cols | Used by | What it is |", "|---|---|---|---|---|"]
    for folder, title in DATA:
        L += [f"## {folder}: {title}", ""] + TH
        for p in sorted((REPO / folder).glob("*")):
            if not p.name.startswith("."):
                k, d = DESC.get(p.name, ("", ""))
                L.append(f"| `{p.name}` | {k} | {shape(p)} | {users(p.name, src, nbt)} | {d} |")
        L.append("")
    L += ["## Results: written by stealiq.py (2-Data-Analysis/results/<shelf>) and webapp.py (5-Live-Webapp/results)", "",
          "| File | Rows × cols | What it is |", "|---|---|---|"]
    L += [f"| `{p.relative_to(REPO).as_posix()}` | {shape(p)} | {RES.get(p.name, '')} |" for p in outputs("results")]
    fn = figure_numbers()
    L += ["", "## Figures", "", "| Figure | Used in |", "|---|---|"]
    L += [f"| `{p.relative_to(REPO).as_posix()}` | {fn.get(p.name, '')} |" for p in outputs("figures")]
    (REPO / "CATALOG.md").write_text("\n".join(L) + "\n")
    print(f"wrote {REPO / 'CATALOG.md'} ({len(L)} lines)")
    shelf_readmes(fn)


def catalog_pdf():
    """0-Catalog.pdf at the repository root: the folder map, then every data file, result and figure, typeset like the paper."""
    import build_paper as BP
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import CondPageBreak, SimpleDocTemplate, Spacer
    # the long tables break across pages, so their headings are not kept with them; CondPageBreak keeps each heading
    # off the foot of a page instead
    H1 = ParagraphStyle("H1c", parent=BP.H1, keepWithNext=0)
    src, nbt = code_sources()
    st = [Spacer(1, 12), BP.para("StealIQ: Catalog", BP.TITLE),
          BP.para("Every folder, data file, result and figure in the repository: what it is and which code uses it", BP.SUB),
          BP.para("1. The repository", BP.H1),
          BP.para("The folders are a research funnel: each stage reads only what the stage before it wrote. Results and figures "
                  "are rebuilt by the code and are not committed; this catalog lists them as they stand on disk, and is rebuilt "
                  "with them (python3 2-Data-Analysis/build_catalog.py)."),
          BP.table([["Folder", "What it holds", "Start with"]] + [[f"<b>{a}</b>", b, c] for a, b, c in FOLDERS], [1.25, 3.75, 1.5])]
    rows, groups = [["File", "Key", "Rows × cols", "Used by", "What it is"]], []
    for folder, title in DATA:
        groups.append(len(rows)); rows.append([f"{folder}: {title}", "", "", "", ""])
        for p in sorted((REPO / folder).glob("*")):
            if not p.name.startswith("."):
                k, d = DESC.get(p.name, ("", ""))
                rows.append([p.name, k, shape(p), users(p.name, src, nbt), d])
    st += [CondPageBreak(1.5 * inch), BP.para("2. Data", H1), *BP.table(rows, [1.6, 0.8, 0.8, 1.05, 2.25], groups=groups, keep=False)]
    rows, groups, last = [["File", "Rows × cols", "What it is"]], [], None
    for p in outputs("results"):
        shelf = p.parent.relative_to(REPO).as_posix()
        if shelf != last:
            groups.append(len(rows)); rows.append([shelf, "", ""]); last = shelf
        rows.append([p.name, shape(p), RES.get(p.name, "")])
    st += [CondPageBreak(1.5 * inch), BP.para("3. Results", H1), *BP.table(rows, [2.2, 0.75, 3.55], groups=groups, keep=False)]
    fn, rows, groups, last = figure_numbers(), [["Figure", "Used in"]], [], None
    for p in outputs("figures"):
        shelf = p.parent.relative_to(REPO).as_posix()
        if shelf != last:
            groups.append(len(rows)); rows.append([shelf, ""]); last = shelf
        rows.append([p.name, fn.get(p.name, "")])
    st += [CondPageBreak(1.5 * inch), BP.para("4. Figures", H1), *BP.table(rows, [3.6, 2.9], groups=groups, keep=False)]

    def footer(c, doc):
        c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(BP.MUTED)
        c.drawString(1.0 * inch, 0.55 * inch, "StealIQ: Catalog"); c.drawRightString(7.5 * inch, 0.55 * inch, str(doc.page))
        c.restoreState()
    out = REPO / "0-Catalog.pdf"
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                      title="StealIQ: Catalog", author="StealIQ").build(st, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}")


def main():
    catalog()
    catalog_pdf()


if __name__ == "__main__":
    main()
