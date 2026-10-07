#!/usr/bin/env python3
"""
core.py — what every StealIQ module shares: paths, the output folders (shelves), the loaders for the per-season meta
tables, and the few models and feature lists used by more than one part (the v16 pipeline, the published-model inputs,
the pitch-type map, the v15 feature lists, calibration error). Nothing here writes results.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import functools, io, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent                      # 2-Data-Analysis/
INGEST = ROOT.parent / "1-Data-Ingestion" / "data"
RAW, META, DELIVERY = INGEST / "raw", INGEST / "meta", INGEST / "delivery"
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
SITE = ROOT.parent / "docs" / "index.html"                  # the live calculator (GitHub Pages serves docs/)

WEBAPP = ROOT.parent / "5-Live-Webapp"                      # the live site's code and outputs

# every output sits on one shelf, picked by its name prefix: results/<shelf>/ and figures/<shelf>/ here, or the live
# site's own results/ and figures/ in 5-Live-Webapp/ for anything without a listed prefix
SHELVES = {"DF_paper_": "1-Paper-GLMM", "Fig_paper_": "1-Paper-GLMM",          # the paper: success + decision GLMMs
           "DF_v16_": "2-v16-Logistic", "Fig_v16_": "2-v16-Logistic",      # v16 pooled logistic (the previous best)
           "DF_v12_": "3-Engines-v12-v13", "DF_v13_": "3-Engines-v12-v13", "Fig_v12_": "3-Engines-v12-v13",
           "Fig_v13_": "3-Engines-v12-v13",
           "DF_add_": "4-Studies", "Fig_add_": "4-Studies",                # Powers, delivery time, pre-pitch, jump speed
           "DF_cmp_": "5-Public-Baselines", "Fig_cmp_": "5-Public-Baselines"}  # published models vs this project
INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe7"   # figure palette


def _shelved(base: Path, name: str) -> Path:
    shelf = next((v for k, v in SHELVES.items() if name.startswith(k)), None)
    path = base / shelf / name if shelf else WEBAPP / base.name / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def res_path(name: str) -> Path:
    """Where a result table lives: results/<shelf>/<name>, or 5-Live-Webapp/results/<name> for the live site's files."""
    return _shelved(RESULTS, name)


def fig_path(name: str) -> Path:
    """Where a figure lives: figures/<shelf>/<name>."""
    return _shelved(FIGURES, name)


# ── loaders: the meta tables are the attempt-level input of every model ──────
ATTEMPT_COLS = ["runner_id", "season", "date", "play_id", "pitcher_id", "pitcher_name", "catcher_id", "catcher_name",
                "fielder_name", "base", "result", "run_value", "lead_at_firstmove_ft", "gain_to_release_ft",
                "lead_at_release_ft", "y"]
CONTEXT_COLS = ["is_lhp", "bat_side_r", "pitch_code", "balls", "strikes", "outs", "inning", "score_diff"]


@functools.lru_cache(maxsize=1)
def _meta() -> pd.DataFrame:
    files = sorted(META.glob("meta_20[0-9][0-9].csv"))
    assert files, f"no meta tables in {META}: run python3 1-Data-Ingestion/ingest.py meta"
    return pd.concat([pd.read_csv(f, low_memory=False) for f in files]).sort_values("attempt_idx").reset_index(drop=True)


def load_meta() -> pd.DataFrame:
    """Every season's meta table in the canonical attempt order (attempt_idx), which fixes the cross-validation folds."""
    return _meta().copy()


def _as_read(df: pd.DataFrame) -> pd.DataFrame:
    """The frame exactly as read_csv would type it (integral float columns back to int)."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def load_attempts() -> pd.DataFrame:
    """One row per tracked running event 2023+ (filter result to SB/CS for steal attempts): the drawer columns."""
    return _as_read(_meta()[ATTEMPT_COLS])


def load_context() -> pd.DataFrame:
    """MLB-feed pitch context per attempt. balls / strikes are the count AFTER the pitch and score_diff the END-of-PA
    score: kept for the record (and for the engines' ablation) but never a model input."""
    m = _meta()
    c = m.loc[m.feed_is_lhp.notna(), ["play_id"] + [f"feed_{k}" for k in CONTEXT_COLS]]
    c = c.rename(columns=lambda k: k.removeprefix("feed_"))
    for k in CONTEXT_COLS:
        if c[k].dtype.kind == "f" and c[k].notna().all() and (c[k] % 1 == 0).all():
            c[k] = c[k].astype("int64")
    return _as_read(c)


def load_delivery(season: int) -> pd.DataFrame:
    """The CV timing run of one season (vision/delivery.py): QA verdict, delivery_s, confidence per attempt."""
    return pd.read_csv(DELIVERY / f"delivery_{season}.csv")


def timed_seasons(min_share: float = 0.95) -> list:
    """Seasons whose CV timing run is complete (>= min_share of the season's SB/CS attempts processed)."""
    a = load_attempts(); n = a[a.result.isin(["SB", "CS"])].groupby("season").size()
    out = []
    for f in sorted(DELIVERY.glob("delivery_20[0-9][0-9].csv")):
        s, k = int(f.stem[-4:]), len(pd.read_csv(f, usecols=["play_id"]))
        if k >= min_share * n.get(s, np.inf):
            out.append(s)
        else:
            print(f"skipping {f.name}: {k} of {n.get(s, 0)} attempts timed so far")
    return out


# ════ Shared models and features (used by more than one of paper.py, research.py and 5-Live-Webapp/webapp.py) ════
# ── per-attempt model: the ~11k-attempt SB-success AUC (the project's grain) ──
# Confirms the thesis quantitatively: the per-pitch LEAD distances (which drive Burst)
# predict whether an individual attempt succeeds. Heavy deps (xgboost/sklearn) are
# imported lazily so the season model above never depends on them.
# Only the two INDEPENDENT lead quantities: where he already was when the pitcher committed, and
# how much he gained from there. lead_at_release_ft is deliberately EXCLUDED because it is their
# exact sum (lead_at_release = lead_at_firstmove + gain_to_release, R^2 = 0.999895; the residual
# caps at 0.1 ft, which is just the rounding granularity). Carrying all three gave VIFs of
# 1,588 / 5,836 / 9,532 — harmless for XGBoost's predictions, but it split feature importance
# arbitrarily across perfectly dependent columns, which made the importance chart unquotable.
# Dropping it costs 0.0014 AUROC (inside noise) and buys an interpretable model.
PA_LEAD_FEATS = ["lead_at_firstmove_ft", "gain_to_release_ft"]


# jump_time is omitted: it is a second measurement of the same thing as sprint_speed (r = -0.59,
# and bolts r = +0.71), which pushed sprint_speed to VIF 16.8. Dropping it takes max VIF to 6.6 AND
# nudges AUROC up (0.7820 -> 0.7829), so nothing is traded away. jump_time is still carried in the
# data and shown on the player card — it is only excluded as a MODEL feature.
PA_RUNNER_FEATS = ["sprint_speed", "accel_gap", "primary_lead", "lead_gain", "bolts"]


# ── the whiteboard model: 4 inputs, plain logistic regression ────────────────
# Chosen by measurement, not taste. Every candidate spec was fit on the sample it could actually
# ship on (recorded in the archived AUC_Roadmap):
#   speed + burst + gain                     n= 7,404   AUROC 0.7259   <- the old spec
#   speed + burst + firstmove + gain + pop   n= 7,264   AUROC 0.7470
#   speed + firstmove + gain + pop           n=10,844   AUROC 0.7559   <- this one
# Burst is DROPPED here and only here. It needs a qualified runner-season (>=10 attempts), so
# carrying it discarded ~3,600 attempts, and once the model already knows what the runner gained
# on THIS pitch it added only +0.002 AUROC while taking a confusing negative coefficient. Burst
# remains the season-level technique metric on the leaderboard, where it is speed-neutral and the
# most repeatable number in the project — it is simply not a per-pitch input.
# Catcher pop time replaces it: worth ~8x more (+0.017), and it is a genuine per-attempt fact.
SIMPLE_FEATS = ["sprint_speed", "lead_at_firstmove_ft", "gain_to_release_ft", "pop_faced"]


def expected_calibration_error(y, p, bins: int = 10) -> tuple[float, int]:
    """(ECE, number of deciles more than 2 binomial SE from their predicted rate).

    Equal-COUNT quantile bins, n-weighted mean |observed - predicted|. The second number is the
    one that actually matters: ECE averages signed errors away, so a model whose deciles miss by
    +4 and -4 points can post a respectable ECE while being wrong everywhere."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    total, beyond, n = 0.0, 0, len(y)
    for b in np.unique(q):
        m = q == b
        obs, pred, nb = float(y[m].mean()), float(p[m].mean()), int(m.sum())
        se = np.sqrt(max(obs * (1 - obs), 1e-12) / nb)
        total += (nb / n) * abs(obs - pred)
        if abs(obs - pred) / se > 2:
            beyond += 1
    return float(total), int(beyond)


# Pitch codes collapse to four classes — per-code dummies would be mostly noise at this n.
PITCH_CLASS = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball", "FA": "fastball", "FT": "fastball",
    "SL": "breaking", "CU": "breaking", "KC": "breaking", "ST": "breaking", "SV": "breaking",
    "SC": "breaking", "CS": "breaking", "KN": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "EP": "offspeed",
}


F4 = SIMPLE_FEATS                                  # speed, firstmove, gain, pop — shipped order


def shipped_rows() -> pd.DataFrame:
    """run_success_model()'s rows, lines for lines (Burst is computed there but is not a feature and
    does not affect which rows survive, so it is skipped)."""
    at = load_attempts()
    at = at[at["result"].isin(["SB", "CS"])].copy()
    at["y"] = (at["result"] == "SB").astype(int)
    lb = pd.read_csv(res_path("DF_v15_leaderboard.csv"))[["runner_id", "season", "sprint_speed", "burst_ft"]]
    at = at.merge(lb, on=["runner_id", "season"], how="left")
    sp = pd.read_csv(RAW / "sprint_speed.csv")
    at = at.merge(sp, on=["runner_id", "season"], how="left")
    at["sprint_speed"] = at["sprint_speed"].fillna(at["sprint_speed_all"])
    pop = pd.read_csv(RAW / "poptime.csv")[["catcher_id", "season", "pop_2b_sba"]] \
            .rename(columns={"pop_2b_sba": "pop_faced"})
    at = at.merge(pop, on=["catcher_id", "season"], how="left")
    at[F4] = at[F4].apply(pd.to_numeric, errors="coerce")
    return at.dropna(subset=F4).reset_index(drop=True)


def ci(b, se):
    return b - 1.96 * se, b + 1.96 * se


CONT = F4                                          # speed, first-move lead, ground gained, pop


BIN = ["pt_breaking", "pt_offspeed", "base_is_3b"]   # references: fastball, stealing 2nd


FEATS = CONT + BIN


BANNED = {"balls", "strikes", "ahead_in_count", "score_diff", "run_value", "result", "y", "lead_at_release_ft"}


def v16_rows() -> pd.DataFrame:
    c = load_context()[["play_id", "pitch_code"]]
    d = shipped_rows().merge(c, on="play_id", how="left")
    d["pitch_class"] = d.pitch_code.map(PITCH_CLASS)
    d = d[d.pitch_class.notna()].reset_index(drop=True)
    d["pt_breaking"] = (d.pitch_class == "breaking").astype(int)
    d["pt_offspeed"] = (d.pitch_class == "offspeed").astype(int)
    d["base_is_3b"] = (d.base.astype(str) == "3B").astype(int)
    assert not BANNED & set(FEATS), "a banned (post-outcome) column is in the feature list"
    return d


class Design(BaseEstimator, TransformerMixin):
    """Logistic design matrix, learned on the TRAINING fold only: z-score the 4 continuous inputs, optionally
    expand them into B-splines, optionally add interactions, then z-score every column so one penalty treats
    all columns alike. Columns 0-3 are continuous, 4+ the binary columns (pitch type dummies, base)."""
    def __init__(self, spline=False, n_knots=4, degree=3, inter="none"):
        self.spline, self.n_knots, self.degree, self.inter = spline, n_knots, degree, inter

    def _raw(self, X):
        c = (X[:, :4] - self.mu_) / self.sd_; b = X[:, 4:]
        parts = [self.spl_.transform(c) if self.spline else c, b]
        if self.inter in ("binary_x_cont", "pairwise"):
            parts.append(np.hstack([c * b[:, [j]] for j in range(b.shape[1])]))
        if self.inter == "pairwise":
            parts.append(np.column_stack([c[:, i] * c[:, j] for i in range(4) for j in range(i + 1, 4)]))
        return np.hstack(parts)

    def fit(self, X, y=None):
        self.mu_, self.sd_ = X[:, :4].mean(0), X[:, :4].std(0)
        if self.spline:
            self.spl_ = SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=False).fit(
                (X[:, :4] - self.mu_) / self.sd_)
        Z = self._raw(X); self.zmu_, self.zsd_ = Z.mean(0), Z.std(0) + 1e-12
        return self

    def transform(self, X):
        return (self._raw(X) - self.zmu_) / self.zsd_


DEFAULT = dict(penalty="l2", C=1.0, spline=False, inter="none", class_weight="none")


def pipe(p: dict) -> Pipeline:
    pen = p["penalty"]
    lr = LogisticRegression(penalty=None if pen == "none" else pen, C=p.get("C", 1.0),
                            l1_ratio=p.get("l1_ratio") if pen == "elasticnet" else None,
                            solver="saga" if pen in ("l1", "elasticnet") else "lbfgs",
                            class_weight=None if p.get("class_weight", "none") == "none" else "balanced",
                            max_iter=5000, tol=1e-4)
    return Pipeline([("design", Design(p.get("spline", False), p.get("n_knots", 4), p.get("degree", 3),
                                       p.get("inter", "none"))), ("lr", lr)])


SPLIT_D = np.arange(5, 95, 5)


def pd_jump_speed(d: pd.DataFrame) -> pd.Series:
    """The Pitcher's Dilemma 'average jump speed': per runner-season power law t = a·d^b on the 5-ft splits."""
    s = pd.read_csv(RAW / "Raw_Season.csv")
    cols = [f"seconds_since_hit_{x:03d}" for x in SPLIT_D]
    fit = {}
    for r in s.dropna(subset=cols).itertuples(index=False):
        t = np.array([getattr(r, c) for c in cols], float)
        b, loga = np.polyfit(np.log(SPLIT_D), np.log(t), 1)          # log t = log a + b log d
        fit[(r.runner_id, r.season)] = (np.exp(loga), b)
    ab = d[["runner_id", "season"]].apply(lambda x: fit.get((x.runner_id, x.season), (np.nan, np.nan)), axis=1)
    a_, b_ = np.array([v[0] for v in ab]), np.array([v[1] for v in ab])
    g = d.gain_to_release_ft.clip(lower=0.5).values
    per = g / (a_ * g ** b_)                                          # ft per second over the jump
    avg = pd.Series(per, index=d.index).groupby([d.runner_id, d.season]).transform("mean")
    return avg


def baseline_rows() -> pd.DataFrame:
    d = v16_rows()
    pop = pd.read_csv(RAW / "poptime.csv")[["catcher_id", "season", "maxeff_arm_2b_3b_sba"]].rename(
        columns={"maxeff_arm_2b_3b_sba": "arm"})
    d = d.merge(pop, on=["catcher_id", "season"], how="left")
    d["jump_speed"] = pd_jump_speed(d)
    d["jump_covered"] = d.jump_speed.notna()
    d["jump_speed"] = d.jump_speed.fillna(d.jump_speed.median())    # runners without splits: league median
    d["post2023"] = 1.0
    return d.dropna(subset=["arm"]).reset_index(drop=True)


