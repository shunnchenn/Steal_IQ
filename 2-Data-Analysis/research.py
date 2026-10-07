#!/usr/bin/env python3
"""
research.py — the studies behind the archived technical report (4-Archive/reports/StealIQ_Technical_Report_2026-10-06.pdf),
kept runnable through stealiq.py. Outputs go to results/ and figures/ shelves 2-v16-Logistic, 3-Engines-v12-v13, 4-Studies
and 5-Public-Baselines.

  §2  per-event engines: v12 success, v13 attempt decision (XGBoost) ........ v12, v13
  §3  Powers et al.: speed suppression, mediation, mixed effects; delivery as a beta ... powers
  §4-6 before the pitch: delivery profile, what moves P(safe), game context ... prepitch, needle, context
  §7  learner choice: logistic vs splines vs boosted trees .................. algo
  §8-9 v16: + pitch type + base, tuning, weights, forward test, checks ...... v16, v16checks
  §10 the public standard vs this model; speed season by season ............. baselines, speed
  §11-12 runner jump speed vs pitcher time; prior delivery (option b) ....... jump, prior
"""
from core import *  # noqa: F401,F403  (paths, loaders, shared models; see core.py)


# ════ 2. Per-event engines: v12 success-given-attempt, v13 attempt-or-not (XGBoost) ═══════════════════════════
# v12: the per-attempt steal model, extended with the three signals v11 was missing. v13: will he attempt at all?
#
# v11 asked: given the ground this runner covered on this pitch, did the attempt succeed?
# It used lead distances + runner skill and reached CV AUROC 0.741 (0.752 with catcher tendency).
# It knew nothing about WHO was pitching, WHAT was thrown, the COUNT, or the catcher's ARM.
#
# v12 adds those, in the order they were expected to pay off:
#   1. battery/pitch   is_lhp, pitch_class, bat_side_r   (MLB play-by-play, joined on play_id)
#   2. count+situation balls, strikes, outs, inning (score_diff removed v16 QA: it is the END-of-PA score)
#   3. catcher arm     pop time to 2B, max-effort arm strength, exchange time (Savant poptime)
#
# v11 is untouched and still the source of the published Steal+/Burst metrics. This section only
# touches the per-attempt SKILL ENGINE, and reports an ablation so each block's contribution is
# visible rather than bundled.
#
# Metrics reported: AUROC (ranking), AUPRC (precision-recall; note the 81% base rate) and Brier
# (calibration) — because a model can rank well and still be badly calibrated.
#
# Inputs: the meta tables (leads + MLB-feed context), raw/poptime.csv (catcher arm), raw/DF_v7_SSSI.csv (runner season
# skill), raw/Raw_Opportunities.csv.gz (v13: every pitch with a runner on 1B and 2B empty).

# pc_fastball is the REFERENCE category and is deliberately omitted: the three pitch-class dummies
# sum to 1 on 99.94% of rows (only 7 "unknown" codes), so carrying all three is the dummy-variable
# trap — it put a VIF of 561 on pc_fastball. Two dummies + an implicit fastball baseline encode the
# same information without the dependency.
BATTERY_FEATS   = ["is_lhp", "bat_side_r", "pc_breaking", "pc_offspeed"]
# score_diff is NOT used anywhere (v16 QA): Raw_Attempt_Context's score comes from the feed's play.result, which is
# the score AFTER the plate appearance, so it can include runs scored after the steal attempt (verified on a live
# game feed; successful steals averaged +0.13 runs). Removing it: v12 AUROC 0.7829 -> 0.7820, v13 0.7797 -> 0.7392. A
# clean start-of-PA score (DF_v16_v12/v13_score_audit.csv) does as well as the leaky one, so the leak did not inflate
# either engine; v13 loses real game-state signal until a clean start-of-PA score is scraped for every row.
SITUATION_FEATS = ["balls", "strikes", "outs", "inning", "ahead_in_count"]
# what actually ships: the post-pitch count columns are dropped (see v12_load()); outs and inning are pre-pitch.
SAFE_SITUATION_FEATS = ["outs", "inning"]
ARM_FEATS       = ["pop_2b_sba", "maxeff_arm_2b_3b_sba", "exchange_2b_3b_sba"]


def v12_load() -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble the per-attempt table: leads + runner skill + pitch context + catcher arm."""
    df = load_attempts()
    df = df[df["result"].isin(["SB", "CS"])].copy()
    df["y"] = (df["result"] == "SB").astype(int)
    df["base_is_3b"] = (df["base"].astype(str) == "3B").astype(int)

    sssi = pd.read_csv(RAW / "DF_v7_SSSI.csv")
    keep = ["runner_id", "season"] + [c for c in PA_RUNNER_FEATS if c in sssi.columns]
    df = df.merge(sssi[keep].drop_duplicates(["runner_id", "season"]),
                  on=["runner_id", "season"], how="left")

    df = df.merge(load_context(), on="play_id", how="left")

    cls = df["pitch_code"].map(PITCH_CLASS).fillna("other")
    for c in ("fastball", "breaking", "offspeed"):
        df[f"pc_{c}"] = (cls == c).astype(int)
    # CAVEAT: balls/strikes come from the play-by-play feed POST-pitch, so strikes==3 is a
    # strikeout and balls==4 a walk (which is why 3-3 and 0-3 appear). That is consistent with
    # the rest of this model — it is descriptive and already uses the lead the runner actually
    # achieved — but it is NOT the count a coach sees when deciding to send him. Deriving the
    # pre-pitch count needs the per-pitch call, which the cache does not keep; a re-scrape
    # would be required. The block is worth ~+0.003 AUROC either way (see the ablation).
    df["ahead_in_count"] = pd.to_numeric(df["balls"], errors="coerce") - \
                           pd.to_numeric(df["strikes"], errors="coerce")

    pop_path = RAW / "poptime.csv"
    if pop_path.exists():
        pop = pd.read_csv(pop_path)[["catcher_id", "season"] + ARM_FEATS]
        df = df.merge(pop, on=["catcher_id", "season"], how="left")
    else:
        for c in ARM_FEATS:
            df[c] = np.nan

    df = df.reset_index(drop=True)
    return df, df["y"].values


def _splits(df, y, split: str, seed: int):
    """Yield (train_idx, val_idx) for a named validation regime.

    random  — StratifiedKFold. Optimistic: a runner's other attempts sit in the training set,
              and his season-level features are constant, so identity partly carries over.
    group   — GroupKFold on runner_id. No runner appears on both sides.
    forward — train on seasons < T, test on season T. The only regime that mirrors real use
              (predicting a season you have not seen) and the only one immune to the season
              aggregates in the feature set leaking backwards.
    """
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    if split == "random":
        yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y)
    elif split == "group":
        yield from GroupKFold(5).split(df, y, df["runner_id"].values)
    elif split == "forward":
        for T in sorted(df["season"].unique())[1:]:
            tr = np.where(df["season"].values < T)[0]
            va = np.where(df["season"].values == T)[0]
            if len(va) >= 200 and len(tr) >= 500:
                yield tr, va
    else:
        raise ValueError(split)


def v12_evaluate(df: pd.DataFrame, y: np.ndarray, feats: list[str], catcher_enc: bool,
             seed: int = 42, split: str = "random", return_pred: bool = False):
    """Pooled out-of-fold AUROC / AUPRC / Brier under a chosen validation regime. Catcher
    tendency, when used, is encoded with an inner fold so no row ever sees its own outcome
    (the v11 leak, fixed)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from xgboost import XGBClassifier

    feats = [f for f in feats if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    prior = float(y.mean())

    def xgb():
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                             eval_metric="logloss", verbosity=0, random_state=seed,
                             use_label_encoder=False)

    def enc_map(idx, sm=20.0):
        s = df.iloc[idx].groupby("catcher_id")["y"].agg(["sum", "count"])
        return (s["sum"] + prior * sm) / (s["count"] + sm)

    oof = np.full(len(df), np.nan)
    for tr, va in _splits(df, y, split, seed):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        if catcher_enc:
            inner_vals = np.full(len(tr), prior)
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
            for i_tr, i_va in inner.split(df.iloc[tr], y[tr]):
                m = enc_map(tr[i_tr])
                inner_vals[i_va] = df.iloc[tr[i_va]]["catcher_id"].map(m).fillna(prior).values
            Xtr["catcher_enc"] = inner_vals
            Xva["catcher_enc"] = df.iloc[va]["catcher_id"].map(enc_map(tr)).fillna(prior).values
        oof[va] = xgb().fit(Xtr.values, y[tr]).predict_proba(Xva.values)[:, 1]

    scored = ~np.isnan(oof)            # forward leaves the first season unscored
    yy, pp = y[scored], oof[scored]
    out = {"auroc": roc_auc_score(yy, pp), "auprc": average_precision_score(yy, pp),
           "brier": brier_score_loss(yy, pp), "n_scored": int(scored.sum())}
    return (out, oof) if return_pred else out


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency. The calculator states a probability, not a rank, so
    this is the check that matters for it — AUROC would not notice systematic over-confidence."""
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b + 1, "n": int(m.sum()),
                     "predicted": round(float(p[m].mean()), 4),
                     "observed": round(float(y[m].mean()), 4),
                     "gap": round(float(y[m].mean() - p[m].mean()), 4)})
    return pd.DataFrame(rows)


def run_v12():
    df, y = v12_load()
    matched = df["is_lhp"].notna().mean()
    print(f"v12 per-attempt table: {len(df)} attempts | base rate {y.mean():.3f} | "
          f"pitch context matched on {matched*100:.1f}% | "
          f"catcher arm on {df['pop_2b_sba'].notna().mean()*100:.1f}%")

    base = PA_LEAD_FEATS + ["base_is_3b"] + [c for c in PA_RUNNER_FEATS if c in df.columns]
    # SHIP_FEATS drops the post-pitch count (see the caveat in v12_load()): balls/strikes arrive
    # after the pitch, so they are not what a coach sees, and they are worth +0.0002. outs /
    # inning are pre-pitch and stay; score_diff is the end-of-PA score and is excluded (v16 QA).
    ship = base + BATTERY_FEATS + SAFE_SITUATION_FEATS + ARM_FEATS
    ladder = [
        ("v11 baseline — leads + runner skill",        base,                                    False),
        ("v11 + catcher tendency (nested OOF)",        base,                                    True),
        ("+ 1. battery & pitch type",                  base + BATTERY_FEATS,                    True),
        ("+ 2. count & situation",                     base + BATTERY_FEATS + SITUATION_FEATS,  True),
        ("+ 3. catcher arm (pop time)",                base + BATTERY_FEATS + SITUATION_FEATS + ARM_FEATS, True),
        ("v12 SHIPPED (post-pitch count dropped)",     ship,                                    True),
    ]

    # each block ALONE on top of the baseline — cumulative order can hide redundancy
    standalone = [
        ("alone: battery & pitch type", base + BATTERY_FEATS,   False),
        ("alone: count & situation",    base + SITUATION_FEATS, False),
        ("alone: catcher arm",          base + ARM_FEATS,       False),
    ]

    rows = []
    for name, feats, enc in ladder + standalone:
        m = v12_evaluate(df, y, feats, enc)
        rows.append({"model": name, "n_features": len([f for f in feats if f in df.columns]) + int(enc),
                     "auroc": round(m["auroc"], 4), "auprc": round(m["auprc"], 4),
                     "brier": round(m["brier"], 4)})
        print(f"  {name:44s} AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | Brier {m['brier']:.4f}")

    # ── validation regimes: the random split is optimistic; report the honest ones too ──
    print("\nvalidation regimes (shipped feature set):")
    vrows = []
    for split in ("random", "group", "forward"):
        m = v12_evaluate(df, y, ship, True, split=split)
        vrows.append({"split": split, "auroc": round(m["auroc"], 4), "auprc": round(m["auprc"], 4),
                      "brier": round(m["brier"], 4), "n_scored": m["n_scored"]})
        print(f"  {split:8s} AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | "
              f"Brier {m['brier']:.4f} | n={m['n_scored']}")
    pd.DataFrame(vrows).to_csv(res_path("DF_v12_Validation.csv"), index=False)

    # ── calibration: does a stated probability mean what it says? ──
    # Assessed under BOTH regimes on purpose. In-distribution the model is well calibrated;
    # the forward holdout drifts because the league itself is drifting (SB success is falling
    # season over season), so a model trained on the past predicts an easier environment than
    # the one it is scored in. That is a base-rate shift, not model over-confidence — isotonic
    # on pooled data would hide the cause rather than fix it, so no correction is applied.
    rels, roc_oof = [], {}
    for split in ("random", "forward"):
        _, oof = v12_evaluate(df, y, ship, True, split=split, return_pred=True)
        roc_oof[split] = oof
        r = reliability(y, oof); r.insert(0, "split", split); rels.append(r)
        print(f"\ncalibration ({split}): max |predicted − observed| = {r['gap'].abs().max():.3f}, "
              f"mean gap {r['gap'].mean():+.4f}")
    pd.concat(rels).to_csv(res_path("DF_v12_Calibration.csv"), index=False)

    rate = df.groupby("season")["y"].mean()
    print("league SB rate by season: " + " · ".join(f"{s} {v:.3f}" for s, v in rate.items()) +
          "  <- the environment is getting harder, which is what the forward gap reflects")

    out = pd.DataFrame(rows)
    out.to_csv(res_path("DF_v12_AUC.csv"), index=False)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        labels = ["v11\nbaseline", "+ catcher\ntendency", "+ battery\n& pitch",
                  "+ count &\nsituation", "+ catcher\narm"]
        vals = out["auroc"].tolist()[:5]
        colors = ["#9CA3AF", "#6B7280", "#2F6FB0", "#1D4ED8", "#10B981"]
        ax.bar(labels, vals, color=colors, width=0.58)
        ax.axhline(vals[0], color="#9CA3AF", ls="--", lw=1, zorder=0)
        ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
        ax.set_ylabel("CV AUROC (out-of-fold)")
        ax.set_title("v12 — what each added signal is worth, per attempt", fontsize=12)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10.5)
        plt.tight_layout(); plt.savefig(fig_path("Fig_v12_AUC.png"), dpi=160); plt.close()

        # the actual ROC curve for the shipped model, random AND forward — the gap between the two
        # curves IS the cost of not being allowed to see the future, drawn rather than just quoted
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(5.8, 5.4), dpi=150)
        for split, color in [("random", "#2F6FB0"), ("forward", "#C0392B")]:
            oof = roc_oof[split]; scored = ~np.isnan(oof)
            fpr, tpr, _ = roc_curve(y[scored], oof[scored])
            auroc = vrows[0]["auroc"] if split == "random" else vrows[2]["auroc"]
            ax.plot(fpr, tpr, lw=2.4, color=color, label=f"{split} (AUROC {auroc:.3f})")
        ax.plot([0, 1], [0, 1], lw=1.1, ls="--", color="#9AA0A6", label="no-skill (0.500)")
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.set_title("ROC — v12 shipped model, random vs forward split\n"
                     "the gap between the curves is the cost of testing on the future",
                     fontsize=10.5, fontweight="bold", color="#0C2340")
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
        fig.tight_layout(); fig.savefig(fig_path("Fig_v12_ROC.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass

    gain = out["auroc"].iloc[5] - out["auroc"].iloc[0]
    print(f"\nv12 vs v11 baseline: AUROC {out['auroc'].iloc[0]:.4f} -> {out['auroc'].iloc[5]:.4f} "
          f"({gain:+.4f})")
    return out


# ============================================================================
# v13 — THE DECISION MODEL: will he ATTEMPT at all?
# Shares PITCH_CLASS and reliability() with the success engine above (same file now).
# ============================================================================
# pc_fastball omitted as the reference category — the three dummies sum to 1 on 99.9% of rows, so
# including all three is the dummy-variable trap (see BATTERY_FEATS above).
# score_diff excluded (v16 QA): end-of-PA score, i.e. it can contain runs scored after this pitch. The opportunity
# table's balls/strikes/outs ARE pre-pitch (ingest.py opportunities records them before applying the event count).
V13_SITUATION = ["balls", "strikes", "outs", "inning", "is_lhp", "bat_side_r",
             "pc_breaking", "pc_offspeed"]
V13_PERSONNEL = ["sprint_speed_all", "runner_rate_prior", "seen_before"]
V13_FEATS     = V13_SITUATION + V13_PERSONNEL


def v13_load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(RAW / "Raw_Opportunities.csv.gz")
    df = df[(df["balls"] <= 3) & (df["strikes"] <= 2)]          # one malformed feed row in 517k

    # Season: the opportunities table has no date, but game_pk is monotonic in time and the
    # seasons occupy cleanly separated blocks. Learn the block edges from the games that DO
    # contain a Savant-tracked play_id, then assign EVERY game by range — joining on play_id
    # alone would silently drop the ~36% of games with no tracked attempt in them.
    att = load_attempts()[["play_id", "season"]]
    known = (df.merge(att, on="play_id", how="inner")
               .groupby("game_pk")["season"].first().reset_index())
    bounds = known.groupby("season")["game_pk"].agg(["min", "max"]).sort_index()
    seasons = bounds.index.to_numpy()
    edges = [(bounds["max"].iloc[i] + bounds["min"].iloc[i + 1]) / 2
             for i in range(len(bounds) - 1)]
    df["season"] = seasons[np.searchsorted(edges, df["game_pk"].values)]

    cls = df["pitch_code"].map(PITCH_CLASS).fillna("other")
    for c in ("fastball", "breaking", "offspeed"):
        df[f"pc_{c}"] = (cls == c).astype(int)

    sp = pd.read_csv(RAW / "sprint_speed.csv").rename(columns={"runner_id": "runner_1b"})
    df = df.merge(sp, on=["runner_1b", "season"], how="left")

    # NOTE: no catcher feature here. The opportunity feed carries no catcher id, and a
    # season-league mean would be constant within a season — a column that looks like a feature
    # but cannot inform a per-pitch decision. Attaching the real catcher needs the boxscore.

    # The "scouting report" prior: how often does THIS runner go? It must be built from PRIOR
    # SEASONS ONLY. A same-season leave-one-out rate still reads the season being predicted, which
    # inflated the random split to AUPRC 0.25 against 0.06 grouped — the same self-contamination
    # that produced the v11 catcher/pitcher bug and was caught again on Burst. Expanding by season
    # is causal and is what you would actually have on hand at gametime.
    league = float(df["attempt"].mean())
    per = (df.groupby(["runner_1b", "season"])["attempt"].agg(["sum", "count"])
             .sort_index().reset_index())
    per[["cum_s", "cum_n"]] = (per.groupby("runner_1b")[["sum", "count"]]
                                  .transform(lambda c: c.shift(1).cumsum()))
    per["runner_rate_prior"] = (per["cum_s"] / per["cum_n"]).fillna(league)
    df = df.merge(per[["runner_1b", "season", "runner_rate_prior"]],
                  on=["runner_1b", "season"], how="left")
    df["runner_rate_prior"] = df["runner_rate_prior"].fillna(league)
    df["seen_before"] = (df["runner_rate_prior"] != league).astype(int)

    df = df.reset_index(drop=True)
    return df, df["attempt"].values


def v13_evaluate(df, y, feats, split="random", seed=42, return_pred=False):
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from xgboost import XGBClassifier

    feats = [f for f in feats if f in df.columns]
    X = df[feats].apply(pd.to_numeric, errors="coerce").values

    def splits():
        if split == "random":
            yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y)
        elif split == "group":
            yield from GroupKFold(5).split(df, y, df["runner_1b"].values)
        elif split == "forward":
            for T in sorted(df["season"].unique())[1:]:
                tr = np.where(df["season"].values < T)[0]
                va = np.where(df["season"].values == T)[0]
                if len(va) >= 500 and len(tr) >= 2000:
                    yield tr, va

    oof = np.full(len(df), np.nan)
    for tr, va in splits():
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0,
                          eval_metric="logloss", verbosity=0, random_state=seed,
                          use_label_encoder=False)
        oof[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]

    ok = ~np.isnan(oof)
    yy, pp = y[ok], oof[ok]
    out = {"auprc": average_precision_score(yy, pp), "auroc": roc_auc_score(yy, pp),
           "brier": brier_score_loss(yy, pp), "base_rate": float(yy.mean()),
           "n_scored": int(ok.sum())}
    return (out, oof) if return_pred else out


def run_v13():
    df, y = v13_load()
    print(f"v13 DECISION MODEL (2023-2026): {len(df):,} opportunity pitches (runner on 1B, 2B empty) | "
          f"attempts {y.sum():,} | attempt rate {y.mean()*100:.2f}%")
    print(f"seasons: {sorted(df['season'].unique())} | distinct runners: {df['runner_1b'].nunique():,}")

    rows = []
    for name, feats in [("situation only", V13_SITUATION),
                        ("personnel only", V13_PERSONNEL),
                        ("v13 FULL (situation + personnel)", V13_FEATS)]:
        m = v13_evaluate(df, y, feats)
        rows.append({"model": name, "split": "random", **{k: round(v, 4) for k, v in m.items()}})
        print(f"  {name:34s} AUPRC {m['auprc']:.4f} (floor {m['base_rate']:.4f}) | "
              f"AUROC {m['auroc']:.4f} | Brier {m['brier']:.4f}")

    print("\nvalidation regimes (full model):")
    for split in ("random", "group", "forward"):
        m = v13_evaluate(df, y, V13_FEATS, split=split)
        rows.append({"model": "v13 FULL", "split": split, **{k: round(v, 4) for k, v in m.items()}})
        print(f"  {split:8s} AUPRC {m['auprc']:.4f} (floor {m['base_rate']:.4f}) | "
              f"AUROC {m['auroc']:.4f} | Brier {m['brier']:.4f} | n={m['n_scored']:,}")
    pd.DataFrame(rows).to_csv(res_path("DF_v13_Attempt.csv"), index=False)

    # ── the diagnostic this model exists for: the selection effect, without a model ──
    # SPEED BINS, DEFINED. Quintiles of the OPPORTUNITY population — every pitch with a runner on
    # 1st — so a runner who reaches base often carries more weight. They are NOT MLB-wide
    # percentiles and NOT equal numbers of runners, and a runner can fall in different bins in
    # different seasons because sprint speed is measured per season.
    q, edges = pd.qcut(df["sprint_speed_all"], 5, retbins=True, duplicates="drop",
                       labels=["slowest", "slow", "mid", "fast", "fastest"])
    tab = df.groupby(q).agg(low_ftps=("sprint_speed_all", "min"),
                            high_ftps=("sprint_speed_all", "max"),
                            mean_ftps=("sprint_speed_all", "mean"),
                            runner_seasons=("runner_1b", "nunique"),
                            opportunities=("attempt", "size"),
                            attempts=("attempt", "sum")).round(2)
    tab["attempt_pct"] = (100 * tab["attempts"] / tab["opportunities"]).round(2)
    tab["share_of_opps_pct"] = (100 * tab["opportunities"] / len(df)).round(1)
    print("\nSPEED BINS — quintiles of the opportunity population (pitch-weighted, NOT MLB percentile):")
    print(tab.to_string())
    tab.to_csv(res_path("DF_v13_SelectionEffect.csv"))

    pcts = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    pct = pd.DataFrame({"percentile": pcts,
                        "sprint_speed_ftps": [round(float(np.percentile(
                            df["sprint_speed_all"].dropna(), p)), 1) for p in pcts]})
    pct.to_csv(res_path("DF_v13_SpeedPercentiles.csv"), index=False)
    print("speed percentiles in this population (ft/s): " +
          " · ".join(f"p{r.percentile}={r.sprint_speed_ftps}" for r in pct.itertuples()))

    print("\nattempt rate by count:")
    ct = df.groupby(["balls", "strikes"])["attempt"].agg(["mean", "size"])
    ct = ct[ct["size"] >= 2000].sort_values("mean", ascending=False)
    print((ct.assign(attempt_pct=(ct["mean"] * 100).round(2))
             .drop(columns="mean").head(6)).to_string())

    _, oof = v13_evaluate(df, y, V13_FEATS, split="forward", return_pred=True)
    rel = reliability(y, oof)
    rel.to_csv(res_path("DF_v13_Calibration.csv"), index=False)
    print(f"\ncalibration (forward): max |predicted − observed| = {rel['gap'].abs().max():.4f}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        t = df.groupby(q)["attempt"].mean() * 100
        ax.bar([str(i) for i in t.index], t.values, color="#2F6FB0", width=0.6)
        ax.set_ylabel("attempt rate (%)"); ax.set_xlabel("runner sprint-speed quintile")
        ax.set_title("Runners self-select: who even tries to steal (2023–2026)", fontsize=12)
        for i, v in enumerate(t.values):
            ax.text(i, v + 0.03, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout(); plt.savefig(fig_path("Fig_v13_Attempt.png"), dpi=160); plt.close()
    except ImportError:
        pass
    return rows


# ════ 3. Powers et al. (2026): speed suppression, mediation, mixed effects; delivery time as a beta ═══════════
# Two read-only analyses on the SHIPPED v15 calculator rows (changes no shipped output).
#
#   A. Delivery time as a beta. The shipped 4-input logistic (fit on every 2023-26 calculator attempt) is FROZEN
#      and used as an offset; on the 2023 attempts that the CV pipeline timed (data/delivery/delivery_2023.csv, lift-off
#      -> release) we estimate one extra coefficient for delivery time. Also: an unfrozen refit, the total
#      effect without ground gained, a pitcher-average variant, and the high-confidence subset.
#   B. Speed suppression (Powers, Ramani, Hahn & Schaefer, arXiv 2601.15608). M0-M3 on identical rows,
#      speed -> gain mediation with a runner-clustered bootstrap, clustered SEs, runner/pitcher random
#      intercepts (statsmodels variational Bayes), and speed-band stability.
#
# Rows: exactly run_success_model()'s preprocessing (asserted to reproduce n and the shipped coefficients). Writes
# results/4-Studies/DF_add_speed_suppression.csv, DF_add_delivery_beta.csv, DF_add_meta.json and two figures.

POWERS_SPEED = {"v1 (Jan 2026)": (0.130, 0.036), "v2 (Aug 2026)": (0.18, 0.04)}   # SB-success speed, Table 1
SPECS = {"M0": ["sprint_speed"],
         "M1": ["sprint_speed", "lead_at_firstmove_ft", "pop_faced"],
         "M2": F4,
         "M3": ["sprint_speed", "gain_to_release_ft", "pop_faced"]}


def shipped_coefs() -> dict:
    sm_ = pd.read_csv(res_path("DF_success_model.csv"))
    return {r.term: float(r.coefficient) for r in sm_.itertuples() if r.term in ["intercept"] + F4}


def vif(df, cols) -> float:
    if len(cols) < 2:
        return 1.0
    X = df[cols].values
    return max(1 / (1 - LinearRegression().fit(np.delete(X, i, 1), X[:, i]).score(np.delete(X, i, 1), X[:, i]))
               for i in range(len(cols)))


def oof_auc(df, cols, seed=42) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(LogisticRegression(max_iter=5000), df[cols].values, df.y.values, cv=cv,
                          method="predict_proba")[:, 1]
    return roc_auc_score(df.y, p)


def logit(df, cols, offset=None, cluster=None):
    X = sm.add_constant(df[cols].values, has_constant="add")
    kw = dict(cov_type="cluster", cov_kwds={"groups": df[cluster].values}) if cluster else {}
    return sm.Logit(df.y.values, X, offset=offset).fit(disp=0, **kw)


# ───────────────────────────── B. speed suppression ─────────────────────────────
def suppression(at, n_boot=1000, seed=0):
    n = len(at)
    rows, fits = [], {}
    for name, cols in SPECS.items():
        assert len(at[cols].dropna()) == n, f"{name} must use identical rows"
        f, fc = logit(at, cols), logit(at, cols, cluster="runner_id")
        v = vif(at, cols); assert v < 10, f"{name} VIF {v:.1f}"
        fits[name] = f
        b, se, sec = f.params[1], f.bse[1], fc.bse[1]
        rows.append(dict(section="model", model=name, terms=" + ".join(cols), n=n, beta_speed=b, se=se,
                         ci_lo=ci(b, se)[0], ci_hi=ci(b, se)[1], se_cluster_runner=sec,
                         ci_lo_cluster=ci(b, sec)[0], ci_hi_cluster=ci(b, sec)[1],
                         auroc_oof=oof_auc(at, cols), max_vif=v,
                         **{f"coef_{c}": f.params[i + 1] for i, c in enumerate(cols)}))
        print(f"{name} n={n} b_speed {b:+.4f} (SE {se:.4f}, cluster {sec:.4f}) AUROC {rows[-1]['auroc_oof']:.4f} VIF {v:.2f}")
    print(f"rows identical across M0-M3: n={n} (asserted)")

    # speed -> gain (attempt level), and the decomposition
    A = sm.OLS(at.gain_to_release_ft, sm.add_constant(at[["sprint_speed", "lead_at_firstmove_ft", "pop_faced"]])).fit()
    A0 = sm.OLS(at.gain_to_release_ft, sm.add_constant(at[["sprint_speed"]])).fit()
    a, b_gain = A.params["sprint_speed"], fits["M2"].params[3]
    direct, total = fits["M2"].params[1], fits["M1"].params[1]
    point = dict(a=a, indirect=a * b_gain, direct=direct, total=total, share=-(a * b_gain) / direct)

    # runner-clustered bootstrap
    rng = np.random.default_rng(seed)
    groups = at.groupby("runner_id").indices; keys = np.array(list(groups))
    X1 = sm.add_constant(at[SPECS["M1"]].values); X2 = sm.add_constant(at[SPECS["M2"]].values)
    XA = sm.add_constant(at[["sprint_speed", "lead_at_firstmove_ft", "pop_faced"]].values)
    y, g = at.y.values, at.gain_to_release_ft.values
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        p1 = sm.Logit(y[idx], X1[idx]).fit(disp=0, start_params=fits["M1"].params).params
        p2 = sm.Logit(y[idx], X2[idx]).fit(disp=0, start_params=fits["M2"].params).params
        pa = np.linalg.lstsq(XA[idx], g[idx], rcond=None)[0]
        boot.append(dict(a=pa[1], indirect=pa[1] * p2[3], direct=p2[1], total=p1[1], share=-(pa[1] * p2[3]) / p2[1]))
    B = pd.DataFrame(boot)
    for k in ["a", "indirect", "direct", "total", "share"]:
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        rows.append(dict(section="mediation", model=k, n=n, estimate=point[k], ci_lo=lo, ci_hi=hi,
                         boot_se=B[k].std(), n_boot=n_boot))
    rows.append(dict(section="mediation", model="direct+indirect", n=n, estimate=direct + a * b_gain,
                     ci_lo=np.percentile(B.direct + B.indirect, 2.5), ci_hi=np.percentile(B.direct + B.indirect, 97.5)))
    rows.append(dict(section="mediation", model="a_unadjusted (gain~speed only)", n=n,
                     estimate=A0.params["sprint_speed"], ci_lo=A0.conf_int().loc["sprint_speed", 0],
                     ci_hi=A0.conf_int().loc["sprint_speed", 1]))
    rows.append(dict(section="mediation", model="corr(speed, gain) attempt level", n=n,
                     estimate=np.corrcoef(at.sprint_speed, at.gain_to_release_ft)[0, 1]))
    print(f"a={a:+.4f}  indirect={a*b_gain:+.4f}  direct={direct:+.4f}  total={total:+.4f}  "
          f"direct+indirect={direct + a*b_gain:+.4f}  share={point['share']:.1%}")

    # random intercepts (variational Bayes; the only GLMM already installed). Its fixed effects carry an
    # N(0, 2) prior, which on raw units crushes the -23 intercept and the 5.9-per-second pop slope — so fit on
    # z-scored inputs (prior then weak) and convert the speed slope back to per ft/s.
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    feats = sorted({c for cols in SPECS.values() for c in cols})
    mu, sd = at[feats].mean(), at[feats].std()
    d = at.assign(runner=at.runner_id.astype(str), pitcher=at.pitcher_id.astype(str))
    d[feats] = (at[feats] - mu) / sd
    for name, vc in [("M1", ["runner"]), ("M2", ["runner"]), ("M1", ["pitcher"]), ("M1", ["runner", "pitcher"]),
                     ("M2", ["runner", "pitcher"])]:
        form = "y ~ " + " + ".join(SPECS[name])
        m = BinomialBayesMixedGLM.from_formula(form, {v: f"0 + C({v})" for v in vc}, d).fit_vb()
        b, s = m.fe_mean[1] / sd["sprint_speed"], m.fe_sd[1] / sd["sprint_speed"]
        sds = {v: float(np.exp(m.vcp_mean[i])) for i, v in enumerate(vc)}
        rows.append(dict(section="random_effects", model=f"{name} + RE({'+'.join(vc)})", n=n, beta_speed=b, se=s,
                         ci_lo=ci(b, s)[0], ci_hi=ci(b, s)[1], **{f"re_sd_{k}": v for k, v in sds.items()}))
        print(f"{name} + RE({'+'.join(vc)}): b_speed {b:+.4f} (post SD {s:.4f}) RE SD {sds}")

    # speed-band stability (attempt-level speed quantiles)
    for k, lab in [(2, "half"), (5, "quintile")]:
        q = pd.qcut(at.sprint_speed, k, labels=False, duplicates="drop")
        for j in sorted(q.unique()):
            sub = at[q == j]
            for name in ["M1", "M2"]:
                f = logit(sub, SPECS[name], cluster="runner_id")
                rows.append(dict(section=f"speed_{lab}", model=f"{name} {lab} {j + 1}/{k}", n=len(sub),
                                 speed_lo=sub.sprint_speed.min(), speed_hi=sub.sprint_speed.max(),
                                 beta_speed=f.params[1], se_cluster_runner=f.bse[1],
                                 ci_lo=ci(f.params[1], f.bse[1])[0], ci_hi=ci(f.params[1], f.bse[1])[1]))
    out = pd.DataFrame(rows)
    out.to_csv(res_path("DF_add_speed_suppression.csv"), index=False)
    return out, B, fits


def verdict(out):
    m1 = out[(out.section == "model") & (out.model == "M1")].iloc[0]
    ind = out[(out.section == "mediation") & (out.model == "indirect")].iloc[0]
    supported = 0.05 <= m1.beta_speed <= 0.20 and ind.ci_hi < 0
    rejected = m1.beta_speed >= 0.28 or ind.ci_lo <= 0 <= ind.ci_hi
    return "SUPPORTED" if supported else "REJECTED" if rejected else "PARTIAL (neither criterion fully met)"


def fig_suppression(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = out[out.section == "model"].set_index("model")
    re_ = out[out.section == "random_effects"].set_index("model")
    labs = [("M0  speed only", m.loc["M0"]), ("M1  speed + first-move lead + pop\n(Powers-like: no ground gained)", m.loc["M1"]),
            ("M2  + ground gained (SHIPPED)", m.loc["M2"]), ("M3  speed + ground gained + pop", m.loc["M3"]),
            ("M1 + runner & pitcher random intercepts", re_.loc["M1 + RE(runner+pitcher)"]),
            ("M2 + runner & pitcher random intercepts", re_.loc["M2 + RE(runner+pitcher)"])]
    fig, ax = plt.subplots(figsize=(8.6, 5.5), dpi=160)
    cols = {"v1 (Jan 2026)": "#F59E0B", "v2 (Aug 2026)": "#A78BFA"}
    for k, (b, s) in POWERS_SPEED.items():
        ax.axvspan(b - 1.96 * s, b + 1.96 * s, color=cols[k], alpha=0.18, lw=0)
        ax.axvline(b, color=cols[k], lw=1.4, ls="--", label=f"Powers et al. {k}: {b:+.3f} (SE {s:.3f}); shaded = 95% interval")
    for i, (lab, r) in enumerate(labs):
        yv = len(labs) - 1 - i
        lo, hi = (r.ci_lo_cluster, r.ci_hi_cluster) if "se_cluster_runner" in r and pd.notna(r.get("ci_lo_cluster")) else (r.ci_lo, r.ci_hi)
        c = "#0C2340" if "SHIPPED" in lab else "#2F6FB0" if i < 4 else "#6B7280"
        ax.errorbar(r.beta_speed, yv, xerr=[[r.beta_speed - lo], [hi - r.beta_speed]], fmt="o", color=c, capsize=4, ms=7)
        ax.text(hi + 0.012, yv, f"{r.beta_speed:+.3f}", va="center", fontsize=9, color=c)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([l for l, _ in labs][::-1], fontsize=8.5)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("sprint-speed coefficient (log-odds of SAFE per +1 ft/s)\nbars = 95% CI (runner-clustered; random-intercept rows: posterior ±1.96 SD)")
    ax.set_title(f"Does leaving out ground gained shrink the speed effect?\nThis project's {int(m.loc['M0'].n):,} attempts (2023-26) vs Powers et al.",
                 fontsize=11, fontweight="bold", color="#0C2340")
    ax.set_xlim(-0.02, 0.44)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    fig.tight_layout(); fig.savefig(fig_path("Fig_add_speed_suppression.png")); plt.close(fig)


# ───────────────────────────── A. delivery time as a beta ─────────────────────────────
def delivery_beta(at):
    cvd = load_delivery(2023)
    cvd = cvd[cvd.qa == "PASS"][["play_id", "delivery_s", "feed", "confidence"]]
    d = at.merge(cvd, on="play_id", how="inner").reset_index(drop=True)
    pmean = cvd.merge(at[["play_id", "pitcher_id"]], on="play_id").groupby("pitcher_id").delivery_s.agg(["mean", "count"])
    d["pitcher_delivery"] = d.pitcher_id.map(pmean["mean"].where(pmean["count"] >= 3))
    c = shipped_coefs()
    d["offset"] = c["intercept"] + sum(c[f] * d[f] for f in F4)
    d["delivery_10"] = d.delivery_s * 10                     # coefficient per 0.1 s
    d["pitcher_delivery_10"] = d.pitcher_delivery * 10
    print(f"\nA. delivery rows: {len(d)} of {len(cvd)} CV-timed 2023 attempts are on the shipped rows "
          f"({(d.confidence == 'high').sum()} high-confidence)")

    rows = []
    base = logit(d, [], offset=d.offset.values)               # frozen model, intercept-only recalibration
    def add(name, f, col_i, sub, note, cluster_f=None):
        b, se = f.params[col_i], f.bse[col_i]
        sec = cluster_f.bse[col_i] if cluster_f is not None else np.nan
        rows.append(dict(model=name, n=len(sub), beta_per_0p1s=b, se=se, se_cluster_pitcher=sec,
                         ci_lo=b - 1.96 * (sec if cluster_f is not None else se),
                         ci_hi=b + 1.96 * (sec if cluster_f is not None else se),
                         odds_mult_per_0p1s=np.exp(b), p=f.pvalues[col_i] if cluster_f is None else cluster_f.pvalues[col_i],
                         note=note))
        print(f"  {name:48s} n={len(sub)} beta/0.1s {b:+.4f} (SE {se:.4f}, pitcher-cluster {sec:.4f}) OR {np.exp(b):.3f}")

    # D1 frozen 10k weights + delivery
    f1 = logit(d, ["delivery_10"], offset=d.offset.values)
    f1c = logit(d, ["delivery_10"], offset=d.offset.values, cluster="pitcher_id")
    add("D1 frozen 10k weights + delivery", f1, 1, d, "shipped coefficients fixed as offset; fits intercept + delivery", f1c)
    lr = 2 * (f1.llf - base.llf)
    from scipy.stats import chi2
    lrp = chi2.sf(lr, 1)
    # D2 unfrozen refit on the subset
    f2 = logit(d, F4 + ["delivery_10"]); f2c = logit(d, F4 + ["delivery_10"], cluster="pitcher_id")
    add("D2 refit all 5 on 2023 subset", f2, 5, d, "all coefficients re-estimated on these rows", f2c)
    # D3 total effect: drop ground gained (delivery buys ground)
    cols3 = ["sprint_speed", "lead_at_firstmove_ft", "pop_faced", "delivery_10"]
    f3 = logit(d, cols3); f3c = logit(d, cols3, cluster="pitcher_id")
    add("D3 without ground gained (total effect)", f3, 4, d, "delivery may act THROUGH ground gained", f3c)
    # D4 pitcher-average delivery (less measurement noise), frozen
    d4 = d.dropna(subset=["pitcher_delivery_10"]).reset_index(drop=True)
    f4 = logit(d4, ["pitcher_delivery_10"], offset=d4.offset.values)
    f4c = logit(d4, ["pitcher_delivery_10"], offset=d4.offset.values, cluster="pitcher_id")
    add("D4 frozen + pitcher-average delivery (>=3 timed)", f4, 1, d4, "averages out per-clip CV error", f4c)
    # D5 high-confidence rows only, frozen
    d5 = d[d.confidence == "high"].reset_index(drop=True)
    f5 = logit(d5, ["delivery_10"], offset=d5.offset.values)
    f5c = logit(d5, ["delivery_10"], offset=d5.offset.values, cluster="pitcher_id")
    add("D5 frozen + delivery, high-confidence only", f5, 1, d5, "inside the CV gold-verified envelope", f5c)

    # mediation: does a slower delivery buy ground?
    G = sm.OLS(d.gain_to_release_ft, sm.add_constant(d[["delivery_10", "sprint_speed", "lead_at_firstmove_ft", "pop_faced"]])).fit(
        cov_type="cluster", cov_kwds={"groups": d.pitcher_id.values})
    a_d = G.params["delivery_10"]
    # discrimination: frozen model as-is vs frozen + delivery (5-fold OOF on the 2 new parameters)
    auc_frozen = roc_auc_score(d.y, d.offset)
    oof = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(d, d.y):
        ft = sm.Logit(d.y.values[tr], sm.add_constant(d.delivery_10.values[tr]), offset=d.offset.values[tr]).fit(disp=0)
        oof[te] = d.offset.values[te] + ft.params[0] + ft.params[1] * d.delivery_10.values[te]
    auc_plus = roc_auc_score(d.y, oof)
    ext = dict(n=len(d), lr_chi2=lr, lr_p=lrp, auc_frozen=auc_frozen, auc_frozen_plus_delivery_oof=auc_plus,
               auc_refit5_oof=oof_auc(d, F4 + ["delivery_10"]), auc_refit4_oof=oof_auc(d, F4),
               gain_per_0p1s=a_d, gain_per_0p1s_se=G.bse["delivery_10"],
               corr_delivery_gain=np.corrcoef(d.delivery_s, d.gain_to_release_ft)[0, 1],
               corr_delivery_speed=np.corrcoef(d.delivery_s, d.sprint_speed)[0, 1],
               mean_delivery_SB=d[d.y == 1].delivery_s.mean(), mean_delivery_CS=d[d.y == 0].delivery_s.mean(),
               frozen_intercept_shift=base.params[0], success_rate=d.y.mean(),
               vif_D2=vif(d, F4 + ["delivery_10"]))
    assert ext["vif_D2"] < 10
    print(f"  LR test (frozen + delivery vs frozen): chi2 {lr:.2f}, p {lrp:.2g} | AUROC frozen {auc_frozen:.4f} -> "
          f"+delivery OOF {auc_plus:.4f} | refit 4 {ext['auc_refit4_oof']:.4f} -> 5 {ext['auc_refit5_oof']:.4f}")
    print(f"  ground gained per +0.1 s delivery: {a_d:+.3f} ft (SE {G.bse['delivery_10']:.3f}); "
          f"corr(delivery, gain) {ext['corr_delivery_gain']:+.3f}; frozen-intercept shift {base.params[0]:+.3f}")
    out = pd.DataFrame(rows)
    pd.concat([out, pd.DataFrame([dict(model="_summary", **ext)])]).to_csv(res_path("DF_add_delivery_beta.csv"), index=False)
    return out, ext, d


def fig_delivery(d, out, ext):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=160)
    # left: observed safe rate vs frozen-model expectation by delivery decile
    q = pd.qcut(d.delivery_s, 8, labels=False)
    g = d.assign(q=q, pexp=1 / (1 + np.exp(-(d.offset + ext["frozen_intercept_shift"])))).groupby("q")
    x = g.delivery_s.mean(); obs = g.y.mean(); exp_ = g.pexp.mean(); nq = g.size()
    se = np.sqrt(obs * (1 - obs) / nq)
    ax = axs[0]
    ax.errorbar(x, obs, yerr=1.96 * se, fmt="o-", color="#0C2340", capsize=3, label="observed safe rate (95% CI)")
    ax.plot(x, exp_, "s--", color="#10B981", label="frozen 10k model's expectation")
    ax.set_xlabel("pitcher delivery time, lift-off -> release (s), 8 equal-count bins")
    ax.set_ylabel("share of attempts safe"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("Slower deliveries are stolen on more —\nbut the model already expects it", fontsize=10.5, fontweight="bold", color="#0C2340")
    # right: coefficients
    ax = axs[1]
    o = out.iloc[::-1]
    for i, r in enumerate(o.itertuples()):
        ax.errorbar(r.beta_per_0p1s, i, xerr=[[r.beta_per_0p1s - r.ci_lo], [r.ci_hi - r.beta_per_0p1s]], fmt="o",
                    color="#2F6FB0" if "D1" not in r.model else "#0C2340", capsize=4)
        ax.text(r.ci_hi + 0.01, i, f"{r.beta_per_0p1s:+.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([m.split(" ", 1)[1] if False else m for m in o.model], fontsize=7.5)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("delivery-time coefficient (log-odds of SAFE per +0.1 s)\n95% CI, pitcher-clustered")
    ax.set_title("Delivery-time beta, five ways", fontsize=10.5, fontweight="bold", color="#0C2340")
    fig.tight_layout(); fig.savefig(fig_path("Fig_add_delivery_beta.png")); plt.close(fig)


def run_powers(n_boot=1000):
    at = shipped_rows()
    c = shipped_coefs()
    lr = LogisticRegression(max_iter=5000).fit(at[F4].values, at.y.values)
    rep = dict(zip(F4, lr.coef_[0]))
    assert all(abs(rep[f] - c[f]) < 5e-4 for f in F4), f"shipped coefficients not reproduced: {rep}"
    print(f"shipped rows reproduced: n={len(at)}, sklearn coefs {', '.join(f'{k} {v:+.4f}' for k, v in rep.items())}")
    sup, B, fits = suppression(at, n_boot)
    print("verdict:", verdict(sup))
    fig_suppression(sup)
    dl, ext, d = delivery_beta(at)
    fig_delivery(d, dl, ext)
    json.dump({"verdict": verdict(sup), "shipped_reproduced": {k: round(v, 5) for k, v in rep.items()},
               "M2_mle": {k: float(fits['M2'].params[i + 1]) for i, k in enumerate(F4)}, "delivery": ext},
              open(res_path("DF_add_meta.json"), "w"), indent=1, default=float)


# ════ 4. Pre-pitch: does the 2023 CV delivery profile help a model before the pitch? ══════════════════════════
# Does a pitcher's CV-timed delivery profile help a PRE-PITCH steal model?
#
# Before the pitch you know the runner's speed, his lead at first move and the catcher's pop time, but not the
# ground he will gain. §3 showed delivery time works only THROUGH ground gained, so its
# one remaining use is as a pre-pitch stand-in for it. This tests that, out of time:
#
#   profiles   built from 2023 only: pitcher's mean CV delivery time (>=3 timed clips, delivery_2023.csv),
#              his shrunk SB-success rate allowed, and his shrunk mean ground gained allowed (Statcast, no CV)
#   test rows  the shipped calculator's 2024-26 attempts against pitchers who have all three profiles
#   models     P0 speed + first-move lead + pop                        (pre-pitch base)
#              P1 P0 + delivery profile                                (the CV question)
#              P2 P0 + SB% allowed                                     (control: "who the pitcher is")
#              P3 P0 + ground-gained-allowed profile                   (control: same idea, no CV needed)
#              P4 P0 + all three
#              CEIL P0 + the attempt's ACTUAL ground gained            (post-pitch ceiling = shipped inputs)
# Reports out-of-fold AUROC (5-fold, as shipped), paired pitcher-clustered bootstrap CIs for each AUROC gain over
# P0, and pitcher-clustered coefficients. Secondary: the same on 2023 rows with leave-one-out profiles.

PREPITCH_BASE = ["sprint_speed", "lead_at_firstmove_ft", "pop_faced"]   # known before the pitch
MODELS = {"P0 pre-pitch base": PREPITCH_BASE,
          "P1 + delivery profile (CV)": PREPITCH_BASE + ["deliv"],
          "P2 + SB% allowed": PREPITCH_BASE + ["sb_allowed"],
          "P3 + ground-gained-allowed profile": PREPITCH_BASE + ["gain_allowed"],
          "P4 + all three": PREPITCH_BASE + ["deliv", "sb_allowed", "gain_allowed"],
          "CEIL + actual ground gained (post-pitch)": PREPITCH_BASE + ["gain_to_release_ft"]}
K_SB, K_GAIN, MIN_TIMED = 20, 10, 3


def profiles(src: pd.DataFrame, cv: pd.DataFrame, loo: bool):
    """Pitcher-level profiles from `src` attempts and `cv` timed clips. With loo=True each row's own attempt is
    left out of its pitcher's profile (for scoring the same season the profile is built from)."""
    league_sb, league_gain = src.y.mean(), src.gain_to_release_ft.mean()
    g = src.groupby("pitcher_id")
    sb_sum, n, gain_sum = g.y.transform("sum"), g.y.transform("count"), g.gain_to_release_ft.transform("sum")
    c = cv.groupby("pitcher_id").delivery_s
    if not loo:
        p = pd.DataFrame({"sb_n": g.y.count(), "sb_sum": g.y.sum(), "gain_sum": g.gain_to_release_ft.sum()})
        p["sb_allowed"] = (p.sb_sum + K_SB * league_sb) / (p.sb_n + K_SB)
        p["gain_allowed"] = (p.gain_sum + K_GAIN * league_gain) / (p.sb_n + K_GAIN)
        p = p.join(pd.DataFrame({"deliv": c.mean(), "n_timed": c.count()}), how="left")
        return p.loc[p.n_timed >= MIN_TIMED, ["deliv", "sb_allowed", "gain_allowed"]]
    out = src.copy()
    out["sb_allowed"] = (sb_sum - src.y + K_SB * league_sb) / (n - 1 + K_SB)
    out["gain_allowed"] = (gain_sum - src.gain_to_release_ft + K_GAIN * league_gain) / (n - 1 + K_GAIN)
    cs, cn = cv.groupby("pitcher_id").delivery_s.sum(), cv.groupby("pitcher_id").delivery_s.count()
    own = out.play_id.map(cv.set_index("play_id").delivery_s)            # this attempt's own timed clip, if any
    tot, cnt = out.pitcher_id.map(cs), out.pitcher_id.map(cn)
    tot, cnt = tot - own.fillna(0), cnt - own.notna()
    out["deliv"] = (tot / cnt).where(cnt >= MIN_TIMED)
    return out


def prepitch_evaluate(d: pd.DataFrame, tag: str, n_boot=1000, seed=0):
    cols_all = sorted({c for v in MODELS.values() for c in v})
    d = d.dropna(subset=cols_all).reset_index(drop=True)
    y = d.y.values
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = {m: cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), d[c].values, y, cv=cv, method="predict_proba")[:, 1]
           for m, c in MODELS.items()}
    auc = {m: roc_auc_score(y, p) for m, p in oof.items()}
    # paired, pitcher-clustered bootstrap of each AUROC gain over P0 (same resample for every model)
    rng, grp = np.random.default_rng(seed), d.groupby("pitcher_id").indices
    keys = np.array(list(grp)); base = "P0 pre-pitch base"
    gains = {m: [] for m in MODELS if m != base}
    for _ in range(n_boot):
        idx = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        if y[idx].min() == y[idx].max():
            continue
        a0 = roc_auc_score(y[idx], oof[base][idx])
        for m in gains:
            gains[m].append(roc_auc_score(y[idx], oof[m][idx]) - a0)
    ceil = auc["CEIL + actual ground gained (post-pitch)"] - auc[base]
    rows = []
    for m, cols in MODELS.items():
        f = logit(d, cols, cluster="pitcher_id")
        v = vif(d, cols); assert v < 10, f"{m} VIF {v:.1f}"
        new = cols[len(PREPITCH_BASE):]
        g = np.array(gains.get(m, [0.0]))
        rows.append(dict(sample=tag, model=m, n=len(d), n_pitchers=d.pitcher_id.nunique(), auroc_oof=auc[m],
                         auroc_gain=auc[m] - auc[base], gain_ci_lo=np.percentile(g, 2.5), gain_ci_hi=np.percentile(g, 97.5),
                         share_of_ceiling=(auc[m] - auc[base]) / ceil if ceil > 0 else np.nan, max_vif=v,
                         new_terms="; ".join(f"{c} {f.params[1 + cols.index(c)]:+.3f} (SE {f.bse[1 + cols.index(c)]:.3f}, "
                                             f"p {f.pvalues[1 + cols.index(c)]:.2g})" for c in new)))
        print(f"[{tag}] {m:42s} n={len(d)} AUROC {auc[m]:.4f}  gain {auc[m]-auc[base]:+.4f} "
              f"[{rows[-1]['gain_ci_lo']:+.4f}, {rows[-1]['gain_ci_hi']:+.4f}]  {rows[-1]['new_terms']}")
    sd = d[["deliv", "sb_allowed", "gain_allowed"]].corr().round(2)
    print(f"[{tag}] profile correlations:\n{sd.to_string()}")
    return rows


def run_prepitch():
    at = shipped_rows()
    cv = load_delivery(2023)
    cv = cv[cv.qa == "PASS"][["play_id", "pitcher_id", "delivery_s"]]
    a23 = at[at.season == 2023]
    prof = profiles(a23, cv, loo=False)
    fwd = at[at.season >= 2024].join(prof, on="pitcher_id", how="inner")
    print(f"profiles from 2023: {len(prof)} pitchers with >= {MIN_TIMED} timed clips; "
          f"forward rows 2024-26 against them: {len(fwd)} of {(at.season >= 2024).sum()}")
    rows = prepitch_evaluate(fwd, "forward 2024-26")
    rows += prepitch_evaluate(profiles(a23, cv, loo=True), "2023 leave-one-out")
    pd.DataFrame(rows).to_csv(res_path("DF_add_prepitch.csv"), index=False)
    print(f"wrote {res_path('DF_add_prepitch.csv')}")


# ════ 5. Pre-pitch: which input moves P(safe) the most? ═══════════════════════════════════════════════════════
# Which pre-pitch input moves P(safe) the most? Speed-only baseline, then each input on its own
# and together, on ONE fixed set of rows (§4's forward sample: 2024-26 attempts against pitchers
# with 2023 profiles), so every number is comparable.
#
# Inputs: sprint speed, lead at first move, catcher pop time, and the three 2023 pitcher profiles (CV delivery
# time, shrunk SB-success rate allowed, shrunk ground gained allowed). Reference only: the attempt's ACTUAL
# ground gained (post-pitch, so not a pre-pitch input).
#
#   1. alone        each input by itself: odds ratio per unit and per SD, out-of-fold AUROC
#   2. + speed      speed-only baseline plus one input: AUROC gain over baseline (paired, pitcher-clustered CI)
#   3. all six      odds ratio per SD, and the 10th->90th-percentile swing in P(safe) with the others at their
#                   means (1,000 pitcher-clustered bootstrap refits); drop-one AUROC loss = unique contribution

SIX = {"sprint_speed": ("Sprint speed", "ft/s"), "lead_at_firstmove_ft": ("Lead at first move", "ft"),
       "pop_faced": ("Catcher pop time", "s"), "deliv": ("Pitcher delivery profile (CV)", "s"),
       "sb_allowed": ("Pitcher SB% allowed", "share"), "gain_allowed": ("Pitcher ground-allowed profile", "ft")}
REF = ("gain_to_release_ft", "Actual ground gained (post-pitch, reference)", "ft")


def rows_fwd() -> pd.DataFrame:
    at = shipped_rows()
    cv = load_delivery(2023)
    cv = cv[cv.qa == "PASS"][["play_id", "pitcher_id", "delivery_s"]]
    prof = profiles(at[at.season == 2023], cv, loo=False)
    d = at[at.season >= 2024].join(prof, on="pitcher_id", how="inner")
    return d.dropna(subset=list(SIX) + [REF[0]]).reset_index(drop=True)


def run_needle(n_boot=1000, seed=0):
    d = rows_fwd(); y = d.y.values; n = len(d)
    sd = d[list(SIX) + [REF[0]]].std()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = lambda cols: cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), d[cols].values, y, cv=cv,
                                         method="predict_proba")[:, 1]
    print(f"rows: {n} attempts (2024-26), {d.pitcher_id.nunique()} pitchers, success rate {y.mean():.3f}")

    # every set of OOF predictions we compare, computed once
    allc = list(SIX)
    P = {"speed": oof(["sprint_speed"]), "all6": oof(allc)}
    for c in list(SIX)[1:] + [REF[0]]:
        P[f"speed+{c}"] = oof(["sprint_speed", c])
    for c in list(SIX) + [REF[0]]:
        P[f"alone:{c}"] = oof([c])
    for c in SIX:
        P[f"drop:{c}"] = oof([x for x in allc if x != c])
    auc = {k: roc_auc_score(y, v) for k, v in P.items()}

    # pitcher-clustered bootstrap: paired AUROC differences + refit of the all-six model for swings
    rng, grp = np.random.default_rng(seed), d.groupby("pitcher_id").indices
    keys = np.array(list(grp))
    X6 = sm.add_constant(d[allc].values); full = sm.Logit(y, X6).fit(disp=0)
    q10, q90, mu = d[allc].quantile(0.10), d[allc].quantile(0.90), d[allc].mean()

    def swings(params):
        out = {}
        for i, c in enumerate(allc):
            base = params[0] + sum(params[j + 1] * mu[x] for j, x in enumerate(allc) if x != c)
            p = lambda v: 1 / (1 + np.exp(-(base + params[i + 1] * v)))
            out[c] = 100 * (p(q90[c]) - p(q10[c]))
        return out

    diffs = {k: [] for k in P if k != "speed"}; sw = []
    for _ in range(n_boot):
        idx = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        a_speed, a_all = roc_auc_score(y[idx], P["speed"][idx]), roc_auc_score(y[idx], P["all6"][idx])
        for k in diffs:
            a = roc_auc_score(y[idx], P[k][idx])
            diffs[k].append(a - (a_all if k.startswith("drop:") else a_speed))
        sw.append(swings(sm.Logit(y[idx], X6[idx]).fit(disp=0, start_params=full.params).params))
    ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    SW, sw_pt = pd.DataFrame(sw), swings(full.params)
    fc = sm.Logit(y, X6).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.pitcher_id.values})

    rows = []
    for c in list(SIX) + [REF[0]]:
        name, unit = SIX[c] if c in SIX else REF[1:]
        f1 = sm.Logit(y, sm.add_constant(d[[c]].values)).fit(disp=0, cov_type="cluster",
                                                              cov_kwds={"groups": d.pitcher_id.values})
        r = dict(input=name, unit=unit, sd=sd[c], p10=d[c].quantile(.1), p90=d[c].quantile(.9),
                 alone_or_per_unit=np.exp(f1.params[1]), alone_or_per_sd=np.exp(f1.params[1] * sd[c]),
                 alone_auroc=auc[f"alone:{c}"])
        if c == "sprint_speed":
            r.update(with_speed_auroc=auc["speed"], with_speed_gain=0.0)
        else:
            k = f"speed+{c}"; lo, hi = ci(diffs[k])
            r.update(with_speed_auroc=auc[k], with_speed_gain=auc[k] - auc["speed"], with_speed_gain_lo=lo, with_speed_gain_hi=hi)
        if c in SIX:
            i = allc.index(c) + 1; b, se = fc.params[i], fc.bse[i]
            lo, hi = ci(diffs[f"drop:{c}"]); slo, shi = ci(SW[c])
            r.update(all6_or_per_sd=np.exp(b * sd[c]), all6_or_per_sd_lo=np.exp((b - 1.96 * se) * sd[c]),
                     all6_or_per_sd_hi=np.exp((b + 1.96 * se) * sd[c]), all6_p=fc.pvalues[i],
                     all6_swing_pp=sw_pt[c], all6_swing_lo=slo, all6_swing_hi=shi,
                     drop_one_loss=auc[f"drop:{c}"] - auc["all6"], drop_one_lo=lo, drop_one_hi=hi)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs = {}
    meta = pd.DataFrame([dict(input="_summary", n=n, pitchers=d.pitcher_id.nunique(), success_rate=y.mean(),
                              auroc_speed_only=auc["speed"], auroc_all6=auc["all6"],
                              auroc_all6_gain=auc["all6"] - auc["speed"], auroc_all6_gain_lo=ci(diffs["all6"])[0],
                              auroc_all6_gain_hi=ci(diffs["all6"])[1], vif_all6=vif(d, allc))])
    assert meta.vif_all6.iloc[0] < 10
    pd.concat([out, meta]).to_csv(res_path("DF_add_needle.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out[["input", "alone_or_per_sd", "alone_auroc", "with_speed_gain", "all6_or_per_sd", "all6_swing_pp",
               "drop_one_loss"]].round(4).to_string())
    print(meta.round(4).to_string())
    needle_fig(out, meta.iloc[0])


def needle_fig(out, meta):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, GREY, INK, MUTED, SURF = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=170, facecolor=SURF, layout="constrained")
    # left: AUROC gain when added to the speed-only baseline
    o = out[out.input != "Sprint speed"].sort_values("with_speed_gain")
    ax = axs[0]; ax.set_facecolor(SURF)
    for i, r in enumerate(o.itertuples()):
        ref = "reference" in r.input
        ax.barh(i, r.with_speed_gain, height=0.55, color=GREY if ref else BLUE)
        ax.errorbar(r.with_speed_gain, i, xerr=[[r.with_speed_gain - r.with_speed_gain_lo], [r.with_speed_gain_hi - r.with_speed_gain]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.text(max(r.with_speed_gain_hi, 0) + 0.004, i, f"{r.with_speed_gain:+.3f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(o))); ax.set_yticklabels(o.input, fontsize=8.5)
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlim(-0.02, 0.26)
    ax.set_xlabel(f"AUROC gain over speed only (baseline AUROC {meta.auroc_speed_only:.3f})\nbars: 95% CI, pitcher-clustered bootstrap")
    ax.set_title("Added to speed, one at a time", fontsize=10.5, fontweight="bold", color=INK, loc="left")
    # right: swing in P(safe) 10th -> 90th percentile inside the all-six model
    o = out[out.input.str.contains("reference") == False].assign(a=lambda t: t.all6_swing_pp.abs()).sort_values("a")
    ax = axs[1]; ax.set_facecolor(SURF)
    for i, r in enumerate(o.itertuples()):
        ax.barh(i, r.all6_swing_pp, height=0.55, color=BLUE)
        ax.errorbar(r.all6_swing_pp, i, xerr=[[r.all6_swing_pp - r.all6_swing_lo], [r.all6_swing_hi - r.all6_swing_pp]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.text(max(r.all6_swing_hi, 0) + 0.4, i, f"{r.all6_swing_pp:+.1f} pts", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([f"{r.input}\n{r.p10:.2f} → {r.p90:.2f} {r.unit}" for r in o.itertuples()], fontsize=7.8)
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlim(min(-2, o.all6_swing_lo.min() - 1), o.all6_swing_hi.max() + 6)
    ax.set_xlabel("change in P(safe), percentage points, 10th → 90th percentile\nothers at average; bars: 95% CI (1,000 pitcher-clustered refits)")
    ax.set_title("All six together: what moves P(safe)", fontsize=10.5, fontweight="bold", color=INK, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False); a.grid(axis="x", color="#ecebe7", lw=0.6); a.set_axisbelow(True)
    fig.suptitle(f"{int(meta.n):,} attempts 2024-26 against {int(meta.pitchers)} pitchers with 2023 profiles. "
             "Grey = post-pitch reference, not available before the pitch.", fontsize=8, color=MUTED, x=0.01, ha="left")
    fig.savefig(fig_path("Fig_add_needle.png"), facecolor=SURF); plt.close(fig)


# ════ 6. Pre-pitch with game context; pitch type thrown x game state ══════════════════════════════════════════
# Two follow-ups on the pre-pitch question and the pitch thrown.
#
# PART 1  Pre-pitch test with game context. Adds pitcher hand, batter side, outs and inning
#         (all known before the pitch; MLB-feed context; score_diff dropped as leaky) to §5's six inputs on the
#         same 2024-26 forward rows. Key question: does the CV delivery profile still add once context is in?
#         Balls/strikes are NOT used: the feed records the count AFTER the pitch (3-strike rows are 28% caught
#         vs 17-20% otherwise). Outs were checked: no 3-out rows, and CS rows do not carry the extra out.
# PART 2  Post-pitch segmentation. On the shipped calculator's attempts: do runners do better on certain
#         pitch types (fastball / breaking / offspeed, the pitch thrown on the attempt), and does that differ by
#         game state (outs, inning, pitcher hand, batter side, base)? Each cell reports the observed safe
#         rate AND observed minus what the shipped 4-input model expects, so a pitch type that is stolen on more
#         only because runners gain more ground on it does not look like a separate effect.
# Writes results/4-Studies/DF_add_context_prepitch.csv, DF_add_pitchtype*.csv and figures/4-Studies/Fig_add_pitchtype.png.

# score_diff is NOT used: the feed's play.result score is AFTER the plate appearance, so it can include runs
# scored after the steal attempt (QA, verified on a live game feed). Outs/inning/hands are pre-pitch.
CTX = ["is_lhp", "bat_side_r", "outs", "inning"]
PROF = ["deliv", "sb_allowed", "gain_allowed"]
CTX_LABEL = {"sprint_speed": "Sprint speed", "lead_at_firstmove_ft": "Lead at first move", "pop_faced": "Catcher pop time",
         "deliv": "Pitcher delivery profile (CV)", "sb_allowed": "Pitcher SB% allowed",
         "gain_allowed": "Pitcher ground-allowed profile", "is_lhp": "Left-handed pitcher",
         "bat_side_r": "Right-handed batter", "outs": "Outs", "inning": "Inning", "score_diff": "Score diff (batting team)"}
PCOL = {"fastball": "#2a78d6", "breaking": "#eb6834", "offspeed": "#1baf7a"}
PMARK = {"fastball": "o", "breaking": "s", "offspeed": "D"}


def pitch_context() -> pd.DataFrame:
    c = load_context()
    c["pitch_class"] = c.pitch_code.map(PITCH_CLASS)
    return c[["play_id", "pitch_class"] + CTX]


# ───────────────────────────── part 1 ─────────────────────────────
def context_part1(n_boot=1000, seed=0):
    d = rows_fwd().merge(pitch_context(), on="play_id", how="left").dropna(subset=CTX).reset_index(drop=True)
    y = d.y.values
    specs = {"Q0 base: speed + lead + pop": PREPITCH_BASE,
             "Q1 base + context": PREPITCH_BASE + CTX,
             "Q2 base + context + delivery profile": PREPITCH_BASE + CTX + ["deliv"],
             "Q3 base + context + Statcast pitcher profiles": PREPITCH_BASE + CTX + ["sb_allowed", "gain_allowed"],
             "Q4 everything (11 inputs)": PREPITCH_BASE + CTX + PROF}
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    P = {k: cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), d[c].values, y, cv=cv, method="predict_proba")[:, 1]
         for k, c in specs.items()}
    auc = {k: roc_auc_score(y, p) for k, p in P.items()}
    pairs = {"context over base": ("Q1 base + context", "Q0 base: speed + lead + pop"),
             "delivery over base + context": ("Q2 base + context + delivery profile", "Q1 base + context"),
             "Statcast profiles over base + context": ("Q3 base + context + Statcast pitcher profiles", "Q1 base + context"),
             "delivery over everything else": ("Q4 everything (11 inputs)", "Q3 base + context + Statcast pitcher profiles"),
             "all profiles over base + context": ("Q4 everything (11 inputs)", "Q1 base + context")}
    rng, grp = np.random.default_rng(seed), d.groupby("pitcher_id").indices
    keys = np.array(list(grp)); diffs = {k: [] for k in pairs}
    for _ in range(n_boot):
        idx = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        for k, (a, b) in pairs.items():
            diffs[k].append(roc_auc_score(y[idx], P[a][idx]) - roc_auc_score(y[idx], P[b][idx]))
    rows = [dict(section="model", item=k, n=len(d), auroc=v, max_vif=vif(d, specs[k])) for k, v in auc.items()]
    assert max(r["max_vif"] for r in rows) < 10
    for k, (a, b) in pairs.items():
        rows.append(dict(section="auroc_gain", item=k, n=len(d), auroc=auc[a] - auc[b],
                         ci_lo=np.percentile(diffs[k], 2.5), ci_hi=np.percentile(diffs[k], 97.5)))
    # coefficients + swings in the full model (pitcher-clustered)
    cols = specs["Q4 everything (11 inputs)"]
    X = sm.add_constant(d[cols].values)
    f = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.pitcher_id.values})
    mu = d[cols].mean()
    for i, c in enumerate(cols):
        lo_v, hi_v = (0, 1) if c in ("is_lhp", "bat_side_r") else (0, 2) if c == "outs" else \
                     (d[c].quantile(.1), d[c].quantile(.9))
        z = f.params[0] + sum(f.params[j + 1] * mu[x] for j, x in enumerate(cols) if x != c)
        p = lambda v: 1 / (1 + np.exp(-(z + f.params[i + 1] * v)))
        b, se = f.params[i + 1], f.bse[i + 1]
        rows.append(dict(section="full_model_term", item=CTX_LABEL[c], n=len(d), coef=b, se=se, p=f.pvalues[i + 1],
                         odds_per_sd=np.exp(b * d[c].std()), lo_value=lo_v, hi_value=hi_v,
                         swing_pp=100 * (p(hi_v) - p(lo_v))))
    # does the delivery coefficient shrink once context is in? (same rows)
    f0 = sm.Logit(y, sm.add_constant(d[PREPITCH_BASE + ["deliv"]].values)).fit(disp=0)
    f2 = sm.Logit(y, sm.add_constant(d[PREPITCH_BASE + CTX + ["deliv"]].values)).fit(disp=0)
    rows.append(dict(section="delivery_shrink", item="delivery coef per s: without context -> with context", n=len(d),
                     coef=f0.params[-1], coef_with_context=f2.params[-1],
                     corr_deliv_is_lhp=np.corrcoef(d.deliv, d.is_lhp)[0, 1]))
    out = pd.DataFrame(rows)
    out.to_csv(res_path("DF_add_context_prepitch.csv"), index=False)
    print(f"PART 1  rows {len(d)} (2024-26 forward), pitchers {d.pitcher_id.nunique()}")
    print(out[out.section != "full_model_term"][["section", "item", "auroc", "ci_lo", "ci_hi", "coef", "coef_with_context",
                                                 "corr_deliv_is_lhp"]].round(4).to_string())
    print(out[out.section == "full_model_term"][["item", "coef", "se", "p", "odds_per_sd", "lo_value", "hi_value",
                                                 "swing_pp"]].round(4).to_string())
    return out


# ───────────────────────────── part 2 ─────────────────────────────
def context_part2(n_boot=1000, seed=0):
    at = shipped_rows().merge(pitch_context(), on="play_id", how="left")
    d = at[at.pitch_class.notna()].dropna(subset=CTX).reset_index(drop=True)
    c = shipped_coefs()
    d["offset"] = c["intercept"] + sum(c[f] * d[f] for f in F4)
    d["p_exp"] = 1 / (1 + np.exp(-d.offset))
    d["resid"] = d.y - d.p_exp
    d["inning_grp"] = pd.cut(d.inning, [0, 3, 6, 99], labels=["1-3", "4-6", "7+"]).astype(str)
    d["outs_grp"] = d.outs.astype(int).astype(str)
    d["hand"] = np.where(d.is_lhp == 1, "LHP", "RHP")
    d["bat"] = np.where(d.bat_side_r == 1, "RHB", "LHB")
    d["base_grp"] = d.base.astype(str)
    print(f"\nPART 2  rows {len(d)} of {len(at)} (dropped {len(at) - len(d)} with an unclassified pitch or missing context)")

    # (a) pitch type main effect: raw, model-expected, observed-expected (runner-clustered bootstrap CI)
    rng, grp = np.random.default_rng(seed), d.groupby("runner_id").indices
    keys = np.array(list(grp)); boot = {k: [] for k in PITCH_CLASS.values()}
    for _ in range(n_boot):
        s = d.iloc[np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])]
        g = s.groupby("pitch_class").resid.mean()
        for k in boot:
            boot[k].append(g.get(k, np.nan))
    rows = []
    for k, g in d.groupby("pitch_class"):
        rows.append(dict(pitch_class=k, n=len(g), share=len(g) / len(d), safe_rate=g.y.mean(), expected=g.p_exp.mean(),
                         obs_minus_exp_pp=100 * g.resid.mean(), ci_lo_pp=100 * np.nanpercentile(boot[k], 2.5),
                         ci_hi_pp=100 * np.nanpercentile(boot[k], 97.5), gain_ft=g.gain_to_release_ft.mean(),
                         firstmove_ft=g.lead_at_firstmove_ft.mean()))
    pt = pd.DataFrame(rows).set_index("pitch_class").loc[["fastball", "breaking", "offspeed"]].reset_index()
    # odds ratios vs fastball: raw, and beyond the shipped model (offset), runner-clustered
    dm = pd.get_dummies(d.pitch_class)[["breaking", "offspeed"]].astype(float)
    raw = sm.Logit(d.y, sm.add_constant(dm)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.runner_id})
    adj = sm.Logit(d.y, sm.add_constant(dm), offset=d.offset).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.runner_id})
    adj0 = sm.Logit(d.y, np.ones((len(d), 1)), offset=d.offset).fit(disp=0)
    adj_nc = sm.Logit(d.y, sm.add_constant(dm), offset=d.offset).fit(disp=0)
    lr_p = chi2.sf(2 * (adj_nc.llf - adj0.llf), 2)
    for k in ["breaking", "offspeed"]:
        i = pt.index[pt.pitch_class == k][0]
        pt.loc[i, "raw_odds_vs_fastball"] = np.exp(raw.params[k]); pt.loc[i, "raw_p"] = raw.pvalues[k]
        pt.loc[i, "adj_odds_vs_fastball"] = np.exp(adj.params[k]); pt.loc[i, "adj_p"] = adj.pvalues[k]
    pt["lr_p_pitch_type_beyond_model"] = lr_p
    pt.to_csv(res_path("DF_add_pitchtype.csv"), index=False)
    print(pt.round(4).to_string())

    # (b) pitch type x game state: cells + interaction LR test per state (offset model), Holm-adjusted
    states = {"Outs": "outs_grp", "Inning": "inning_grp", "Pitcher hand": "hand", "Batter side": "bat", "Base stolen": "base_grp"}
    cells, tests = [], []
    for sname, col in states.items():
        for (lvl, pc), g in d.groupby([col, "pitch_class"]):
            se = np.sqrt((g.p_exp * (1 - g.p_exp)).sum()) / len(g)            # binomial SE of mean(y - p)
            cells.append(dict(state=sname, level=lvl, pitch_class=pc, n=len(g), safe_rate=g.y.mean(),
                              expected=g.p_exp.mean(), obs_minus_exp_pp=100 * g.resid.mean(),
                              ci_lo_pp=100 * (g.resid.mean() - 1.96 * se), ci_hi_pp=100 * (g.resid.mean() + 1.96 * se)))
        Xm = pd.get_dummies(d[[col, "pitch_class"]].astype(str), drop_first=True).astype(float)
        inter = pd.DataFrame({f"{a}:{b}": Xm[a] * Xm[b] for a in Xm if a.startswith(col) for b in Xm if b.startswith("pitch_class")})
        m0 = sm.Logit(d.y, sm.add_constant(Xm), offset=d.offset).fit(disp=0)
        m1 = sm.Logit(d.y, sm.add_constant(pd.concat([Xm, inter], axis=1)), offset=d.offset).fit(disp=0)
        lr = 2 * (m1.llf - m0.llf)
        tests.append(dict(state=sname, lr_chi2=lr, df=inter.shape[1], p=chi2.sf(lr, inter.shape[1])))
    T = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
    k = len(T)
    T["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(T.p)]))
    C = pd.DataFrame(cells).merge(T[["state", "p", "p_holm"]], on="state").rename(
        columns={"p": "interaction_p", "p_holm": "interaction_p_holm"})
    C.to_csv(res_path("DF_add_pitchtype_state.csv"), index=False)
    print(T.round(4).to_string())
    return pt, C, T, d


def context_fig(pt, C, T):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe7"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig = plt.figure(figsize=(12.5, 6.4), dpi=170, facecolor=SURF, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
    # A: safe rate observed vs expected by pitch class
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURF)
    x = np.arange(len(pt))
    ax.bar(x - 0.19, 100 * pt.safe_rate, 0.36, color=[PCOL[k] for k in pt.pitch_class], label="observed")
    ax.bar(x + 0.19, 100 * pt.expected, 0.36, color="#d6d5d0", label="shipped model expects")
    for i, r in pt.iterrows():
        ax.text(i - 0.19, 100 * r.safe_rate + 0.4, f"{100 * r.safe_rate:.1f}%", ha="center", fontsize=8.5, color=INK)
        ax.text(i + 0.19, 100 * r.expected + 0.4, f"{100 * r.expected:.1f}%", ha="center", fontsize=8.5, color=MUTED)
        ax.text(i, 100 * max(r.safe_rate, r.expected) + 2.2, f"obs − exp {r.obs_minus_exp_pp:+.1f} pts\n[{r.ci_lo_pp:+.1f}, {r.ci_hi_pp:+.1f}]",
                ha="center", fontsize=7.8, color=INK, va="bottom")
    ax.set_xticks(x); ax.set_xticklabels([f"{k.capitalize()}\nn={int(r.n):,}\nground {r.gain_ft:.1f} ft" for k, r in zip(pt.pitch_class, pt.itertuples())], fontsize=8)
    ax.set_ylim(70, 93); ax.set_ylabel("share of attempts safe (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Safe rate by the pitch thrown on the attempt", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    # B: obs - exp by pitch class within each game state
    ax = fig.add_subplot(gs[1]); ax.set_facecolor(SURF)
    order = ["Outs", "Inning", "Pitcher hand", "Batter side", "Base stolen"]
    lvl_order = {"Outs": ["0", "1", "2"], "Inning": ["1-3", "4-6", "7+"],
                 "Pitcher hand": ["RHP", "LHP"], "Batter side": ["RHB", "LHB"], "Base stolen": ["2B", "3B"]}
    ylabels, ypos, yv = [], [], 0
    for s in order:
        for lv in lvl_order[s]:
            for j, pc in enumerate(["fastball", "breaking", "offspeed"]):
                r = C[(C.state == s) & (C.level == lv) & (C.pitch_class == pc)]
                if r.empty or r.n.iloc[0] < 30:
                    continue
                r = r.iloc[0]; yy = yv + (j - 1) * 0.24
                ax.errorbar(r.obs_minus_exp_pp, yy, xerr=[[r.obs_minus_exp_pp - r.ci_lo_pp], [r.ci_hi_pp - r.obs_minus_exp_pp]],
                            fmt=PMARK[pc], color=PCOL[pc], ms=5, elinewidth=1, capsize=0,
                            label=pc.capitalize() if (s, lv) == ("Outs", "0") else None)
            p = T.set_index("state").loc[s, "p_holm"]
            ylabels.append(f"{s}: {lv}" + (f"   (interaction p={p:.2f}, Holm)" if lv == lvl_order[s][0] else ""))
            ypos.append(yv); yv += 1
        yv += 0.6
    ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=7.8); ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("observed − expected safe rate (percentage points); bars: 95% CI\n"
                  "expected = shipped 4-input model (speed, first-move lead, ground gained, pop)")
    ax.set_title("Pitch type × game state, beyond the model", loc="left", fontsize=10.5,
                 fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right", title="pitch thrown", title_fontsize=8)
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False); a.grid(axis="x" if a is ax else "y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.savefig(fig_path("Fig_add_pitchtype.png"), facecolor=SURF); plt.close(fig)


def run_context():
    context_part1()
    pt, C, T, _ = context_part2()
    context_fig(pt, C, T)
    print("wrote DF_add_context_prepitch.csv, DF_add_pitchtype.csv, DF_add_pitchtype_state.csv, Fig_add_pitchtype.png")


# ════ 7. Learner choice: logistic vs splines vs nested-tuned XGBoost / LightGBM ═══════════════════════════════
# Is it worth swapping the logistic regression for a tuned or boosted model?
#
# Same inputs, same rows, same outer 5-fold split (the shipped seed), four learners:
#   logistic        the shipped learner (sklearn defaults)
#   logistic+spline natural-cubic-ish splines on every input (SplineTransformer, 4 knots), mildly penalised;
#                   tests whether bending the curves (e.g. speed matters more for fast runners) buys anything
#   XGBoost         tuned by an INNER 3-fold grid search inside each outer training fold (nested: no peeking)
#   LightGBM        same nested tuning
# Two feature sets:
#   post-pitch  the shipped calculator's 4 inputs on all of its attempts
#   pre-pitch   the 6 pre-pitch inputs on §5's forward rows (2024-26, 2023 profiles)
# Reports out-of-fold AUROC and Brier, and the paired, clustered bootstrap CI of each learner's AUROC minus
# logistic's (runner-clustered for post-pitch, pitcher-clustered for pre-pitch).

def algo_learners(seed=42):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    inner = StratifiedKFold(3, shuffle=True, random_state=seed + 1)
    xgb = GridSearchCV(XGBClassifier(n_estimators=400, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", verbosity=0, random_state=seed, n_jobs=2),
                       {"max_depth": [2, 3, 4], "min_child_weight": [5, 20], "reg_lambda": [1, 10]},
                       scoring="roc_auc", cv=inner, n_jobs=2)
    lgb = GridSearchCV(LGBMClassifier(n_estimators=400, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, verbose=-1, random_state=seed, n_jobs=2),
                       {"num_leaves": [4, 8, 16], "min_child_samples": [20, 80], "reg_lambda": [0, 10]},
                       scoring="roc_auc", cv=inner, n_jobs=2)
    return {"logistic (shipped)": LogisticRegression(max_iter=5000),
            "logistic + splines": make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3),
                                                LogisticRegression(C=1.0, max_iter=5000)),
            "XGBoost (nested-tuned)": xgb, "LightGBM (nested-tuned)": lgb}


def algo_run(d, cols, cluster, tag, n_boot=1000, seed=0):
    X, y = d[cols].values, d.y.values
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    P, rows = {}, []
    for name, est in algo_learners().items():
        t = time.time()
        P[name] = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
        print(f"[{tag}] {name:26s} AUROC {roc_auc_score(y, P[name]):.4f}  Brier {brier_score_loss(y, P[name]):.4f}  ({time.time()-t:.0f}s)", flush=True)
    rng, grp = np.random.default_rng(seed), d.groupby(cluster).indices
    keys = np.array(list(grp)); base = "logistic (shipped)"
    diff = {k: [] for k in P if k != base}
    for _ in range(n_boot):
        idx = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        a0 = roc_auc_score(y[idx], P[base][idx])
        for k in diff:
            diff[k].append(roc_auc_score(y[idx], P[k][idx]) - a0)
    for k, p in P.items():
        dd = np.array(diff.get(k, [0.0]))
        rows.append(dict(features=tag, learner=k, n=len(d), n_features=len(cols), auroc=roc_auc_score(y, p),
                         brier=brier_score_loss(y, p), auroc_vs_logistic=roc_auc_score(y, p) - roc_auc_score(y, P[base]),
                         ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5)))
    return rows


def run_algo():
    rows = algo_run(shipped_rows(), F4, "runner_id", "post-pitch: shipped 4 inputs")
    rows += algo_run(rows_fwd(), list(SIX), "pitcher_id", "pre-pitch: 6 inputs (forward 2024-26)")
    out = pd.DataFrame(rows); out.to_csv(res_path("DF_add_algo.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string())


# ════ 8. v16: + pitch type + base; leakage audit, nested Optuna tuning, weights, forward test ═════════════════
# The v16 CANDIDATE odds model: the shipped 4 inputs + the pitch type thrown on the attempt
# (fastball reference / breaking / offspeed) + the base being stolen (3rd vs 2nd), with a leakage audit, an Optuna-tuned logistic regression scored by
# NESTED cross-validation, standardized variable weights, and a BETA variant that adds CV delivery time.
# Nothing here touches the shipped calculator (docs/index.html, DF_success_model.csv) — that is the next step.
#
#   1. leakage audit  every candidate column is classed by WHEN it is known; anything recorded after the pitch's
#                     outcome (count, end-of-PA score, run value, result) is banned and asserted absent
#   2. head-to-head   shipped 4 inputs -> + pitch type -> + pitch type + base, same rows, same outer folds
#   3. tuning         Optuna TPE over the logistic's whole design space, nested: an inner study per outer fold,
#                     objective = inner-CV log-loss; then one final study on all rows for the shipped config
#   4. weights        unpenalised MLE, runner-clustered: log-odds and odds ratio per unit and per SD, share of
#                     total weight, and the 10th->90th percentile swing in P(safe)
#   5. checks         forward-in-time (train 2023-25, tuned on 2023-25 only, test 2026); prior-season pop time
#   6. beta           + CV delivery time on the 2023 timed attempts (frozen v16 weights and refit)

V16_LABEL = {"sprint_speed": "Sprint speed", "lead_at_firstmove_ft": "Lead at first move",
         "gain_to_release_ft": "Ground gained to release", "pop_faced": "Catcher pop time",
         "pt_breaking": "Breaking ball (vs fastball)", "pt_offspeed": "Offspeed (vs fastball)",
         "base_is_3b": "Stealing 3rd (vs 2nd)"}
UNIT = {"sprint_speed": "ft/s", "lead_at_firstmove_ft": "ft", "gain_to_release_ft": "ft", "pop_faced": "s",
        "pt_breaking": "0→1", "pt_offspeed": "0→1", "base_is_3b": "0→1"}
# when each candidate column is known, relative to the steal attempt
AUDIT = [("sprint_speed", "season trait (all running plays)", True, "not outcome-derived"),
         ("lead_at_firstmove_ft", "pre-pitch (pitcher's first move)", True, "measured before the outcome"),
         ("gain_to_release_ft", "at release", True, "measured before the outcome"),
         ("pop_faced", "season trait of the catcher", True, "same-season average includes this throw: checked with prior-season pop"),
         ("pitch type", "at release", True, "the pitch thrown; known before the catch/tag"),
         ("base stolen (2B / 3B)", "pre-pitch", True, "which base the runner is going for; not outcome-derived"),
         ("lead_at_release_ft", "at release", False, "= first-move lead + ground gained exactly (collinear, VIF ~9,500)"),
         ("balls / strikes", "AFTER the pitch (feed count)", False, "LEAK: 3-strike rows are 28% caught vs 17-20%"),
         ("score_diff", "END of plate appearance (feed play.result)", False, "LEAK: can include runs scored after the attempt (verified on a live feed)"),
         ("outs", "pre-pitch (checked: no 3-out rows; CS rows carry no extra out)", False, "clean; not significant in the pre-pitch test (p = 0.15); left out to keep the model minimal"),
         ("run_value / result", "the outcome itself", False, "LEAK by definition")]


def suggest(trial) -> dict:
    p = {"penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet", "none"])}
    if p["penalty"] != "none":
        p["C"] = trial.suggest_float("C", 1e-3, 1e3, log=True)
    if p["penalty"] == "elasticnet":
        p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
    p["spline"] = trial.suggest_categorical("spline", [False, True])
    if p["spline"]:
        p["n_knots"] = trial.suggest_int("n_knots", 3, 7)
        p["degree"] = trial.suggest_int("degree", 1, 3)
    p["inter"] = trial.suggest_categorical("inter", ["none", "binary_x_cont", "pairwise"])
    p["class_weight"] = trial.suggest_categorical("class_weight", ["none", "balanced"])
    return p


def study(X, y, n_trials, seed, k=3):
    """Optuna TPE study minimising k-fold log-loss on (X, y); the shipped default is enqueued as trial 0."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(k, shuffle=True, random_state=seed + 7)

    def obj(trial):
        p = suggest(trial)
        oof = cross_val_predict(pipe(p), X, y, cv=cv, method="predict_proba")[:, 1]
        trial.set_user_attr("auroc", roc_auc_score(y, oof))
        return log_loss(y, oof)
    s = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    s.enqueue_trial({"penalty": "l2", "C": 1.0, "spline": False, "inter": "none", "class_weight": "none"})
    s.optimize(obj, n_trials=n_trials, n_jobs=2)
    return s


def params_of(t) -> dict:
    p = dict(t.params)
    return {**DEFAULT, **p}


def ece(y, p):
    return expected_calibration_error(y, p)


def crossfit_isotonic(y, p, seed=42):
    """Calibrate OOF predictions with isotonic fit on OTHER folds only (no row calibrates itself)."""
    out = np.zeros_like(p)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + 3).split(p.reshape(-1, 1), y):
        out[te] = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr]).predict(p[te])
    return out


def metrics_row(name, y, p, n_feat=None):
    e, bad = ece(y, p)
    return dict(model=name, auroc=roc_auc_score(y, p), log_loss=log_loss(y, p), brier=brier_score_loss(y, p),
                ece=e, bad_deciles=bad, n=len(y), n_features=n_feat)


def run_v16(outer_trials=40, final_trials=120, n_boot=1000, seed=0):
    t0 = time.time()
    d = v16_rows(); X, y = d[FEATS].values.astype(float), d.y.values
    n_ship = len(shipped_rows())
    print(f"rows {len(d)} (shipped {n_ship:,} minus {n_ship - len(d)} with no classified pitch); success {y.mean():.3f}")
    pd.DataFrame(AUDIT, columns=["column", "known", "used", "note"]).to_csv(res_path("DF_v16_leakage_audit.csv"), index=False)
    max_vif = vif(d, FEATS); assert max_vif < 10, f"VIF {max_vif:.1f}"

    # ── 2 + 3. same outer folds: shipped, + pitch type (default), + pitch type (nested-tuned) ──
    outer = StratifiedKFold(5, shuffle=True, random_state=42)
    X4 = np.c_[X[:, :4], np.zeros((len(d), len(BIN)))]                         # binaries zeroed = shipped inputs
    Xpt = X.copy(); Xpt[:, FEATS.index("base_is_3b")] = 0                        # pitch type only
    P = {"shipped 4 inputs (default LR)": cross_val_predict(pipe(DEFAULT), X4, y, cv=outer, method="predict_proba")[:, 1],
         "+ pitch type (default LR)": cross_val_predict(pipe(DEFAULT), Xpt, y, cv=outer, method="predict_proba")[:, 1],
         "v16: + pitch type + base (default LR)": cross_val_predict(pipe(DEFAULT), X, y, cv=outer, method="predict_proba")[:, 1]}
    tuned, chosen = np.zeros(len(d)), []
    for i, (tr, te) in enumerate(outer.split(X, y)):
        s = study(X[tr], y[tr], outer_trials, seed + i)
        bp = params_of(s.best_trial); chosen.append(dict(fold=i + 1, inner_log_loss=s.best_value, **bp))
        tuned[te] = pipe(bp).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        print(f"  outer fold {i+1}: best inner log-loss {s.best_value:.4f} -> {bp}  ({time.time()-t0:.0f}s)", flush=True)
    P["v16: Optuna-tuned (nested)"] = tuned
    P["v16 tuned + cross-fitted isotonic"] = crossfit_isotonic(y, tuned)
    P["v16 default + cross-fitted isotonic"] = crossfit_isotonic(y, P["v16: + pitch type + base (default LR)"])
    rows_m = [metrics_row(k, y, v) for k, v in P.items()]
    # paired, runner-clustered bootstrap of differences vs the shipped model and vs v16 default
    rng, grp = np.random.default_rng(seed), d.groupby("runner_id").indices
    keys = np.array(list(grp)); ref1, ref2 = "shipped 4 inputs (default LR)", "v16: + pitch type + base (default LR)"
    B = {k: {"auc1": [], "ll1": [], "auc2": [], "ll2": []} for k in P}
    for _ in range(n_boot):
        idx = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        a1, l1 = roc_auc_score(y[idx], P[ref1][idx]), log_loss(y[idx], P[ref1][idx])
        a2, l2 = roc_auc_score(y[idx], P[ref2][idx]), log_loss(y[idx], P[ref2][idx])
        for k in P:
            a, l = roc_auc_score(y[idx], P[k][idx]), log_loss(y[idx], P[k][idx])
            B[k]["auc1"].append(a - a1); B[k]["ll1"].append(l - l1); B[k]["auc2"].append(a - a2); B[k]["ll2"].append(l - l2)
    for r in rows_m:
        b = B[r["model"]]
        for key in ["auc1", "ll1", "auc2", "ll2"]:
            r[f"d_{key}_lo"], r[f"d_{key}_hi"] = np.percentile(b[key], [2.5, 97.5])
        r["d_auc_vs_shipped"] = r["auroc"] - roc_auc_score(y, P[ref1]); r["d_ll_vs_shipped"] = r["log_loss"] - log_loss(y, P[ref1])
        r["d_auc_vs_v16default"] = r["auroc"] - roc_auc_score(y, P[ref2]); r["d_ll_vs_v16default"] = r["log_loss"] - log_loss(y, P[ref2])
    MT = pd.DataFrame(rows_m); MT.to_csv(res_path("DF_v16_models.csv"), index=False)
    pd.DataFrame(chosen).to_csv(res_path("DF_v16_nested_choices.csv"), index=False)
    print(MT[["model", "auroc", "log_loss", "brier", "ece", "bad_deciles", "d_auc_vs_shipped", "d_auc1_lo", "d_auc1_hi",
              "d_auc_vs_v16default", "d_auc2_lo", "d_auc2_hi"]].round(4).to_string())

    # final study on all rows -> the configuration that would ship, + parameter importances
    import optuna
    fs = study(X, y, final_trials, seed + 99, k=5)
    final = params_of(fs.best_trial)
    trials = fs.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials.to_csv(res_path("DF_v16_final_trials.csv"), index=False)
    try:
        imp = optuna.importance.get_param_importances(fs)
    except Exception as e:                                  # fANOVA needs enough completed trials
        imp = {"error": str(e)}
    default_ll = trials.loc[trials.number == 0, "value"].iloc[0]
    print(f"final study: best 5-fold log-loss {fs.best_value:.5f} vs default {default_ll:.5f} -> {final}")
    print("param importances:", {k: round(v, 3) for k, v in imp.items()} if "error" not in imp else imp)

    # ── 4. weights: unpenalised MLE, runner-clustered ─────────────────────
    f = sm.Logit(y, sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.runner_id.values})
    mu, sd = d[FEATS].mean(), d[FEATS].std()
    W = []
    for i, c in enumerate(FEATS):
        b, se = f.params[i + 1], f.bse[i + 1]
        lo_v, hi_v = (0, 1) if c in BIN else (d[c].quantile(.1), d[c].quantile(.9))
        base = f.params[0] + sum(f.params[j + 1] * (0 if x in BIN else mu[x]) for j, x in enumerate(FEATS) if x != c)
        p = lambda v: 1 / (1 + np.exp(-(base + b * v)))
        W.append(dict(variable=V16_LABEL[c], unit=UNIT[c], logodds_per_unit=b, se=se, p=f.pvalues[i + 1],
                      or_per_unit=np.exp(b), sd=sd[c], logodds_per_sd=b * sd[c], logodds_per_sd_lo=(b - 1.96 * se) * sd[c],
                      logodds_per_sd_hi=(b + 1.96 * se) * sd[c], or_per_sd=np.exp(b * sd[c]),
                      p10=lo_v, p90=hi_v, swing_pp=100 * (p(hi_v) - p(lo_v))))
    W = pd.DataFrame(W)
    W["share_of_weight"] = W.logodds_per_sd.abs() / W.logodds_per_sd.abs().sum()
    W.to_csv(res_path("DF_v16_weights.csv"), index=False)
    print(W[["variable", "logodds_per_unit", "or_per_unit", "logodds_per_sd", "or_per_sd", "share_of_weight", "swing_pp", "p"]].round(4).to_string())

    # ── 5. checks: forward in time, and prior-season pop time ─────────────
    trn, tst = d.season <= 2025, d.season == 2026
    # QA fix: hyper-parameters for the forward test are tuned on 2023-25 ONLY; `final` was chosen with 2026 in view
    final_fwd = params_of(study(X[trn.values], y[trn.values], final_trials, seed + 199, k=5).best_trial)
    fwd = []
    for name, p_, Xa in [("shipped 4 inputs (default LR)", DEFAULT, X4), ("+ pitch type (default LR)", DEFAULT, Xpt),
                         ("v16: + pitch type + base (default LR)", DEFAULT, X),
                         ("v16 tuned on 2023-25 only", final_fwd, X)]:
        pr = pipe(p_).fit(Xa[trn], y[trn]).predict_proba(Xa[tst])[:, 1]
        fwd.append(metrics_row(name, y[tst], pr))
    FW = pd.DataFrame(fwd); FW["train"] = "2023-25"; FW["test"] = f"2026 (to {pd.to_datetime(d.date[d.season == 2026]).max():%d %b})"; FW["tuned_params"] = str(final_fwd)
    FW.to_csv(res_path("DF_v16_forward.csv"), index=False)
    print(FW[["model", "auroc", "log_loss", "ece", "n"]].round(4).to_string())
    pop = pd.read_csv(RAW / "poptime.csv")[["catcher_id", "season", "pop_2b_sba"]]
    pop["season"] += 1                                                     # last season's pop, joined to this season
    dp = d.merge(pop.rename(columns={"pop_2b_sba": "pop_prior"}), on=["catcher_id", "season"], how="left").dropna(subset=["pop_prior"])
    Xc = dp[FEATS].values.astype(float); Xp = Xc.copy(); Xp[:, 3] = dp.pop_prior.values
    cvp = StratifiedKFold(5, shuffle=True, random_state=42)
    fc = sm.Logit(dp.y.values, sm.add_constant(Xc)).fit(disp=0); fp = sm.Logit(dp.y.values, sm.add_constant(Xp)).fit(disp=0)
    POP = pd.DataFrame([dict(pop="same season (shipped)", n=len(dp),
                             auroc=roc_auc_score(dp.y, cross_val_predict(pipe(DEFAULT), Xc, dp.y.values, cv=cvp, method="predict_proba")[:, 1]),
                             b_pop=fc.params[4], b_breaking=fc.params[5], b_offspeed=fc.params[6]),
                        dict(pop="PRIOR season (leak-proof)", n=len(dp),
                             auroc=roc_auc_score(dp.y, cross_val_predict(pipe(DEFAULT), Xp, dp.y.values, cv=cvp, method="predict_proba")[:, 1]),
                             b_pop=fp.params[4], b_breaking=fp.params[5], b_offspeed=fp.params[6])])
    POP.to_csv(res_path("DF_v16_pop_prior.csv"), index=False)
    print(POP.round(4).to_string())

    # ── 6. BETA: + CV delivery time on the 2023 timed attempts ────────────
    cvd = load_delivery(2023)
    cvd = cvd[cvd.qa == "PASS"][["play_id", "delivery_s"]]
    db = d.merge(cvd, on="play_id", how="inner").reset_index(drop=True)
    db["delivery_10"] = db.delivery_s * 10
    off = f.params[0] + db[FEATS].values @ f.params[1:]
    yb = db.y.values
    fz0 = sm.Logit(yb, np.ones((len(db), 1)), offset=off).fit(disp=0)
    fz = sm.Logit(yb, sm.add_constant(db[["delivery_10"]].values), offset=off).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": db.pitcher_id.values})
    fz_nc = sm.Logit(yb, sm.add_constant(db[["delivery_10"]].values), offset=off).fit(disp=0)
    cvb = StratifiedKFold(5, shuffle=True, random_state=42)
    a6 = roc_auc_score(yb, cross_val_predict(pipe(DEFAULT), db[FEATS].values.astype(float), yb, cv=cvb, method="predict_proba")[:, 1])
    Xb7 = db[FEATS + ["delivery_10"]].values.astype(float)
    a7 = roc_auc_score(yb, cross_val_predict(pipe(DEFAULT), Xb7, yb, cv=cvb, method="predict_proba")[:, 1])   # same learner as a6
    tot = sm.Logit(yb, sm.add_constant(db[["sprint_speed", "lead_at_firstmove_ft", "pop_faced"] + BIN + ["delivery_10"]].values)).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": db.pitcher_id.values})
    BETA = dict(n=len(db), beta_per_0p1s=fz.params[1], se=fz.bse[1], ci_lo=fz.params[1] - 1.96 * fz.bse[1],
                ci_hi=fz.params[1] + 1.96 * fz.bse[1], or_per_0p1s=np.exp(fz.params[1]), lr_p=chi2.sf(2 * (fz_nc.llf - fz0.llf), 1),
                auroc_v16=a6, auroc_v16_plus_delivery=a7, total_beta_no_gain=tot.params[-1], total_se=tot.bse[-1],
                logodds_per_sd=fz.params[1] * db.delivery_10.std())
    print("BETA delivery:", {k: round(v, 4) for k, v in BETA.items()})

    meta = dict(rows=len(d), vif=max_vif, final_params=final, final_log_loss=fs.best_value, default_log_loss=default_ll,
                importances=imp, mle={FEATS[i]: float(f.params[i + 1]) for i in range(len(FEATS))},
                intercept=float(f.params[0]), beta_delivery=BETA, outer_trials=outer_trials, final_trials=final_trials,
                seconds=time.time() - t0)
    json.dump(meta, open(res_path("DF_v16_meta.json"), "w"), indent=1, default=float)
    v16_figs(MT, W, imp, trials, BETA)
    print(f"done in {time.time() - t0:.0f}s")


def v16_figs(MT, W, imp, trials, BETA):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREY, INK, MUTED, SURF, GRID = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe7"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
    # weights
    w = W.assign(a=W.logodds_per_sd.abs()).sort_values("a")
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170, facecolor=SURF, layout="constrained"); ax.set_facecolor(SURF)
    for i, r in enumerate(w.itertuples()):
        col = ORANGE if "(vs" in r.variable else BLUE
        ax.barh(i, r.logodds_per_sd, height=0.55, color=col)
        ax.errorbar(r.logodds_per_sd, i, xerr=[[r.logodds_per_sd - r.logodds_per_sd_lo], [r.logodds_per_sd_hi - r.logodds_per_sd]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.text(max(r.logodds_per_sd_hi, 0) + 0.02, i, f"{r.logodds_per_sd:+.2f}  (odds ×{r.or_per_sd:.2f}; {r.share_of_weight:.0%} of weight)",
                va="center", fontsize=8.3, color=INK)
    ax.set_yticks(range(len(w))); ax.set_yticklabels(w.variable)
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlim(-0.05, w.logodds_per_sd_hi.max() + 0.75)
    ax.set_xlabel("log-odds of SAFE per 1 SD of the input (bars: 95% CI, runner-clustered)")
    ax.set_title("v16 candidate: how much each input weighs, on one scale", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.savefig(fig_path("Fig_v16_weights.png"), facecolor=SURF); plt.close(fig)
    # tuning
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=170, facecolor=SURF, layout="constrained")
    ax = axs[0]; ax.set_facecolor(SURF)
    if "error" not in imp:
        it = sorted(imp.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in it], [v for _, v in it], color=BLUE, height=0.55)
        for i, (_, v) in enumerate(it):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.3)
    ax.set_xlabel("fANOVA importance (share of log-loss variance across trials)")
    ax.set_title("Which tuning knobs mattered", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax = axs[1]; ax.set_facecolor(SURF)
    t = trials.sort_values("number")
    ax.plot(t.number, t.value, ".", color=GREY, ms=4, label="trial log-loss (5-fold)")
    ax.plot(t.number, t.value.cummin(), "-", color=BLUE, lw=2, label="best so far")
    ax.axhline(t.value.iloc[0], color=ORANGE, lw=1.2, ls="--", label=f"shipped default {t.value.iloc[0]:.4f}")
    ax.set_ylim(t.value.min() - 0.002, min(t.value.quantile(0.9), t.value.min() + 0.03))
    ax.set_xlabel("Optuna trial"); ax.set_ylabel("log-loss (lower is better)")
    ax.set_title("Search history: tuning shaves log-loss a little", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False); a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
    fig.savefig(fig_path("Fig_v16_tuning.png"), facecolor=SURF); plt.close(fig)


# ════ 9. v16 checks: base stolen, drift recalibration, per-base slopes ════════════════════════════════════════
# Follow-ups on the v16 candidate.
#
#   1. Base stolen   does stealing 3rd (vs 2nd) add to v16? The base is known before the pitch and cannot leak the
#                    outcome. Same rows, same 5 folds, same learner (pipe(DEFAULT)); paired,
#                    runner-clustered bootstrap; unpenalised odds ratio with runner-clustered CI.
#   2. Drift recalibration   the steal success rate falls every season since 2023, so a model
#                    fit on past seasons over-predicts the next one. Two forward folds (2023-24 -> 2025, 2023-25 -> 2026),
#                    each scoring six ways of producing probabilities:
#                      none                 the fitted model as is
#                      isotonic, all train  the shipped approach: one isotonic map on out-of-fold training predictions
#                      intercept, last season   shift the log-odds so the most recent training season's OOF predictions
#                                           match its observed rate (one parameter; 'calibration-in-the-large')
#                      isotonic, last season    isotonic map fit on the most recent training season's OOF predictions
#                      recency-weighted fit training rows weighted 0.5 ** (seasons before the latest)
#                      recency + intercept  both
#                    Every calibrator is fit on training-season OUT-OF-FOLD predictions only; the test season is never seen.

def base_stolen(d, n_boot=1000, seed=0):
    y = d.y.values
    cols17 = FEATS                                         # v16 now includes the base
    cols16 = [c for c in FEATS if c != "base_is_3b"]      # the same model without it
    assert not BANNED & set(cols17)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    p16 = cross_val_predict(pipe(DEFAULT), d[cols16].values.astype(float), y, cv=cv, method="predict_proba")[:, 1]
    p17 = cross_val_predict(pipe(DEFAULT), d[cols17].values.astype(float), y, cv=cv, method="predict_proba")[:, 1]
    rng, grp = np.random.default_rng(seed), d.groupby("runner_id").indices
    keys = np.array(list(grp)); da, dl = [], []
    for _ in range(n_boot):
        i = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        da.append(roc_auc_score(y[i], p17[i]) - roc_auc_score(y[i], p16[i]))
        dl.append(log_loss(y[i], p17[i]) - log_loss(y[i], p16[i]))
    f = sm.Logit(y, sm.add_constant(d[cols17].values.astype(float))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d.runner_id.values})
    b, se = f.params[-1], f.bse[-1]
    out = dict(n=len(d), n_3b=int(d.base_is_3b.sum()), safe_2b=d[d.base_is_3b == 0].y.mean(), safe_3b=d[d.base_is_3b == 1].y.mean(),
               auroc_v16=roc_auc_score(y, p16), auroc_v16_plus_base=roc_auc_score(y, p17),
               d_auroc=roc_auc_score(y, p17) - roc_auc_score(y, p16), d_auroc_lo=np.percentile(da, 2.5), d_auroc_hi=np.percentile(da, 97.5),
               d_logloss=log_loss(y, p17) - log_loss(y, p16), d_logloss_lo=np.percentile(dl, 2.5), d_logloss_hi=np.percentile(dl, 97.5),
               coef_3b=b, se_3b=se, p_3b=f.pvalues[-1], or_3b=np.exp(b), or_3b_lo=np.exp(b - 1.96 * se), or_3b_hi=np.exp(b + 1.96 * se),
               vif=vif(d, cols17),
               mean_gain_2b=d[d.base_is_3b == 0].gain_to_release_ft.mean(), mean_gain_3b=d[d.base_is_3b == 1].gain_to_release_ft.mean(),
               mean_lead_2b=d[d.base_is_3b == 0].lead_at_firstmove_ft.mean(), mean_lead_3b=d[d.base_is_3b == 1].lead_at_firstmove_ft.mean())
    assert out["vif"] < 10
    pd.DataFrame([out]).to_csv(res_path("DF_v16_base.csv"), index=False)
    print("BASE STOLEN:", {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()})
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def recal(d, seed=42):
    X, y, season = d[FEATS].values.astype(float), d.y.values, d.season.values
    rows = []
    for test_season in (2025, 2026):
        tr, te = season < test_season, season == test_season
        last = season[tr].max()
        Xtr, ytr, str_ = X[tr], y[tr], season[tr]
        w = 0.5 ** (last - str_)                                    # recency weights: latest season 1, previous 0.5, ...
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        variants = {}
        for wname, sw in [("unweighted", None), ("recency", w)]:
            oof = np.zeros(len(ytr))
            for a, b in cv.split(Xtr, ytr):                          # out-of-fold TRAINING predictions for the calibrators
                m = pipe(DEFAULT).fit(Xtr[a], ytr[a], **({} if sw is None else {"lr__sample_weight": sw[a]}))
                oof[b] = m.predict_proba(Xtr[b])[:, 1]
            full = pipe(DEFAULT).fit(Xtr, ytr, **({} if sw is None else {"lr__sample_weight": sw}))
            pte = full.predict_proba(X[te])[:, 1]
            variants[wname] = (oof, pte)
        oof, pte = variants["unweighted"]
        lastm = str_ == last
        # intercept shift: one offset logistic on the last training season's OOF log-odds
        shift = sm.Logit(ytr[lastm], np.ones((lastm.sum(), 1)), offset=_logit(oof[lastm])).fit(disp=0).params[0]
        roof, rpte = variants["recency"]
        rshift = sm.Logit(ytr[lastm], np.ones((lastm.sum(), 1)), offset=_logit(roof[lastm])).fit(disp=0).params[0]
        sig = lambda z: 1 / (1 + np.exp(-z))
        P = {"none": pte,
             "isotonic, all train (shipped approach)": IsotonicRegression(out_of_bounds="clip").fit(oof, ytr).predict(pte),
             "intercept shift, last season": sig(_logit(pte) + shift),
             "isotonic, last season": IsotonicRegression(out_of_bounds="clip").fit(oof[lastm], ytr[lastm]).predict(pte),
             "recency-weighted fit": rpte,
             "recency-weighted + intercept shift": sig(_logit(rpte) + rshift)}
        for k, p in P.items():
            p = np.clip(p, 1e-6, 1 - 1e-6)
            e, bad = ece(y[te], p)
            rows.append(dict(test_season=test_season, train=f"{season[tr].min()}-{last}", method=k, n=int(te.sum()),
                             observed=y[te].mean(), mean_pred=p.mean(), gap_pp=100 * (p.mean() - y[te].mean()),
                             auroc=roc_auc_score(y[te], p), log_loss=log_loss(y[te], p), brier=brier_score_loss(y[te], p),
                             ece=e, bad_deciles=bad, intercept_shift=shift if "intercept" in k and "recency" not in k
                             else rshift if k.startswith("recency-weighted +") else np.nan))
    # in-season recalibration: fit on 2023-25, shift the intercept on 2026's EARLY attempts (to 31 May), score June on
    d26 = d[d.season == 2026]; early = pd.to_datetime(d26.date) <= "2026-05-31"
    tr = season <= 2025
    oof_tr = cross_val_predict(pipe(DEFAULT), X[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                               method="predict_proba")[:, 1]
    full = pipe(DEFAULT).fit(X[tr], y[tr])
    pe, pl = full.predict_proba(d26[early][FEATS].values.astype(float))[:, 1], full.predict_proba(d26[~early][FEATS].values.astype(float))[:, 1]
    ye, yl = d26[early].y.values, d26[~early].y.values
    sh = sm.Logit(ye, np.ones((len(ye), 1)), offset=_logit(pe)).fit(disp=0).params[0]
    sig = lambda z: 1 / (1 + np.exp(-z))
    for k, p in {"none": pl, "isotonic, all train (shipped approach)": IsotonicRegression(out_of_bounds="clip").fit(oof_tr, y[tr]).predict(pl),
                 "IN-SEASON intercept shift (Apr-May 2026)": sig(_logit(pl) + sh)}.items():
        p = np.clip(p, 1e-6, 1 - 1e-6); e, bad = ece(yl, p)
        rows.append(dict(test_season="2026 from June", train="2023-2025", method=k, n=len(yl), observed=yl.mean(), mean_pred=p.mean(),
                         gap_pp=100 * (p.mean() - yl.mean()), auroc=roc_auc_score(yl, p), log_loss=log_loss(yl, p),
                         brier=brier_score_loss(yl, p), ece=e, bad_deciles=bad,
                         intercept_shift=sh if k.startswith("IN-SEASON") else np.nan, n_calib=len(ye)))
    out = pd.DataFrame(rows); out.to_csv(res_path("DF_v16_recal.csv"), index=False)
    pd.set_option("display.width", 220)
    print(out[["test_season", "method", "observed", "mean_pred", "gap_pp", "auroc", "log_loss", "brier", "ece", "bad_deciles",
               "intercept_shift"]].round(4).to_string())
    return out


def base_slopes(d, n_boot=1000, seed=0):
    """Does a foot of lead (or of ground gained) count differently off 2nd than off 1st? v16 vs v16 + base x lead +
    base x gain: still a plain, readable logistic (two extra slopes), same rows/folds/learner, raw and cross-fitted isotonic."""
    d = d.assign(b3_x_lead=d.base_is_3b * d.lead_at_firstmove_ft, b3_x_gain=d.base_is_3b * d.gain_to_release_ft)
    y = d.y.values; cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cols = {"v16 (linear)": FEATS, "v16 + base x lead + base x gain": FEATS + ["b3_x_lead", "b3_x_gain"]}
    P = {k: cross_val_predict(pipe(DEFAULT), d[c].values.astype(float), y, cv=cv, method="predict_proba")[:, 1] for k, c in cols.items()}
    P.update({k + " + isotonic": crossfit_isotonic(y, P[k]) for k in list(P)})
    rng, grp = np.random.default_rng(seed), d.groupby("runner_id").indices; keys = np.array(list(grp))
    ref = "v16 (linear)"; B = {k: [] for k in P}; L = {k: [] for k in P}
    for _ in range(n_boot):
        i = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        a0, l0 = roc_auc_score(y[i], P[ref][i]), log_loss(y[i], P[ref][i])
        for k in P:
            B[k].append(roc_auc_score(y[i], P[k][i]) - a0); L[k].append(log_loss(y[i], P[k][i]) - l0)
    f = sm.Logit(y, sm.add_constant(d[cols["v16 + base x lead + base x gain"]].values.astype(float))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d.runner_id.values})
    nm = ["const"] + cols["v16 + base x lead + base x gain"]; c = dict(zip(nm, f.params))
    rows = []
    for k, p in P.items():
        e, bad = ece(y, p)
        rows.append(dict(model=k, auroc=roc_auc_score(y, p), log_loss=log_loss(y, p), brier=brier_score_loss(y, p), ece=e,
                         bad_deciles=bad, d_auroc=roc_auc_score(y, p) - roc_auc_score(y, P[ref]),
                         d_auroc_lo=np.percentile(B[k], 2.5), d_auroc_hi=np.percentile(B[k], 97.5),
                         d_logloss=log_loss(y, p) - log_loss(y, P[ref]), d_ll_lo=np.percentile(L[k], 2.5), d_ll_hi=np.percentile(L[k], 97.5),
                         lead_slope_2b=c["lead_at_firstmove_ft"], lead_slope_3b=c["lead_at_firstmove_ft"] + c["b3_x_lead"],
                         gain_slope_2b=c["gain_to_release_ft"], gain_slope_3b=c["gain_to_release_ft"] + c["b3_x_gain"],
                         p_b3_x_lead=f.pvalues[nm.index("b3_x_lead")], p_b3_x_gain=f.pvalues[nm.index("b3_x_gain")],
                         vif=vif(d, cols["v16 + base x lead + base x gain"])))
    out = pd.DataFrame(rows); out.to_csv(res_path("DF_v16_base_slopes.csv"), index=False)
    print(out[["model", "auroc", "log_loss", "ece", "bad_deciles", "d_auroc", "d_auroc_lo", "d_auroc_hi", "d_logloss"]].round(4).to_string())
    print(out.iloc[1][["lead_slope_2b", "lead_slope_3b", "gain_slope_2b", "gain_slope_3b", "p_b3_x_lead", "p_b3_x_gain", "vif"]].round(4).to_dict())
    return out


def run_v16_checks():
    d = v16_rows()
    base_stolen(d)
    recal(d)
    base_slopes(d)


# ════ 10. The public standard: Powers et al. and The Pitcher's Dilemma vs this model ═══════════════════════════
# The public standard for stolen-base probability vs this model, and the rationale figures for
# each modelling step. Visualisation layer only: every model drawn here is either (a) a PUBLISHED specification applied
# with its published coefficients, (b) the same specification refit on these rows (its best case), or (c) the project's
# own v15 / v16 — nothing here feeds the shipped product.
#
# Public baselines (the two papers are cited in the report):
#   Powers, Ramani, Hahn & Schaefer (2026, arXiv 2601.15608 v1), Table 1 SB success, fixed effects:
#       logit = 1.157 + 0.052·lead_at_first_move(ft) + 0.130·sprint_speed(ft/s) − 0.054·catcher_arm(mph)
#   The Pitcher's Dilemma (42 Analytics), Table 1 generic model:
#       logit = −16.219 + 0.2356·lead(ft) + 0.2047·sprint_speed + 0.4583·avg_jump_speed(ft/s) + 1.5094·pop(s) + 0.4521·post2023
#       avg_jump_speed is rebuilt as they describe: a power law t = a·d^b fitted to each runner-season's 5-ft running
#       splits, jump time = time to cover the ground he gained, averaged over his attempts.

C_PUB, C_REFIT, C_OURS, C_BASE = "#a3a29c", "#eb6834", "#2a78d6", "#d6d5d0"
POWERS_V1 = dict(intercept=1.157, lead_at_firstmove_ft=0.052, sprint_speed=0.130, arm=-0.054)
DILEMMA = dict(intercept=-16.2190, lead_at_firstmove_ft=0.2356, sprint_speed=0.2047, jump_speed=0.4583,
               pop_faced=1.5094, post2023=0.4521)


def _style(ax):
    ax.set_facecolor(SURF); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def frozen(d, coef):
    z = coef["intercept"] + sum(v * (d["arm"] if k == "arm" else d[k]) for k, v in coef.items() if k != "intercept")
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def _recal_intercept(y, p_frozen, cv):
    """Published slopes kept; only the intercept is re-learned on training folds (a fair calibration test)."""
    import statsmodels.api as sm
    z = np.log(p_frozen / (1 - p_frozen)); out = np.zeros_like(p_frozen)
    for tr, te in cv.split(z.reshape(-1, 1), y):
        c = sm.Logit(y[tr], np.ones((len(tr), 1)), offset=z[tr]).fit(disp=0).params[0]
        out[te] = 1 / (1 + np.exp(-(z[te] + c)))
    return out


def baseline_models(d):
    y = d.y.values; cv = StratifiedKFold(5, shuffle=True, random_state=42)
    lr = lambda cols: cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                                        d[cols].values.astype(float), y, cv=cv, method="predict_proba")[:, 1]
    base = np.zeros(len(d))
    for tr, te in cv.split(d, y):
        base[te] = y[tr].mean()
    pw, dl = frozen(d, POWERS_V1), frozen(d, DILEMMA)
    v16 = cross_val_predict(pipe(DEFAULT), d[FEATS].values.astype(float), y, cv=cv, method="predict_proba")[:, 1]
    P = {("baseline", "League success rate only"): base,
         ("baseline", "Sprint speed only"): lr(["sprint_speed"]),
         ("published", "Powers et al. v1, as published"): pw,
         ("published", "Powers et al. v1, intercept re-fit"): _recal_intercept(y, pw, cv),
         ("published", "Pitcher's Dilemma, as published"): dl,
         ("published", "Pitcher's Dilemma, intercept re-fit"): _recal_intercept(y, dl, cv),
         ("refit", "Powers spec refit here (lead, speed, arm)"): lr(["lead_at_firstmove_ft", "sprint_speed", "arm"]),
         ("refit", "Pitcher's Dilemma spec refit here (lead, speed, jump, pop)"):
             lr(["lead_at_firstmove_ft", "sprint_speed", "jump_speed", "pop_faced"]),
         ("ours", "StealIQ v15 (shipped: speed, lead, ground gained, pop)"): lr(CONT),
         ("ours", "StealIQ v16 (+ pitch type + base)"): v16,
         ("ours", "StealIQ v16 + isotonic (recommended)"): crossfit_isotonic(y, v16)}
    return P, y


def baseline_score(P, y, d, n_boot=1000, seed=0):
    rng, grp = np.random.default_rng(seed), d.groupby("runner_id").indices; keys = np.array(list(grp))
    boot = {k: [] for k in P}
    for _ in range(n_boot):
        i = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        for k, p in P.items():
            boot[k].append(roc_auc_score(y[i], p[i]))
    out = []
    for (fam, name), p in P.items():
        p = np.clip(p, 1e-6, 1 - 1e-6); e, bad = ece(y, p)
        lo, hi = np.percentile(boot[(fam, name)], [2.5, 97.5])
        out.append(dict(family=fam, model=name, n=len(y), auroc=roc_auc_score(y, p), auroc_lo=lo, auroc_hi=hi,
                        log_loss=log_loss(y, p), brier=brier_score_loss(y, p), ece=e, bad_deciles=bad,
                        mean_pred=p.mean(), observed=y.mean()))
    return pd.DataFrame(out)


def baseline_ladder(d):
    y = d.y.values; cv = StratifiedKFold(5, shuffle=True, random_state=42)
    steps = [("League rate", [], "everyone gets the league rate: no information"),
             ("+ sprint speed", ["sprint_speed"], "what fans and scouts reach for first"),
             ("+ lead at first move", ["sprint_speed", "lead_at_firstmove_ft"], "the lead public models use (Powers, 42 Analytics)"),
             ("+ catcher pop time", ["sprint_speed", "lead_at_firstmove_ft", "pop_faced"], "the other side of the race: the throw"),
             ("+ ground gained to release", CONT, "the secondary lead public data lacks: the biggest step"),
             ("+ pitch type", CONT + ["pt_breaking", "pt_offspeed"], "slower pitches buy time after release"),
             ("+ base stolen", FEATS, "a foot of lead off 2nd is not a foot off 1st")]
    out = []
    for name, cols, why in steps:
        if cols:
            p = cross_val_predict(pipe(DEFAULT), d[cols].values.astype(float), y, cv=cv,   # same learner as v16
                                  method="predict_proba")[:, 1]
            a, ll = roc_auc_score(y, p), log_loss(y, p)
        else:
            a, ll = 0.5, log_loss(y, np.full(len(y), y.mean()))
        out.append(dict(step=name, inputs=", ".join(cols) or "—", why=why, auroc=a, log_loss=ll))
    t = pd.DataFrame(out); t["gain"] = t.auroc.diff().fillna(0)
    return t


def baseline_seasons(d):
    lg = pd.read_csv(RAW / "league_sb_rates.csv")
    tr = d.groupby("season").agg(tracked_attempts=("y", "size"), tracked_success=("y", "mean"))
    by = d.assign(b=np.where(d.base_is_3b == 1, "3B", "2B")).pivot_table(index="season", columns="b", values="y", aggfunc="mean")
    by.columns = [f"tracked_success_{c}" for c in by.columns]
    return lg.merge(tr, left_on="season", right_index=True, how="left").merge(by, left_on="season", right_index=True, how="left")


def baseline_scenarios(d, P):
    """Real runner-seasons: what each model expects on his average attempt vs what he actually did."""
    import statsmodels.api as sm
    picks = {"Josh Naylor 2025": (647304, 2025), "Shohei Ohtani 2024": (660271, 2024)}
    for nm, sid in [("Chandler Simpson", None), ("Elly De La Cruz", None)]:
        s = pd.read_csv(RAW / "Raw_Season.csv")
        hit = s[s.player_name == nm].sort_values("sb_attempts")
        if len(hit):
            r = hit.iloc[-1]; picks[f"{nm} {int(r.season)}"] = (int(r.runner_id), int(r.season))
    y = d.y.values
    fits = {"Sprint speed only": ["sprint_speed"], "Powers spec refit": ["lead_at_firstmove_ft", "sprint_speed", "arm"],
            "StealIQ v16": FEATS}
    out = []
    for label, (rid, yr) in picks.items():
        sub = d[(d.runner_id == rid) & (d.season == yr)]
        if sub.empty:
            continue
        rec = dict(runner=label, attempts=len(sub), actual=sub.y.mean(), sprint_speed=sub.sprint_speed.mean(),
                   lead=sub.lead_at_firstmove_ft.mean(), gain=sub.gain_to_release_ft.mean())
        for k, cols in fits.items():
            f = sm.Logit(y, sm.add_constant(d[cols].values.astype(float))).fit(disp=0)
            rec[k] = float(np.mean(f.predict(sm.add_constant(sub[cols].values.astype(float), has_constant="add"))))
        out.append(rec)
    return pd.DataFrame(out)


# ─────────────────────────────── figures ───────────────────────────────
def baseline_figures(S, LD, SE, SC, P, y, d):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK})
    fam_col = {"baseline": C_BASE, "published": C_PUB, "refit": C_REFIT, "ours": C_OURS}

    # 1. the MLB standard: league success and attempts, 2015-2026
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), dpi=170, facecolor=SURF, layout="constrained")
    for ax, col, lab in [(axs[0], "success_pct", "league stolen-base success (%)"),
                         (axs[1], "attempts_per_game", "steal attempts per game")]:
        _style(ax); ax.axvspan(2022.5, 2026.5, color="#eef4fc", lw=0)
        ax.plot(SE.season, SE[col], "o-", color=C_OURS, lw=2, ms=5)
        for yy in (2015, 2022, 2023, SE.season.max()):
            v = SE.loc[SE.season == yy, col].iloc[0]
            ax.annotate(f"{v:.1f}%" if col == "success_pct" else f"{v:.2f}", (yy, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=INK)
        ax.set_ylabel(lab); ax.set_xticks(SE.season); ax.tick_params(axis="x", rotation=45)
        ax.text(2022.6, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()), "2023 rules: bigger bases,\n2-disengagement limit",
                fontsize=7.5, color=MUTED)
    axs[0].set_title("Success jumped with the 2023 rules, then slid every year", loc="left", fontweight="bold", color=INK)
    axs[1].set_title("Attempts surged, peaked in 2024", loc="left", fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_cmp_league_rates.png"), facecolor=SURF); plt.close(fig)

    # 2. public baselines vs ours: AUROC and calibration
    show = S[~S.model.str.contains("as published")]                      # intercept-refit versions are the fair ones
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=170, facecolor=SURF, layout="constrained",
                            gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]; _style(ax); ax.grid(axis="x", color=GRID, lw=0.6); ax.grid(axis="y", visible=False)
    for i, r in enumerate(show.iloc[::-1].itertuples()):
        ax.barh(i, r.auroc - 0.5, left=0.5, color=fam_col[r.family], height=0.6)
        ax.errorbar(r.auroc, i, xerr=[[r.auroc - r.auroc_lo], [r.auroc_hi - r.auroc]], fmt="none", ecolor=INK, capsize=2, lw=0.8)
        ax.text(r.auroc_hi + 0.004, i, f"{r.auroc:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(show))); ax.set_yticklabels(show.model.iloc[::-1], fontsize=8)
    ax.set_xlim(0.48, 0.84); ax.set_xlabel("AUROC, out-of-fold on identical attempts (95% CI, runner-clustered)")
    ax.set_title("Who can tell a safe steal from a caught one?", loc="left", fontweight="bold", color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=fam_col[f], label=l) for f, l in [
        ("baseline", "naive baselines"), ("published", "published public models (intercept re-fit)"),
        ("refit", "public specs, refit here (their best case)"), ("ours", "this project")]],
        frameon=False, fontsize=7.5, loc="upper right")
    ax = axs[1]; _style(ax)
    # reliability: predicted vs observed by decile
    for (fam, name), c, ls in [(("published", "Powers et al. v1, intercept re-fit"), C_PUB, "--"),
                               (("refit", "Pitcher's Dilemma spec refit here (lead, speed, jump, pop)"), C_REFIT, "-."),
                               (("ours", "StealIQ v16 + isotonic (recommended)"), C_OURS, "-")]:
        p = P[(fam, name)]; q = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
        short = {"Powers et al. v1, intercept re-fit": "Powers et al. v1", "StealIQ v16 + isotonic (recommended)": "StealIQ v16 + isotonic"}
        ax.plot(g.p, g.y, ls, marker="o", ms=4, color=c, lw=1.8, label=short.get(name, "Pitcher's Dilemma spec (refit)"))
    ax.plot([0.55, 1], [0.55, 1], color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0.5, 1); ax.set_ylim(0.5, 1); ax.set_xlabel("predicted P(safe), decile mean"); ax.set_ylabel("observed share safe")
    ax.set_title("…and does 80% mean 80%? (deciles)", loc="left", fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.savefig(fig_path("Fig_cmp_baselines.png"), facecolor=SURF); plt.close(fig)

    # 3. separation: predicted P(safe) for safe vs caught attempts
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), dpi=170, facecolor=SURF, layout="constrained", sharey=True)
    for ax, key in zip(axs, [("baseline", "Sprint speed only"), ("refit", "Powers spec refit here (lead, speed, arm)"),
                             ("ours", "StealIQ v16 (+ pitch type + base)")]):
        _style(ax); p = P[key]; bins = np.linspace(0.3, 1, 36)
        ax.hist(p[y == 1], bins=bins, color=C_OURS, alpha=0.55, density=True, label="safe")
        ax.hist(p[y == 0], bins=bins, color=C_REFIT, alpha=0.55, density=True, label="caught")
        ax.set_title(key[1].split(" (")[0].replace(" here", ""), loc="left", fontweight="bold", color=INK, fontsize=9.5)
        ax.text(0.31, ax.get_ylim()[1] * 0.85, f"AUROC {roc_auc_score(y, p):.3f}", fontsize=8.5, color=INK)
        ax.set_xlabel("predicted P(safe)")
    axs[0].set_ylabel("density"); axs[0].legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle("The more the two humps separate, the more the model knows before the tag", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_cmp_separation.png"), facecolor=SURF); plt.close(fig)

    # 4. why speed is the wrong lever
    sel = pd.read_csv(res_path("DF_v13_SelectionEffect.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=170, facecolor=SURF, layout="constrained")
    for ax, col, lab in [(axs[0], "sprint_speed", "sprint speed (ft/s), quintiles"), (axs[1], "gain_to_release_ft", "ground gained to release (ft), quintiles")]:
        _style(ax); q = pd.qcut(d[col], 5); g = d.groupby(q).y.agg(["mean", "size"])
        x = d.groupby(q)[col].median().values; se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
        ax.errorbar(x, 100 * g["mean"], yerr=196 * se, fmt="o-", color=C_OURS if col != "sprint_speed" else C_REFIT, capsize=3, lw=2)
        ax.set_ylim(55, 100); ax.set_xlabel(lab + " (points at bin medians)"); ax.set_ylabel("share of attempts safe (%)")
        sw = 100 * (g["mean"].iloc[-1] - g["mean"].iloc[0])
        ax.set_title(f"{'Speed' if col == 'sprint_speed' else 'Ground gained'}: {sw:+.0f} pts slowest→fastest quintile"
                     if col == "sprint_speed" else f"Ground gained: {sw:+.0f} pts least→most quintile", loc="left", fontweight="bold", color=INK, fontsize=9.5)
    ax = axs[2]; _style(ax)
    ax.bar(range(len(sel)), sel.attempt_pct, color=C_REFIT, width=0.6)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel.sprint_speed_all, fontsize=8)
    for i, v in enumerate(sel.attempt_pct):
        ax.text(i, v + 0.08, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_ylabel("attempt rate per opportunity (%)"); ax.set_xlabel("sprint-speed quintile")
    ax.set_title(f"…because speed decides WHO runs ({sel.attempt_pct.iloc[-1] / sel.attempt_pct.iloc[0]:.1f}×)", loc="left",
                 fontweight="bold", color=INK, fontsize=9.5)
    fig.savefig(fig_path("Fig_cmp_speed_vs_ground.png"), facecolor=SURF); plt.close(fig)

    # 5. the rationale ladder: what each step adds, and why it is there
    import textwrap
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=170, facecolor=SURF, layout="constrained"); _style(ax)
    prev = 0.5
    for i, r in LD.iterrows():
        col = C_BASE if i == 0 else (C_REFIT if r.step in ("+ sprint speed", "+ lead at first move", "+ catcher pop time") else C_OURS)
        ax.bar(i, max(r.auroc - prev, 0.0005) if i else 0.0005, bottom=prev if i else 0.5, color=col, width=0.62)
        ax.text(i, r.auroc + 0.004, f"{r.auroc:.3f}" + (f"\n({r.gain:+.3f})" if i else ""), ha="center", fontsize=8, color=INK)
        prev = r.auroc
    ax.set_xticks(range(len(LD)))
    ax.set_xticklabels([f"{r.step}\n" + textwrap.fill(r.why, 22) for r in LD.itertuples()], fontsize=7.6)
    ax.set_ylim(0.5, 0.82); ax.set_ylabel("AUROC, out-of-fold")
    ax.legend(handles=[Patch(color=C_REFIT, label="inputs public models already use"),
                       Patch(color=C_OURS, label="inputs this project adds")], frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Why each input is in the model: what it adds, on the same attempts", loc="left", fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_cmp_ladder.png"), facecolor=SURF); plt.close(fig)

    # 6. scenarios: real runners, what each model expected vs what happened
    if len(SC):
        fig, ax = plt.subplots(figsize=(10.5, 3.9), dpi=170, facecolor=SURF, layout="constrained"); _style(ax)
        keys = [("Sprint speed only", C_REFIT), ("Powers spec refit", C_PUB), ("StealIQ v16", C_OURS), ("actual", INK)]
        w = 0.2
        for j, (k, c) in enumerate(keys):
            ax.bar(np.arange(len(SC)) + (j - 1.5) * w, 100 * SC[k], width=w, color=c, label=k if k != "actual" else "what actually happened")
        for i, r in SC.iterrows():
            ax.text(i, 101.5, f"{r.sprint_speed:.1f} ft/s · gains {r.gain:.1f} ft · n={int(r.attempts)}", ha="center", fontsize=7.5, color=MUTED)
        ax.set_xticks(range(len(SC))); ax.set_xticklabels(SC.runner, fontsize=8.5)
        ax.set_ylim(60, 104); ax.set_ylabel("P(safe) on his average attempt (%)")
        ax.legend(frameon=False, fontsize=7.8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title("Four real runners: v16 sees the slow runner's big jump, but under-rates elite runners with modest jumps",
                     loc="left", fontweight="bold", color=INK, fontsize=9.5)
        fig.savefig(fig_path("Fig_cmp_scenarios.png"), facecolor=SURF); plt.close(fig)

    # 7. the mediation path: why speed looks weak in public models
    sup = pd.read_csv(res_path("DF_add_speed_suppression.csv")); med = sup[sup.section == "mediation"].set_index("model")
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=170, facecolor=SURF, layout="constrained"); ax.set_facecolor(SURF); ax.axis("off")
    box = dict(boxstyle="round,pad=0.5", fc="white", ec=MUTED, lw=1)
    for xy, t in [((0.08, 0.35), "Sprint speed"), ((0.5, 0.82), "Ground gained\nby release"), ((0.92, 0.35), "Safe?")]:
        ax.text(*xy, t, ha="center", va="center", fontsize=10, bbox=box, color=INK)
    arrow = dict(arrowstyle="-|>", lw=1.6)
    ax.annotate("", xy=(0.84, 0.35), xytext=(0.16, 0.35), arrowprops=dict(**arrow, color=C_OURS))
    ax.annotate("", xy=(0.42, 0.76), xytext=(0.12, 0.45), arrowprops=dict(**arrow, color=C_REFIT))
    ax.annotate("", xy=(0.88, 0.45), xytext=(0.58, 0.76), arrowprops=dict(**arrow, color=C_OURS))
    ax.text(0.5, 0.28, f"direct: {med.loc['direct'].estimate:+.2f} log-odds per ft/s", ha="center", fontsize=9, color=C_OURS)
    ax.text(0.2, 0.66, f"faster → {med.loc['a'].estimate:+.2f} ft\nless ground per ft/s", ha="center", fontsize=8.5, color=C_REFIT)
    ax.text(0.8, 0.66, "more ground →\nsafer", ha="center", fontsize=8.5, color=C_OURS)
    ax.text(0.5, 0.06, f"Leave ground gained out (as public data must) and the two paths cancel: speed shows "
                       f"{med.loc['total'].estimate:+.2f} instead of {med.loc['direct'].estimate:+.2f} "
                       f"({med.loc['share'].estimate:.0%} hidden)", ha="center", fontsize=8.5, color=INK)
    ax.set_title("Why speed looks weak in public models", loc="left", fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_cmp_mediation.png"), facecolor=SURF); plt.close(fig)

    # 8. the leakage timeline: what is known when
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=170, facecolor=SURF, layout="constrained"); ax.set_facecolor(SURF); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ev = [(0.04, "season\ntraits"), (0.24, "pitcher's\nfirst move"), (0.46, "release"), (0.66, "catch /\nthrow"), (0.80, "tag:\nSAFE or OUT"), (0.95, "end of plate\nappearance")]
    ax.plot([0.02, 0.98], [0.5, 0.5], color=MUTED, lw=1.2)
    for x, t in ev:
        ax.plot(x, 0.5, "o", color=INK, ms=5); ax.text(x, 0.36, t, ha="center", va="top", fontsize=8, color=INK)
    ax.axvspan(0.80, 0.99, ymin=0.15, ymax=0.95, color="#fbe3dc", lw=0)
    used = [(0.04, "sprint speed · pop time"), (0.24, "lead at first move · base"), (0.46, "ground gained · pitch type")]
    for x, t in used:
        ax.text(x, 0.66, t, ha="center", fontsize=8, color=C_OURS, fontweight="bold")
    ax.text(0.895, 0.72, "BANNED (known after the outcome):\ncount after the pitch · end-of-PA score\n· run value · result",
            ha="center", fontsize=7.6, color="#b2361c")
    ax.text(0.02, 0.92, "Every model input must be known before the tag. Anything recorded after it leaks the answer.",
            fontsize=9, color=INK)
    ax.set_title("The leakage rule, on the timeline of one steal", loc="left", fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_cmp_timeline.png"), facecolor=SURF); plt.close(fig)


def runner_level(d, P, y, min_att=15):
    """Every runner-season with >= min_att attempts: mean out-of-fold P(safe) vs his actual success rate."""
    keys = {"Sprint speed only": ("baseline", "Sprint speed only"),
            "Powers spec refit": ("refit", "Powers spec refit here (lead, speed, arm)"),
            "Pitcher's Dilemma spec refit": ("refit", "Pitcher's Dilemma spec refit here (lead, speed, jump, pop)"),
            "StealIQ v16 + isotonic": ("ours", "StealIQ v16 + isotonic (recommended)")}
    t = d[["runner_id", "season"]].assign(y=y, **{k: P[v] for k, v in keys.items()})
    g = t.groupby(["runner_id", "season"]).agg(n=("y", "size"), actual=("y", "mean"), **{k: (k, "mean") for k in keys})
    g = g[g.n >= min_att].reset_index()
    summ = pd.DataFrame([dict(model=k, runner_seasons=len(g), r=np.corrcoef(g[k], g.actual)[0, 1],
                              mae_pts=100 * (g[k] - g.actual).abs().mean(), sd_pred_pts=100 * g[k].std(),
                              sd_actual_pts=100 * g.actual.std()) for k in keys])
    return g, summ


def fig_runner_level(g, summ):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=170, facecolor=SURF, layout="constrained", sharey=True)
    for ax, k, c in zip(axs, ["Sprint speed only", "Pitcher's Dilemma spec refit", "StealIQ v16 + isotonic"],
                        [C_REFIT, C_PUB, C_OURS]):
        _style(ax); r = summ.set_index("model").loc[k]
        ax.scatter(100 * g[k], 100 * g.actual, s=10 + g.n / 2, color=c, alpha=0.55, lw=0)
        ax.plot([55, 100], [55, 100], color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(55, 100); ax.set_ylim(40, 101); ax.set_xlabel("model's average P(safe) for him (%)")
        ax.set_title(f"{k}\nr = {r.r:.2f} · mean error {r.mae_pts:.1f} pts · spread {r.sd_pred_pts:.1f} pts",
                     loc="left", fontsize=9, fontweight="bold", color=INK)
    axs[0].set_ylabel("his actual success rate (%)")
    fig.suptitle(f"Runner-seasons with 15+ attempts (n = {int(summ.runner_seasons.iloc[0])}): does the model know who is good?",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_cmp_runner_level.png"), facecolor=SURF); plt.close(fig)


def run_baselines():
    d = baseline_rows(); P, y = baseline_models(d)
    S = baseline_score(P, y, d); S.to_csv(res_path("DF_cmp_baselines.csv"), index=False)
    LD = baseline_ladder(d); LD.to_csv(res_path("DF_cmp_ladder.csv"), index=False)
    SE = baseline_seasons(d); SE.to_csv(res_path("DF_cmp_seasons.csv"), index=False)
    SC = baseline_scenarios(d, P); SC.to_csv(res_path("DF_cmp_scenarios.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"rows {len(d)}; jump-speed splits coverage {d.jump_covered.mean():.0%}")
    print(S[["family", "model", "auroc", "auroc_lo", "auroc_hi", "log_loss", "ece", "bad_deciles", "mean_pred"]].round(4).to_string())
    print(LD[["step", "auroc", "gain"]].round(4).to_string()); print(SE.round(3).to_string()); print(SC.round(3).to_string())
    G, GS = runner_level(d, P, y); GS.to_csv(res_path("DF_cmp_runner_level.csv"), index=False)
    print(GS.round(3).to_string())
    baseline_figures(S, LD, SE, SC, P, y, d); fig_runner_level(G, GS)


def speed_by_season(min_att=10, bin_w=0.5):
    """'Sprint speed is significant in the steal model' (Powers et al.) — true, and small. Season by season on the shipped
    calculator's rows: SB and CS counts in 0.5 ft/s sprint-speed bins (how often each speed runs, and is caught); the
    success rate per bin with Wilson 95% CIs; speed alone as a logistic (runner-clustered SE: significant at this n) next to
    what it is worth (10th -> 90th percentile swing in P(safe), AUROC) and the same for ground gained; and the runner-level
    correlation of speed with success rate (runner-seasons with min_att+ attempts)."""
    at = shipped_rows()
    edges = np.arange(np.floor(at.sprint_speed.min() / bin_w) * bin_w, at.sprint_speed.max() + bin_w, bin_w)
    rows, cells = [], []
    for season, d in [*at.groupby("season"), ("2023-26", at)]:
        out = dict(season=str(season), attempts=len(d), sb=int(d.y.sum()), cs=int(len(d) - d.y.sum()), success=d.y.mean())
        for col, tag in [("sprint_speed", "speed"), ("gain_to_release_ft", "ground")]:
            f = sm.Logit(d.y, sm.add_constant(d[col])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.runner_id})
            b0, b, se = f.params.iloc[0], f.params.iloc[1], f.bse.iloc[1]
            q10, q90 = d[col].quantile([0.1, 0.9])
            pr = lambda v: 1 / (1 + np.exp(-(b0 + b * v)))
            out.update({f"{tag}_beta": b, f"{tag}_ci_lo": b - 1.96 * se, f"{tag}_ci_hi": b + 1.96 * se, f"{tag}_p": f.pvalues.iloc[1],
                        f"{tag}_p10": q10, f"{tag}_p90": q90, f"{tag}_swing_pp": 100 * (pr(q90) - pr(q10)),
                        f"{tag}_auroc": roc_auc_score(d.y, d[col])})
        rs = d.groupby(["runner_id", "season"]).agg(n=("y", "size"), safe=("y", "mean"), speed=("sprint_speed", "mean"))
        rs = rs[rs.n >= min_att]
        r = np.corrcoef(rs.speed, rs.safe)[0, 1]; z, h = np.arctanh(r), 1.96 / np.sqrt(len(rs) - 3)
        out.update(runner_seasons=len(rs), r_runner=r, r_lo=np.tanh(z - h), r_hi=np.tanh(z + h))
        rows.append(out)
        if season != "2023-26":
            c = pd.cut(d.sprint_speed, edges)
            for iv, g in d.groupby(c, observed=True):
                n, k = len(g), int(g.y.sum()); ph = k / n; zz = 1.96
                mid, half = (ph + zz**2 / (2 * n)) / (1 + zz**2 / n), zz * np.sqrt(ph * (1 - ph) / n + zz**2 / (4 * n**2)) / (1 + zz**2 / n)
                cells.append(dict(season=season, bin_lo=iv.left, bin_hi=iv.right, attempts=n, sb=k, cs=n - k, runners=g.runner_id.nunique(),
                                  success=ph, ci_lo=mid - half, ci_hi=mid + half))
    S, C = pd.DataFrame(rows), pd.DataFrame(cells)
    S.to_csv(res_path("DF_cmp_speed_by_season.csv"), index=False); C.to_csv(res_path("DF_cmp_speed_bins.csv"), index=False)
    pd.set_option("display.width", 220)
    print(S[["season", "attempts", "success", "speed_beta", "speed_p", "speed_swing_pp", "speed_auroc", "ground_swing_pp",
             "ground_auroc", "runner_seasons", "r_runner", "r_lo", "r_hi"]].round(4).to_string(index=False))
    fig_speed_by_season(S, C, bin_w)
    return S, C


def fig_speed_by_season(S, C, bin_w):
    """One row per season (sized to read at page width in the report): SB and CS counts overlaid per bin with the runners
    in each bin, then the share safe per bin with its Wilson CI and the season's speed and ground-gained effects."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seasons = sorted(C.season.unique())
    fig, axs = plt.subplots(len(seasons), 2, figsize=(9.2, 2.5 * len(seasons) + 0.7), dpi=170, facecolor=SURF,
                            layout="constrained", sharex=True, sharey="col")
    for i, s in enumerate(seasons):
        c, r = C[C.season == s], S.set_index("season").loc[str(s)]
        x = (c.bin_lo + c.bin_hi) / 2
        ax = axs[i, 0]; _style(ax)
        ax.bar(x, c.sb, width=bin_w * 0.9, color=C_OURS, label="stolen base")
        ax.bar(x, c.cs, width=bin_w * 0.5, color=C_REFIT, label="caught stealing")
        for xi, sb, nr in zip(x, c.sb, c.runners):
            ax.text(xi, sb, f"{nr}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(f"{s}: {int(r.attempts):,} attempts, {r.success:.1%} safe", loc="left", fontsize=10, fontweight="bold", color=INK)
        ax.set_ylabel("attempts per bin")
        ax = axs[i, 1]; _style(ax)
        big = c.attempts >= 20
        ax.errorbar(x[big], c.success[big], yerr=[c.success[big] - c.ci_lo[big], c.ci_hi[big] - c.success[big]], fmt="o",
                    color=INK, ms=4, capsize=2, lw=0.9)
        ax.plot(x[~big], c.success[~big], "o", mfc="none", color=MUTED, ms=4)
        ax.axhline(r.success, color=MUTED, lw=0.8, ls="--")
        ax.set_ylim(0.3, 1.03); ax.set_ylabel("share safe")
        sig = "p < 0.001" if r.speed_p < 0.001 else f"p = {r.speed_p:.3f}"
        ax.text(0.02, 0.04, f"speed alone: {r.speed_swing_pp:+.1f} pts from 10th to 90th pct ({sig}), AUROC {r.speed_auroc:.2f}\n"
                            f"ground gained: {r.ground_swing_pp:+.1f} pts, AUROC {r.ground_auroc:.2f}\n"
                            f"runner-level r = {r.r_runner:+.2f} (CI {r.r_lo:+.2f} to {r.r_hi:+.2f}; {int(r.runner_seasons)} runners, 10+ attempts)",
                transform=ax.transAxes, fontsize=7.8, color=INK, va="bottom")
    for ax in axs[-1]:
        ax.set_xlabel("runner sprint speed (ft/s), 0.5 ft/s bins")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    axs[0, 1].set_title("95% Wilson CI per bin (hollow: under 20 attempts)\ndashed line: the season's rate", loc="left",
                        fontsize=8.5, color=MUTED)
    per = S[S.season != "2023-26"]
    fig.suptitle(f"Sprint speed vs steal success, season by season (numbers above bars: runners in the bin)\n"
                 f"Slow to fast runners (10th→90th percentile of speed): share safe +{per.speed_swing_pp.min():.0f} to "
                 f"+{per.speed_swing_pp.max():.0f} points\nThe same range of ground gained: +{per.ground_swing_pp.min():.0f} to "
                 f"+{per.ground_swing_pp.max():.0f} points",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_cmp_speed_by_season.png"), facecolor=SURF); plt.close(fig)


def run_speed():
    speed_by_season()


# ════ 11. Runner jump speed vs pitcher time (CV-timed attempts) ════════════════════════════════════════════════
# Splits ground gained into its runner part and its pitcher part.
#
#     ground gained by release (ft)  =  runner's jump speed (ft/s)  ×  time the runner has (s)
#
# The time comes from the CV-timed pitcher delivery (lead-foot lift-off -> release; data/delivery/delivery_<season>.csv), with two
# corrections, both estimated from data rather than assumed:
#   bias   the CV reads the delivery short against the hand-labelled gold clips (data/delivery/delivery_gold.csv)
#   delta  Statcast starts counting ground gained at the pitcher's 'first move', which can come before the lift-off the
#          CV sees. If a runner moves at a steady v over the window, ground = v·(T + delta), so the intercept / slope of
#          ground on T estimates delta (runner-clustered bootstrap CI).
# Jump speed contains ground gained (it is the numerator), so it is never tested on the attempt it was measured on.
# The fair tests:
#   1 speed-neutrality     correlation with sprint speed, attempt- and runner-level (vs ground gained)
#   2 decomposition        how much of the spread in log ground gained is runner jump speed vs pitcher time
#   3 repeatability        split-half reliability within a season (alternate attempts per runner, Spearman-Brown),
#                          and season to season (first timed season vs each later one, runners with 3+ in both)
#   4 out-of-sample value  a runner's jump speed from his OTHER attempts predicting this attempt: on top of every input
#                          except this attempt's ground gained, and on top of the full v16 model; compared with the same
#                          leave-one-out profile of ground gained and with The Pitcher's Dilemma's estimated jump speed
#   5 runners              the busiest runners: is their ground their own jump, or slow pitchers?

C_RUN, C_PIT, C_GAIN, C_PD = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
V_MIN, V_MAX = 3.0, 30.0                 # plausible jump speeds (ft/s); outside = a CV timing error, flagged and excluded
K_SHRINK = 5                             # leave-one-out profiles shrink toward the league mean with this many pseudo-attempts


def cv_bias() -> float:
    """Mean (gold − CV) delivery time on gold clips the production design passed: positive = CV reads short."""
    g = pd.read_csv(DELIVERY / "delivery_gold.csv")
    g = g[(g.qa == "PASS") & g.gold_lift.notna()]
    return float(((g.gold_release - g.gold_lift) / g.fps - g.delivery_s).mean())


def pd_per_attempt(d: pd.DataFrame) -> pd.Series:
    """The Pitcher's Dilemma jump speed for each attempt (power law on the runner-season's running splits)."""
    s = pd.read_csv(RAW / "Raw_Season.csv"); cols = [f"seconds_since_hit_{x:03d}" for x in SPLIT_D]; fit = {}
    for r in s.dropna(subset=cols).itertuples(index=False):
        b, loga = np.polyfit(np.log(SPLIT_D), np.log(np.array([getattr(r, c) for c in cols], float)), 1)
        fit[(r.runner_id, r.season)] = (np.exp(loga), b)
    ab = np.array([fit.get((a, b), (np.nan, np.nan)) for a, b in zip(d.runner_id, d.season)])
    g = d.gain_to_release_ft.clip(lower=0.5).values
    return pd.Series(g / (ab[:, 0] * g ** ab[:, 1]), index=d.index)


def jump_rows() -> pd.DataFrame:
    d = baseline_rows().rename(columns={"jump_speed": "pd_jump"})                # as published: includes this attempt
    d["pd_jump_attempt"] = pd_per_attempt(d)
    cvs = [load_delivery(s) for s in timed_seasons()]                    # a timing run still in progress is skipped
    cvs = [c[c.qa == "PASS"][["play_id", "delivery_s"]] for c in cvs]
    t = d.merge(pd.concat(cvs).drop_duplicates("play_id"), on="play_id", how="inner")
    return t.reset_index(drop=True)


def estimate_delta(t, bias, n_boot=1000, seed=0):
    T = t.delivery_s + bias
    def fit(g, T_):
        b, a = np.polyfit(T_, g, 1); return a / b, b, a
    delta, slope, icpt = fit(t.gain_to_release_ft.values, T.values)
    rng, grp = np.random.default_rng(seed), t.groupby("runner_id").indices; keys = np.array(list(grp)); bs = []
    for _ in range(n_boot):
        i = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        bs.append(fit(t.gain_to_release_ft.values[i], T.values[i])[0])
    return delta, slope, icpt, np.percentile(bs, [2.5, 97.5])


def split_half(t, col, min_att=4, n_boot=1000, seed=0):
    """Alternate attempts (in date order) per runner into two halves; correlate runner means; Spearman-Brown."""
    s = t.sort_values("date").assign(half=lambda x: x.groupby("runner_id").cumcount() % 2)
    n = s.groupby("runner_id").size(); s = s[s.runner_id.map(n) >= min_att]
    h = s.groupby(["runner_id", "half"])[col].mean().unstack().dropna()
    sb = lambda r: 2 * r / (1 + r)
    r = np.corrcoef(h[0], h[1])[0, 1]
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(h), len(h)); bs.append(sb(np.corrcoef(h[0].values[i], h[1].values[i])[0, 1]))
    return sb(r), np.percentile(bs, [2.5, 97.5]), len(h)


def loo(t, col):
    """Runner's shrunk mean of `col` over his OTHER attempts (NaN if he has fewer than 2 others)."""
    g = t.groupby("runner_id")[col]
    s, n = g.transform("sum"), g.transform("count")
    league = t[col].mean()
    v = (s - t[col] + K_SHRINK * league) / (n - 1 + K_SHRINK)
    return v.where(n - 1 >= 2)


def oos(t, n_boot=1000, seed=0):
    t = t.assign(loo_jump=loo(t, "jump_v"), loo_gain=loo(t, "gain_to_release_ft"), loo_pd=loo(t, "pd_jump_attempt"))
    t["loo_pd"] = t.loo_pd.fillna(t.pd_jump_attempt.median())                # runners without splits: league median
    t = t.dropna(subset=["loo_jump"]).reset_index(drop=True)
    y = t.y.values; cv = StratifiedKFold(5, shuffle=True, random_state=42)
    ctx = ["sprint_speed", "lead_at_firstmove_ft", "pop_faced", "pt_breaking", "pt_offspeed", "base_is_3b"]
    oof = lambda cols: cross_val_predict(pipe(DEFAULT), t[cols].values.astype(float), y, cv=cv, method="predict_proba")[:, 1]
    specs = {"context (no ground gained)": ctx,
             "context + runner's jump speed (other attempts)": ctx + ["loo_jump"],
             "context + runner's ground gained (other attempts)": ctx + ["loo_gain"],
             "context + Pitcher's Dilemma estimated jump speed": ctx + ["pd_jump"],
             "context + Pitcher's Dilemma estimate (other attempts)": ctx + ["loo_pd"],
             "full v16 (this attempt's ground gained)": FEATS,
             "full v16 + runner's jump speed (other attempts)": FEATS + ["loo_jump"]}
    P = {k: oof(c) for k, c in specs.items()}
    pairs = {"context + runner's jump speed (other attempts)": "context (no ground gained)",
             "context + runner's ground gained (other attempts)": "context (no ground gained)",
             "context + Pitcher's Dilemma estimated jump speed": "context (no ground gained)",
             "context + Pitcher's Dilemma estimate (other attempts)": "context (no ground gained)",
             "full v16 + runner's jump speed (other attempts)": "full v16 (this attempt's ground gained)"}
    rng, grp = np.random.default_rng(seed), t.groupby("runner_id").indices; keys = np.array(list(grp))
    D = {k: [] for k in pairs}
    for _ in range(n_boot):
        i = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        for a, b in pairs.items():
            D[a].append(roc_auc_score(y[i], P[a][i]) - roc_auc_score(y[i], P[b][i]))
    out = []
    for k, p in P.items():
        r = dict(test="out_of_sample", item=k, n=len(t), value=roc_auc_score(y, p))
        if k in pairs:
            r.update(gain=r["value"] - roc_auc_score(y, P[pairs[k]]), lo=np.percentile(D[k], 2.5), hi=np.percentile(D[k], 97.5))
        out.append(r)
    return out


def cross_season(t, min_att=3, n_boot=1000, seed=0):
    """Runner means in the first timed season vs each later one (runners with min_att+ attempts in both)."""
    out, ss = [], sorted(t.season.unique())
    for s2 in ss[1:]:
        m = t[t.season.isin([ss[0], s2])].groupby(["runner_id", "season"]).agg(
            n=("y", "size"), jump_v=("jump_v", "mean"), gain=("gain_to_release_ft", "mean"), T=("T_eff", "mean"), y=("y", "mean"))
        m = m[m.n >= min_att].unstack("season").dropna()
        if len(m) < 10:
            continue
        rng = np.random.default_rng(seed)
        for col, lab in [("jump_v", "runner jump speed"), ("gain", "ground gained"), ("T", "pitcher time faced"), ("y", "success (safe = 1)")]:
            a, b = m[(col, ss[0])].values, m[(col, s2)].values
            bs = [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
            out.append(dict(test="cross_season", item=f"{ss[0]}→{s2}: {lab}", n=len(m), value=np.corrcoef(a, b)[0, 1],
                            lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
            print(f"cross-season {out[-1]['item']}: r = {out[-1]['value']:+.3f} (n = {len(m)} runners)")
    return out


def run_jump():
    t = jump_rows(); bias = cv_bias()
    delta, slope, icpt, dci = estimate_delta(t, bias)
    t["T_eff"] = t.delivery_s + bias + delta
    t["jump_v"] = t.gain_to_release_ft / t.T_eff
    t["flag"] = ~t.jump_v.between(V_MIN, V_MAX)
    print(f"timed attempts on model rows: {len(t)} (seasons {t.season.value_counts().sort_index().to_dict()}); "
          f"CV bias {bias * 1000:+.0f} ms; delta {delta:+.3f} s (95% CI {dci[0]:+.3f} to {dci[1]:+.3f}); "
          f"steady-speed slope {slope:.2f} ft/s; flagged {int(t.flag.sum())}")
    t.drop(columns=[]).to_csv(res_path("DF_add_jump_metric.csv"), index=False,
                              columns=["play_id", "runner_id", "season", "date", "pitcher_id", "y", "delivery_s", "T_eff",
                                       "gain_to_release_ft", "jump_v", "flag", "sprint_speed", "pd_jump", "pd_jump_attempt"])
    u = t[~t.flag].reset_index(drop=True)
    R = [dict(test="setup", item="CV delivery bias (s, gold − CV)", value=bias),
         dict(test="setup", item="first-move to lift-off offset delta (s)", value=delta, lo=dci[0], hi=dci[1]),
         dict(test="setup", item="steady-speed slope (ft/s)", value=slope),
         dict(test="setup", item="attempts used / flagged", value=len(u), n=int(t.flag.sum())),
         dict(test="setup", item="median jump speed (ft/s)", value=u.jump_v.median(), lo=u.jump_v.quantile(.05), hi=u.jump_v.quantile(.95))]
    # 1 speed-neutrality
    rl = u.groupby("runner_id").agg(n=("y", "size"), jump_v=("jump_v", "mean"), gain=("gain_to_release_ft", "mean"),
                                    speed=("sprint_speed", "mean"), T=("T_eff", "mean"), safe=("y", "mean"))
    rl3 = rl[rl.n >= 3]
    for nm, a, b, df in [("attempt: jump speed vs sprint speed", "jump_v", "sprint_speed", u),
                         ("attempt: ground gained vs sprint speed", "gain_to_release_ft", "sprint_speed", u),
                         ("runner (3+): jump speed vs sprint speed", "jump_v", "speed", rl3),
                         ("runner (3+): ground gained vs sprint speed", "gain", "speed", rl3),
                         ("attempt: jump speed vs Pitcher's Dilemma estimate", "jump_v", "pd_jump", u),
                         ("attempt: jump speed vs ground gained", "jump_v", "gain_to_release_ft", u)]:
        R.append(dict(test="correlation", item=nm, n=len(df), value=np.corrcoef(df[a], df[b])[0, 1]))
    # 2 decomposition of log ground gained
    lg, lv, lT = np.log(u.gain_to_release_ft.clip(lower=0.5)), np.log(u.jump_v), np.log(u.T_eff)
    var, cov = lg.var(), np.cov(lv, lT)[0, 1]
    run_sh, pit_sh = (lv.var() + cov) / var, (lT.var() + cov) / var
    R += [dict(test="decomposition", item="runner jump speed share of log ground-gained variance", value=run_sh),
          dict(test="decomposition", item="pitcher time share of log ground-gained variance", value=pit_sh),
          dict(test="decomposition", item="corr(log jump speed, log time)", value=np.corrcoef(lv, lT)[0, 1])]
    # 3 repeatability
    for col, lab in [("jump_v", "runner jump speed"), ("gain_to_release_ft", "ground gained"), ("T_eff", "pitcher time faced"),
                     ("y", "success (safe = 1)")]:
        rel, ci, nr = split_half(u, col)
        R.append(dict(test="split_half_reliability", item=lab, n=nr, value=rel, lo=ci[0], hi=ci[1]))
    R += cross_season(u)
    # 4 out-of-sample value
    R += oos(u)
    T = pd.DataFrame(R); T.to_csv(res_path("DF_add_jump_tests.csv"), index=False)
    pd.set_option("display.width", 200); print(T.round(4).to_string())
    # 5 runners
    lv_m, lT_m = lv.mean(), lT.mean()
    rr = u.assign(lv=lv, lT=lT).groupby("runner_id").agg(n=("y", "size"), safe=("y", "mean"), gain=("gain_to_release_ft", "mean"),
                                                         jump_v=("jump_v", "mean"), T=("T_eff", "mean"), speed=("sprint_speed", "mean"),
                                                         lv=("lv", "mean"), lT=("lT", "mean"))
    names = pd.concat([pd.read_csv(RAW / "Raw_Season.csv")[["runner_id", "player_name"]],
                       pd.read_csv(res_path("DF_v15_leaderboard.csv"))[["runner_id", "player_name"]],
                       pd.read_csv(RAW / "people.csv").rename(columns={"player_id": "runner_id", "name": "player_name"})]) \
              .dropna().drop_duplicates("runner_id").set_index("runner_id").player_name
    rr["name"] = [names.get(i, f"runner {i}") for i in rr.index]; rr["runner_part_pct"] = 100 * (np.exp(rr.lv - lv_m) - 1)
    rr["pitcher_part_pct"] = 100 * (np.exp(rr.lT - lT_m) - 1)
    top = rr.sort_values("n", ascending=False).head(14)
    top.to_csv(res_path("DF_add_jump_runners.csv"))
    print(top[["name", "n", "safe", "gain", "jump_v", "T", "speed", "runner_part_pct", "pitcher_part_pct"]].round(2).to_string())
    jump_figs(u, T, rl3, top, delta, dci, slope, bias)


def jump_figs(u, T, rl3, top, delta, dci, slope, bias):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d6d5d0", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
    style = lambda ax: (ax.set_facecolor(SURF), ax.spines[["top", "right"]].set_visible(False), ax.grid(color=GRID, lw=0.6), ax.set_axisbelow(True))
    g = T.set_index("item")
    # decomposition
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=170, facecolor=SURF, layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]; style(ax)
    ax.scatter(u.T_eff, u.gain_to_release_ft, s=6, alpha=0.25, color=MUTED, lw=0)
    x = np.linspace(u.T_eff.quantile(.01), u.T_eff.quantile(.99), 50)
    for v, c in [(8, C_PIT), (11, C_GAIN), (14, C_RUN)]:
        ax.plot(x, v * x, color=c, lw=1.6); ax.text(x[-1], v * x[-1], f" {v} ft/s", color=c, fontsize=8, va="center")
    ax.set_xlabel(f"time the runner has (s) = CV delivery + {bias * 1000:.0f} ms bias + δ = {delta:+.2f} s (95% CI {dci[0]:+.2f} to {dci[1]:+.2f})")
    ax.set_ylabel("ground gained by release (ft)")
    ax.set_title("Ground gained = jump speed × time: each line is one steady jump speed", loc="left", fontweight="bold", color=INK)
    ax = axs[1]; style(ax)
    sh = [g.loc["runner jump speed share of log ground-gained variance"].value, g.loc["pitcher time share of log ground-gained variance"].value]
    ax.bar(["runner's\njump speed", "pitcher's\ndelivery time"], [100 * v for v in sh], color=[C_RUN, C_PIT], width=0.55)
    for i, v in enumerate(sh):
        ax.text(i, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_ylabel("share of the spread in ground gained (%)"); ax.set_ylim(0, 110)
    ax.set_title("Who supplies the ground?", loc="left", fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_add_jump_decomp.png"), facecolor=SURF); plt.close(fig)
    # tests
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=170, facecolor=SURF, layout="constrained")
    ax = axs[0]; style(ax)
    ax.scatter(rl3.speed, rl3.jump_v, s=8 + rl3.n, color=C_RUN, alpha=0.6, lw=0)
    r = g.loc["runner (3+): jump speed vs sprint speed"].value
    ax.set_xlabel("sprint speed (ft/s)"); ax.set_ylabel("runner's mean jump speed (ft/s)")
    ax.set_title(f"Jump speed is not sprint speed\n(runners, r = {r:+.2f})", loc="left", fontweight="bold", color=INK)
    ax = axs[1]; style(ax)
    rel = T[T.test == "split_half_reliability"].reset_index(drop=True)
    cols = [C_RUN, C_GAIN, C_PIT, C_PD]
    for i, rw in rel.iterrows():
        ax.bar(i, rw.value, color=cols[i], width=0.6)
        ax.errorbar(i, rw.value, yerr=[[rw.value - rw.lo], [rw.hi - rw.value]], fmt="none", ecolor=INK, capsize=3)
        ax.text(i, max(rw.hi, 0) + 0.03, f"{rw.value:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(rel))); ax.set_xticklabels([s.replace(" ", "\n", 1) for s in rel["item"]], fontsize=8)
    ax.set_ylabel("split-half reliability (Spearman-Brown)"); ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"Is it a stable skill?\n({int(rel.n.iloc[0])} runners with 4+ attempts)", loc="left", fontweight="bold", color=INK)
    ax = axs[2]; style(ax); ax.grid(axis="y", visible=False)
    o = T[(T.test == "out_of_sample") & T.gain.notna()].iloc[::-1].reset_index(drop=True)
    lab = {"context + runner's jump speed (other attempts)": "his jump speed (other attempts)",
           "context + runner's ground gained (other attempts)": "his ground gained (other attempts)",
           "context + Pitcher's Dilemma estimated jump speed": "Pitcher's Dilemma (as published)",
           "context + Pitcher's Dilemma estimate (other attempts)": "Pitcher's Dilemma (other attempts)",
           "full v16 + runner's jump speed (other attempts)": "his jump speed, on top of full v16"}
    for i, rw in o.iterrows():
        c = C_RUN if "jump speed (other" in rw["item"] else (C_GAIN if "ground" in rw["item"] else C_PD)
        ax.barh(i, rw.gain, color=c, height=0.6)
        ax.errorbar(rw.gain, i, xerr=[[rw.gain - rw.lo], [rw.hi - rw.gain]], fmt="none", ecolor=INK, capsize=3)
        ax.text(rw.hi + 0.001, i, f"{rw.gain:+.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([lab[s] for s in o["item"]], fontsize=7.8)
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlabel("AUROC gain over context (95% CI, runner-clustered)")
    ax.set_title("Does it predict his next steal?\n(out of sample)", loc="left", fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_add_jump_tests.png"), facecolor=SURF); plt.close(fig)
    # runners
    t2 = top.sort_values("gain")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=170, facecolor=SURF, layout="constrained"); style(ax); ax.grid(axis="y", visible=False)
    y = np.arange(len(t2))
    ax.barh(y - 0.18, t2.runner_part_pct, height=0.34, color=C_RUN, label="his jump speed vs league (%)")
    ax.barh(y + 0.18, t2.pitcher_part_pct, height=0.34, color=C_PIT, label="time the pitchers gave him vs league (%)")
    ax.set_yticks(y); ax.set_yticklabels([f"{r.name} ({r.gain:.1f} ft, {r.speed:.1f} ft/s, n={int(r.n)})" for r in t2.itertuples()], fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlabel("percent above (+) or below (−) the league, on a log scale")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Busiest runners: is their ground their own jump, or slow pitchers?", loc="left", fontweight="bold", color=INK)
    fig.savefig(fig_path("Fig_add_jump_runners.png"), facecolor=SURF); plt.close(fig)


# ════ 12. Option (b): the pitcher's delivery on earlier dates as a pre-pitch input ═════════════════════════════
# Option (b): does a pitcher's delivery time from his EARLIER timed attempts predict the next steal?
#
# This is the version a dugout could actually use: for each attempt, the pitcher's mean CV delivery time
# (lead-foot lift-off -> release, delivery_<season>.csv, PASS clips) over attempts on strictly earlier dates only.
# No same-day and no later clip ever enters the profile (§4's same-season check used leave-one-out,
# which lets later clips in).
#
#   profiles   prior delivery   mean CV delivery of his earlier timed attempts (same season; >= MIN prior clips)
#              prior ground     mean ground gained he allowed on his earlier attempts (Statcast, no CV) — control
#   samples    all attempts     every shipped attempt whose pitcher has >= MIN prior timed clips
#              own clip timed   the subset whose own clip also passed, so the oracle (this attempt's delivery) and
#                               the post-pitch ceiling (this attempt's ground gained) are on the same rows
#   models     P0 speed + first-move lead + pop (pre-pitch base); + prior delivery; + prior ground; + own delivery;
#              + both priors; + own ground gained
# Seasons still being timed are used up to their last timed date (the run goes in date order, so the prefix is
# complete). With two or more seasons timed, an 'all earlier' variant also lets earlier seasons into the profile.
# Out-of-fold AUROC (5-fold), paired pitcher-clustered bootstrap CIs for each gain over P0.

C_ORC = "#a3a29c"                      # known only after the pitch (oracle / ceiling bars)


def prior_timed() -> pd.DataFrame:
    c = pd.concat([pd.read_csv(f) for f in sorted(DELIVERY.glob("delivery_20[0-9][0-9].csv"))]).drop_duplicates("play_id")
    return c[["play_id", "qa", "delivery_s"]]


def prior_mean(rows, src, val, key, min_n):
    """Mean of src[val] over src rows with the same `key` on dates strictly before each row's date."""
    day = src.groupby(key + ["date"])[val].agg(["sum", "count"]).groupby(level=key).cumsum().reset_index()
    m = pd.merge_asof(rows[["date"] + key].reset_index().sort_values("date"), day.sort_values("date"), on="date",
                      by=key, allow_exact_matches=False).set_index("index").reindex(rows.index)
    return (m["sum"] / m["count"]).where(m["count"] >= min_n), m["count"]


def prior_rows(key, min_n):
    at = shipped_rows(); at["date"] = pd.to_datetime(at.date)
    t = prior_timed().merge(at[["play_id", "season", "date", "pitcher_id"]], on="play_id")
    last = t.groupby("season").date.max()                                   # timed prefix of each season
    at = at[at.season.isin(last.index) & (at.date <= at.season.map(last))].copy()
    ok = t[t.qa == "PASS"]
    at["prior_deliv"], at["n_prior"] = prior_mean(at, ok, "delivery_s", key, min_n)
    at["prior_gain"], _ = prior_mean(at, at, "gain_to_release_ft", key, min_n)
    at["own_deliv"] = at.play_id.map(ok.set_index("play_id").delivery_s)
    return at.dropna(subset=["prior_deliv", "prior_gain"] + PREPITCH_BASE).reset_index(drop=True)


def prior_evaluate(d, models, tag, n_boot=1000, seed=0):
    y = d.y.values; cv = StratifiedKFold(5, shuffle=True, random_state=42)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    oof = {m: cross_val_predict(lr(), d[c].values, y, cv=cv, method="predict_proba")[:, 1] for m, c in models.items()}
    auc = {m: roc_auc_score(y, p) for m, p in oof.items()}
    base = next(iter(models)); rng, grp = np.random.default_rng(seed), d.groupby("pitcher_id").indices
    keys = np.array(list(grp)); G = {m: [] for m in models if m != base}
    for _ in range(n_boot):
        i = np.concatenate([grp[k] for k in rng.choice(keys, len(keys))])
        if y[i].min() == y[i].max():
            continue
        a0 = roc_auc_score(y[i], oof[base][i])
        for m in G:
            G[m].append(roc_auc_score(y[i], oof[m][i]) - a0)
    out = []
    for m in models:
        g = np.array(G.get(m, [np.nan]))
        out.append(dict(sample=tag, model=m, n=len(d), n_pitchers=d.pitcher_id.nunique(), auroc=auc[m],
                        gain=auc[m] - auc[base], lo=np.nanpercentile(g, 2.5), hi=np.nanpercentile(g, 97.5)))
        print(f"[{tag}] {m:34s} n={len(d):5d} AUROC {auc[m]:.4f} gain {out[-1]['gain']:+.4f} "
              f"[{out[-1]['lo']:+.4f}, {out[-1]['hi']:+.4f}]")
    return out


def run_prior():
    rows, keep = [], {}
    seasons = prior_timed().merge(shipped_rows()[["play_id", "season"]], on="play_id").season.unique()
    variants = [("same season, 2+ prior", ["pitcher_id", "season"], 2), ("same season, 3+ prior", ["pitcher_id", "season"], 3)]
    if len(seasons) > 1:
        variants.append(("all earlier, 3+ prior", ["pitcher_id"], 3))
    for name, key, k in variants:
        d = prior_rows(key, k); keep[name] = d
        print(f"\n{name}: {len(d):,} attempts, {d.pitcher_id.nunique()} pitchers, seasons {sorted(d.season.unique())}")
        rows += prior_evaluate(d, {"P0 pre-pitch base": PREPITCH_BASE, "+ prior delivery (CV)": PREPITCH_BASE + ["prior_deliv"],
                             "+ prior ground allowed (no CV)": PREPITCH_BASE + ["prior_gain"],
                             "+ both priors": PREPITCH_BASE + ["prior_deliv", "prior_gain"]}, f"{name} | all attempts")
        o = d.dropna(subset=["own_deliv"]).reset_index(drop=True)
        rows += prior_evaluate(o, {"P0 pre-pitch base": PREPITCH_BASE, "+ prior delivery (CV)": PREPITCH_BASE + ["prior_deliv"],
                             "+ this attempt's delivery (oracle)": PREPITCH_BASE + ["own_deliv"],
                             "+ this attempt's ground gained": PREPITCH_BASE + ["gain_to_release_ft"]}, f"{name} | own clip timed")
        for a, b in [("prior_deliv", "own_deliv"), ("prior_deliv", "gain_to_release_ft"), ("own_deliv", "gain_to_release_ft")]:
            r = o[[a, b]].corr().iloc[0, 1]
            rows.append(dict(sample=f"{name} | own clip timed", model=f"corr {a} vs {b}", n=len(o), auroc=r))
            print(f"[{name}] corr {a} vs {b}: {r:+.3f} (n={len(o)})")
    R = pd.DataFrame(rows); R.to_csv(res_path("DF_add_prior_delivery.csv"), index=False)
    print(f"\nwrote {res_path('DF_add_prior_delivery.csv')}")
    prior_fig(R, keep[variants[0][0]], variants[0][0])


def prior_fig(R, d, name):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    style = lambda ax: (ax.set_facecolor(SURF), ax.spines[["top", "right"]].set_visible(False), ax.grid(color=GRID, lw=0.6), ax.set_axisbelow(True))
    f, axs = plt.subplots(1, 2, figsize=(12.5, 4.3), dpi=170, facecolor=SURF, layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})
    o = d.dropna(subset=["own_deliv"]); r = o[["prior_deliv", "own_deliv"]].corr().iloc[0, 1]
    ax = axs[0]; style(ax)
    ax.scatter(o.prior_deliv, o.own_deliv, s=9, alpha=0.35, color=C_PIT, lw=0)
    lim = [min(o.prior_deliv.min(), o.own_deliv.min()), max(o.prior_deliv.max(), o.own_deliv.max())]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--"); ax.text(lim[1], lim[1], " same", color=MUTED, fontsize=8, va="center")
    ax.set_xlabel("pitcher's mean delivery on earlier dates (s)"); ax.set_ylabel("this attempt's delivery (s)")
    ax.set_title(f"Is delivery time a stable pitcher trait?\n({name}: r = {r:+.2f}, n = {len(o):,})", loc="left", fontweight="bold", color=INK)
    ax = axs[1]; style(ax); ax.grid(axis="y", visible=False)
    S = R[R["sample"].str.startswith(name) & R.lo.notna()].iloc[::-1].reset_index(drop=True)
    col = {"+ prior delivery (CV)": C_PIT, "+ prior ground allowed (no CV)": C_GAIN, "+ both priors": "#7a4fb5",
           "+ this attempt's delivery (oracle)": C_ORC, "+ this attempt's ground gained": C_ORC}
    for i, s in S.iterrows():
        ax.barh(i, s.gain, color=col[s.model], height=0.6)
        ax.errorbar(s.gain, i, xerr=[[s.gain - s.lo], [s.hi - s.gain]], fmt="none", ecolor=INK, capsize=3)
        ax.text(max(s.hi, 0) + 0.003, i, f"{s.gain:+.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(S)))
    ax.set_yticklabels([f"{s.model.lstrip('+ ')}  [{s['sample'].split('| ')[1]}, n={int(s.n):,}]" for _, s in S.iterrows()], fontsize=7.8)
    ax.axvline(0, color=MUTED, lw=0.8); ax.set_xlabel("AUROC gain over pre-pitch base (95% CI, pitcher-clustered)")
    ax.set_title("Does his earlier delivery predict the next steal?\n(grey = known only after the pitch)", loc="left", fontweight="bold", color=INK)
    f.savefig(fig_path("Fig_add_prior_delivery.png"), facecolor=SURF); print(f"wrote {fig_path('Fig_add_prior_delivery.png')}")


