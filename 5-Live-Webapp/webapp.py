#!/usr/bin/env python3
"""
webapp.py — builds the live site (docs/index.html, served by GitHub Pages from docs/): the season metrics (Steal+,
Burst), the leaderboard and the steal-odds calculator, written into the page's embedded payload.

  python3 5-Live-Webapp/webapp.py        refit and rebuild the site payload (the same as python3 2-Data-Analysis/stealiq.py v15)

Outputs: 5-Live-Webapp/results/ and figures/. How to update and deploy: 5-Live-Webapp/README.md.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "2-Data-Analysis"))
from core import *  # noqa: E402,F401,F403  (paths, loaders, shared models; see 2-Data-Analysis/core.py)


# ════ 1. Season metrics (Steal+, Burst) and the shipped v15 calculator ════════════════════════════════════════
# The season metric suite and the calculator that ship on the site (docs/index.html).
#
# Written in two stages a coder can test on a tiny sample before trusting the full pool:
#
#   load_era()            -> the 2023-26 runner-season pool, with measured ground covered
#                            folded in from the per-attempt data.
#   fit_league(era)       -> league CONSTANTS (the ground~speed regression, the league SB
#                            rate, and the Steal+/Burst reference distributions used for
#                            percentiles). Depends on the whole population — fit ONCE.
#   score(rows, fit)      -> every v11 metric for ANY set of runner-seasons. PURE given
#                            `fit`: scoring 1, 5, or 408 rows gives byte-identical per-row
#                            values. That is what lets us smoke-test the pipeline on n=1,
#                            then n=5, then run the full pool and get the SAME numbers for
#                            the test rows (asserted in the notebook).
#
# Two independent metrics (2023-26 lead-tracking era, clear units) — kept SEPARATE, not
# averaged, because they answer different questions:
#   Steal+   HEADLINE. Net bases (SB - CS) above what an average runner of his sprint speed
#            would produce over the same attempts. VOLUME-AWARE and speed-normalized: 30 bags
#            from a 24 ft/s body counts. Best single answer to "who steals well" — same-season
#            r≈0.88 with success rate.       — "the bases your skill adds above your wheels"
#   Burst    A DIFFERENT lens (near-zero corr with Steal+): feet of ground the runner gains off
#            the base from the pitcher's first move until the pitcher's RELEASE (his secondary
#            lead) ABOVE what his speed predicts. The endpoint is release, not the ball reaching
#            the catcher — the feed has no ball-arrival timestamp, so that is not observable. The coachable process / upside;
#            replaces v10's Statcast 'SB Run Value'.  — "the coachable jump/lead, before the throw"
#            'Ground' is a WEIGHTED blend of the calculator's own two lead features — lead at first
#            move and gain to release — weighted by the calculator's fitted coefficients (see
#            ground_weights()), so Burst is built from the same two quantities the v14 calculator
#            runs on, weighted the way the calculator itself weighs them.
#   (A v10 'Steal Grade' averaged the two percentiles; dropped in v11 — validation showed it
#    predicts net steals / success no better than Steal+ alone and mis-ranks pure producers.)
#
# Decomposition (exact):  Net Bases = NetSpeed + Steal+   (Steal+ IS the surplus term)
#   p_speed  = a0 + a1*sprint_speed                  league success rate expected from SPEED ONLY
#   NetSpeed = sb_attempts*(2*p_speed - 1)           net bases your raw speed alone buys
#   Steal+   = Net Bases - NetSpeed                  net bases your SKILL adds above your speed
#
# Every number in the report/web app is validated empirically by validate() below:
#   - speed-independence (corr with sprint speed ~ 0 for the v11 metrics);
#   - the honest year T -> T+1 tests (Net Bases forecasts next-year VOLUME best; Steal+
#     forecasts next-year SKILL best; Burst is the most repeatable TECHNIQUE).
#
# Also fits the PER-ATTEMPT SB-success model (run_perattempt) on the ~11k individual
# attempts — the grain that actually decides a steal — as the quantitative proof that the
# per-pitch lead distances (which drive Burst) carry the signal (5-fold OOF AUC ~0.74).
#
# Reads the meta tables (attempts) and raw/Raw_Season.csv; writes 5-Live-Webapp/results/ (leaderboard, validation, reliability,
# DF_success_model.csv, DF_perattempt_*.csv, v15_players.json) and 5-Live-Webapp/figures/, and syncs the site payload.
# (the per-attempt XGBoost stage is skipped with a note if xgboost is absent)

ERA_MIN    = 2023           # lead-tracking era start
TEST_IDS   = [647304, 665742]   # Naylor, Soto — the two pipeline-test subjects


# ── small helpers ───────────────────────────────────────────────────────────
def pearson_r(x, y) -> float:
    """Pearson correlation coefficient between two equal-length series, ignoring rows
    where either is NaN. Returns NaN if fewer than 6 usable pairs remain."""
    paired = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    return float(np.corrcoef(paired.x, paired.y)[0, 1]) if len(paired) > 5 else float("nan")

def percentile_rank(values, reference) -> np.ndarray:
    """For each value, its 0-100 percentile against a FIXED reference distribution.
    Reference-based (not rank-within-the-batch), so a runner's percentile does not
    change with how many other runners happen to be scored alongside him."""
    ref = np.sort(np.asarray(reference, float)); n = len(ref)
    return np.array([np.nan if pd.isna(v) else np.searchsorted(ref, v, "right") / n * 100
                     for v in values])

def year_over_year_corr(df, metric_col, next_year_col, min_att=8):
    """Correlate a metric in season T with an outcome in season T+1 for the SAME runner
    (does metric(T) predict next_year_col(T+1)?). Pairs each qualified runner-season to
    that runner's following season. Returns (correlation, number of paired seasons)."""
    cols = ["runner_id", "season"] + list(dict.fromkeys([metric_col, next_year_col]))
    qualified = df[df.sb_attempts >= min_att][cols].dropna()
    year_t  = qualified[["runner_id", "season", metric_col]].rename(columns={metric_col: "x"}).copy()
    year_t["next_season"] = year_t.season + 1                       # line season T up with T+1
    year_t1 = qualified[["runner_id", "season", next_year_col]].rename(
        columns={next_year_col: "y", "season": "next_season"})
    paired = year_t.merge(year_t1, on=["runner_id", "next_season"])
    return pearson_r(paired["x"].values, paired["y"].values), len(paired)


def catcher_faced(attempts: pd.DataFrame) -> pd.DataFrame:
    """Average catcher POP TIME and ARM STRENGTH faced, per runner-season — the opponent-
    difficulty context behind a steal. Pop time is the catch-to-second-base transfer+throw; arm
    is max-effort velocity. Both come from Savant's poptime leaderboard (2015-2026) joined on the
    catcher actually behind the plate for each tracked attempt."""
    pop_path = RAW / "poptime.csv"
    if not pop_path.exists():
        return pd.DataFrame(columns=["runner_id", "season", "pop_faced", "arm_faced"])
    pop = pd.read_csv(pop_path)[["catcher_id", "season", "pop_2b_sba", "maxeff_arm_2b_3b_sba"]]
    a = attempts.merge(pop, on=["catcher_id", "season"], how="left")
    return (a.groupby(["runner_id", "season"])
             .agg(pop_faced=("pop_2b_sba", "mean"),
                  arm_faced=("maxeff_arm_2b_3b_sba", "mean")).reset_index())


def sync_site_payload(payload: dict, marker: str) -> None:
    """Write a payload into docs/index.html in place of the JSON literal after `marker`.
    Keeps the published site in lockstep with the model instead of relying on a manual swap
    (the main payload silently went stale once when catcher pop/arm were added)."""
    site = SITE
    if not site.exists():
        return
    h = site.read_text(encoding="utf-8")
    if marker not in h:
        return
    i = h.index(marker) + len(marker)
    if h[i] != "{":
        return
    depth, j, instr, esc = 0, i, False, False
    while j < len(h):
        c = h[j]
        if instr:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': instr = False
        else:
            if c == '"': instr = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    j += 1; break
        j += 1
    site.write_text(h[:i] + json.dumps(payload, separators=(",", ":")) + h[j:], encoding="utf-8")
    print(f"[sync] docs/index.html <- {marker.strip()}")

# ── ground gained: a blend of the calculator's own two lead features ────────
def ground_weights(coef_path: Path) -> tuple[float, float]:
    """(w_lead, w_gain), summing to 1, so 'ground' stays in feet — a weighted AVERAGE of
    lead_at_firstmove_ft and gain_to_release_ft rather than an unweighted mean of the two.

    v14's calculator fits both as separate per-attempt inputs and learns that gain_to_release
    matters roughly 4x more per foot than lead_at_firstmove (their coefficients, in the currently
    committed fit: 0.303 vs 0.069). An unweighted 50/50 average was tested against the current
    gain-only definition and made the season metric WORSE, not better: year-over-year self-
    stability fell from 0.526 to 0.380, because lead_at_firstmove barely varies between runners
    (its variance is dominated by within-runner noise — see run_success_model's primary_lead
    stat) and diluting it in unweighted made Burst noisier for no offsetting gain. Weighting by
    the calculator's OWN fitted coefficients recovers almost all of that cost (YoY 0.489, Burst
    YoY 0.370 vs the gain-only 0.402) while making the season metric a direct reflection of what
    the per-pitch model actually weighs — which is the point of this change.

    Reads the coefficients the calculator last fit (DF_success_model.csv for the modern era, the
    classic payload's success_model.coef for the classic one) rather than re-fitting here, so this
    has no import-time dependency on sklearn and no ordering dependency on run_success_model()
    having already run in THIS invocation — the coefficients are deterministic given fixed data
    and a fixed seed, so reading yesterday's fit and today's are the same to the reported
    precision. Falls back to (0, 1) — pure gain_to_release, the pre-v15 definition — if no fit
    has been written yet (a fresh clone before the first full run)."""
    if not coef_path.exists():
        return 0.0, 1.0
    if coef_path.suffix == ".json":
        coef = json.loads(coef_path.read_text(encoding="utf-8"))["success_model"]["coef"]
    else:
        sm = pd.read_csv(coef_path)
        coef = {r.term: r.coefficient for r in sm.itertuples() if isinstance(r.term, str)}
    b_lead, b_gain = coef.get("lead_at_firstmove_ft"), coef.get("gain_to_release_ft")
    if not b_lead or not b_gain or b_lead <= 0 or b_gain <= 0:
        return 0.0, 1.0
    return b_lead / (b_lead + b_gain), b_gain / (b_lead + b_gain)


# ── raw era pool (per-attempt ground folded onto each runner-season) ────────
def load_era() -> pd.DataFrame:
    S = pd.read_csv(RAW / "Raw_Season.csv")
    A = load_attempts()
    assert (S["sb_attempts"] == S["SB"] + S["CS"]).all(), "sb_attempts must equal SB+CS"
    era = S[S["season"] >= ERA_MIN].copy()
    era["raw_succ"] = era["SB"] / era["sb_attempts"].clip(lower=1)
    era["net_sb"]   = (era["SB"] - era["CS"]).astype(int)
    av = A[A["result"].isin(["SB", "CS"])].copy()
    w_lead, w_gain = ground_weights(res_path("DF_success_model.csv"))
    av["_ground"] = w_lead * pd.to_numeric(av["lead_at_firstmove_ft"], errors="coerce") \
                  + w_gain * pd.to_numeric(av["gain_to_release_ft"], errors="coerce")
    g = (av.groupby(["runner_id", "season"])
           .agg(ground=("_ground", "mean"),
                lead_rel=("lead_at_release_ft", "mean"),
                tracked=("gain_to_release_ft", "count")).reset_index())
    era = era.merge(g, on=["runner_id", "season"], how="left")
    return era.merge(catcher_faced(av), on=["runner_id", "season"], how="left")


# ── stage 1: fit league constants on the whole population (once) ────────────
def fit_league(era: pd.DataFrame) -> dict:
    f = era.dropna(subset=["ground", "sprint_speed"])
    b1, b0 = np.polyfit(f["sprint_speed"], f["ground"], 1)        # ground_hat = b0 + b1*speed
    # success rate expected from SPEED ONLY (linear; slope ~0.01/ft/s, nearly flat)
    s = era.dropna(subset=["sprint_speed", "raw_succ"])
    a1, a0 = np.polyfit(s["sprint_speed"], s["raw_succ"], 1)      # p_speed = a0 + a1*speed
    league = era["SB"].sum() / era["sb_attempts"].sum()
    p_speed  = a0 + a1 * era["sprint_speed"]
    netspeed = era["sb_attempts"] * (2 * p_speed - 1)
    sp_ref = (era["net_sb"] - netspeed).dropna()                 # volume-aware surplus distribution
    bu_ref = (era["ground"] - (b0 + b1 * era["sprint_speed"])).dropna()
    return dict(b0=float(b0), b1=float(b1), a0=float(a0), a1=float(a1),
                league=float(league), sp_ref=sp_ref.values, bu_ref=bu_ref.values)


# ── stage 2: score ANY set of runner-seasons (pure given `fit`) ─────────────
def score(rows: pd.DataFrame, fit: dict) -> pd.DataFrame:
    d = rows.copy()
    p_speed = fit["a0"] + fit["a1"] * d["sprint_speed"]       # success rate expected from SPEED ONLY
    d["netspeed"]   = d["sb_attempts"] * (2 * p_speed - 1)    # net bases his wheels alone buy
    d["steal_plus"] = d["net_sb"] - d["netspeed"]             # (SB-CS) above the same-speed average
    d["surplus"]    = d["steal_plus"]                         # Steal+ IS the surplus term (kept as alias)
    d["burst_ft"]   = d["ground"] - (fit["b0"] + fit["b1"] * d["sprint_speed"])
    d["steal_plus_pct"] = percentile_rank(d["steal_plus"], fit["sp_ref"])
    d["burst_pct"]      = percentile_rank(d["burst_ft"],   fit["bu_ref"])
    return d

def check_invariants(scored: pd.DataFrame) -> dict:
    """Row-level guarantees that hold for ANY n — the smoke test asserts these."""
    closes = (scored["netspeed"] + scored["surplus"] - scored["net_sb"]).abs().max()
    assert closes < 1e-9, f"decomposition must close exactly (off by {closes})"
    return {"n": len(scored), "decomp_max_resid": float(closes)}


# ── validation: the empirical "why" (kept honest) ───────────────────────────
def validate(era: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(question, metric, value, note=""):
        rows.append({"question": question, "metric": metric, "value": round(value, 3), "note": note})

    # (1) speed-neutrality — a good skill metric should NOT just re-measure raw speed
    for col, lab in [("net_sb", "Net Bases Gained"), ("sb_run_value", "Statcast SB Run Value"),
                     ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("corr with sprint speed (|low| = skill not wheels)", lab, pearson_r(era[col], era["sprint_speed"]))

    # (1b) agreement with Statcast's accepted SB run value (context, not a target)
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("corr with Statcast SB Run Value", lab, pearson_r(era[col], era["sb_run_value"]))

    # (2) how well each metric DESCRIBES the two targets in the SAME season
    for col, lab in [("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("describes net steals SB-CS (same season)", lab, pearson_r(era[col], era["net_sb"]))
    for col, lab in [("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        add("describes success rate (same season)", lab, pearson_r(era[col], era["raw_succ"]))

    # (3) how well each metric PREDICTS the same runner's NEXT season (T -> T+1)
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        r, n = year_over_year_corr(era, col, "net_sb");   add("predicts NEXT-YEAR net steals (SB-CS)", lab, r, f"n={n}")
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        r, n = year_over_year_corr(era, col, "raw_succ"); add("predicts NEXT-YEAR success rate", lab, r, f"n={n}")

    # (4) which metric is the most self-stable year to year (repeatable = more skill, less luck)
    for col, lab in [("net_sb", "Net Bases Gained"), ("steal_plus", "Steal+"), ("burst_ft", "Burst")]:
        r, n = year_over_year_corr(era, col, col); add("year-over-year self-stability", lab, r, f"n={n}")

    # (5) Steal+ and Burst are independent lenses (near-zero corr = they measure different things)
    add("Steal+ vs Burst (independent lenses; ~0 = orthogonal)", "Steal+ x Burst",
        pearson_r(era["steal_plus"], era["burst_ft"]))
    return pd.DataFrame(rows)


# ── reliability: how much of one season's Steal+ is skill, and how much is coin-flips ──
def reliability_audit(full: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """HOW MUCH OF A SINGLE SEASON'S Steal+ IS ACTUALLY SIGNAL?

    Steal+ is a function of roughly 17 binary outcomes at the median volume, so a large part of its
    spread has to be luck. That is measurable rather than a matter of opinion.

    Steal+ = 2(SB - attempts x p_speed), and under the null that a runner is exactly as good as his
    speed predicts, SB ~ Binomial(attempts, p_speed). So the spread Steal+ would show from pure
    chance alone is

        sd_chance = 2 * sqrt(attempts * p * (1 - p))

    Subtracting the mean chance variance from the observed variance leaves the true between-runner
    skill variance, and their ratio is the reliability of a one-season Steal+.

    WHY THIS MATTERS. The reliability that falls out here (~0.20) independently reproduces the
    observed year-over-year self-correlation of Steal+ (0.192, n=224) — two calculations that share
    no code arriving at the same number. That is a strong argument that the low year-to-year figure
    is NOT a defect in Steal+: it is the irreducible consequence of grading a season on ~17 coin
    flips. Burst, which averages hundreds of tracked pitches instead, repeats at 0.479.

    The practical reading: Steal+ is a description of what a runner DID, and is only a projection of
    what he WILL DO once volume is large. sd_chance is the honest error bar to quote beside it."""
    p = fit["a0"] + fit["a1"] * full["sprint_speed"]
    n = full["sb_attempts"]
    sd_chance = 2 * np.sqrt(n * p * (1 - p))
    obs_var = float(full["steal_plus"].var())
    noise_var = float((sd_chance ** 2).mean())
    true_var = max(obs_var - noise_var, 0.0)
    z = full["steal_plus"] / sd_chance
    rows = [
        ("observed SD of Steal+ (bases)", round(float(np.sqrt(obs_var)), 2)),
        ("SD expected from chance alone (bases)", round(float(np.sqrt(noise_var)), 2)),
        ("implied true skill SD (bases)", round(float(np.sqrt(true_var)), 2)),
        ("reliability of a one-season Steal+", round(true_var / obs_var, 3)),
        ("observed year-over-year self-corr (independent check)",
         round(year_over_year_corr(full, "steal_plus", "steal_plus")[0], 3)),
        ("share of runner-seasons beyond 1 chance-SD (%)", round(100 * float((z.abs() > 1).mean()), 1)),
        ("share of runner-seasons beyond 2 chance-SD (%)", round(100 * float((z.abs() > 2).mean()), 1)),
        ("chance SD at the median volume (bases)",
         round(float(2 * np.sqrt(n.median() * 0.79 * 0.21)), 2)),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value"])
    out.to_csv(res_path("DF_v15_reliability.csv"), index=False)

    ranked = full.assign(sd_chance=sd_chance, z=z).nlargest(8, "z")[
        ["player_name", "season", "sb_attempts", "steal_plus", "sd_chance", "z"]]
    ranked.round(2).to_csv(res_path("DF_v15_reliability_top.csv"), index=False)
    fig_reliability(full, sd_chance, z, true_var, noise_var, obs_var)
    return out


def fig_reliability(full, sd_chance, z, true_var, noise_var, obs_var):
    """Two panels answering 'is Steal+ real?' and 'where do net bases come from?'."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=150)

    # A · how much of the spread is luck
    ax[0].bar(["observed\nspread", "expected from\nCHANCE alone", "implied true\nSKILL spread"],
              [np.sqrt(obs_var), np.sqrt(noise_var), np.sqrt(true_var)],
              color=["#0C2340", "#C0392B", "#1A7F47"], width=0.55)
    for i, v in enumerate([np.sqrt(obs_var), np.sqrt(noise_var), np.sqrt(true_var)]):
        ax[0].text(i, v + 0.08, f"{v:.2f}", ha="center", fontweight="bold", fontsize=11)
    ax[0].set_ylabel("standard deviation of Steal+ (bases)")
    ax[0].set_title("A · Most of one season's Steal+ spread is coin-flips\n"
                    f"reliability = {true_var/obs_var:.2f}  —  and Steal+ repeats year to year at 0.19",
                    fontsize=10.5, fontweight="bold", color="#0C2340")
    ax[0].set_ylim(0, np.sqrt(obs_var) * 1.25)

    # B · where net bases come from, by speed quintile
    d = full.dropna(subset=["sprint_speed", "net_sb"]).copy()
    d["q"] = pd.qcut(d["sprint_speed"], 5, labels=["slowest", "slow", "mid", "fast", "fastest"])
    g = d.groupby("q", observed=True).agg(ns=("netspeed", "mean"), sp=("steal_plus", "mean"),
                                          spd=("sprint_speed", "mean"))
    xs = np.arange(len(g))
    ax[1].bar(xs, g.ns, color="#9AA0A6", label="NetSpeed — what the wheels alone buy")
    ax[1].bar(xs, g.sp, bottom=g.ns, color="#2F6FB0", label="Steal+ — what the skill adds")
    ax[1].axhline(0, color="#333", lw=0.8)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{i}\n{s:.1f} ft/s" for i, s in zip(g.index, g.spd)], fontsize=9)
    ax[1].set_ylabel("mean net bases (SB − CS)")
    ax[1].set_title("B · Net Bases = NetSpeed + Steal+, exactly\n"
                    "Speed sets the level; Steal+ averages ~0 in every speed group by construction",
                    fontsize=10.5, fontweight="bold", color="#0C2340")
    ax[1].legend(fontsize=8.5, frameon=False, loc="upper left")

    for a in ax:
        a.grid(alpha=0.13, lw=0.7, axis="y")
        for sp_ in ("top", "right"): a.spines[sp_].set_visible(False)
    fig.text(0.5, -0.04,
             "Left: under the null that a runner is exactly as good as his speed predicts, "
             "SB ~ Binomial(attempts, p), so chance alone produces an SD of 3.73 bases at these "
             "volumes.\nOnly 20% of the observed variance survives as skill — which is why "
             "Steal+ describes a season well (r = 0.64 with net steals) but forecasts the next one "
             "weakly.  Right: the two components sum to net bases with zero residual (< 1e-9).",
             ha="center", fontsize=8.6, color="#444")
    fig.tight_layout(); fig.savefig(fig_path("Fig_v15_Reliability.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── webapp record export ─────────────────────────────────────────────────────
def to_records(full: pd.DataFrame) -> list:
    def num(v, nd=1):
        return None if (v is None or pd.isna(v)) else round(float(v), nd)
    recs = []
    for _, r in full.iterrows():
        recs.append({
            "id": int(r["runner_id"]), "name": r["player_name"],
            "team": (r["team"] if isinstance(r["team"], str) else ""), "season": int(r["season"]),
            "speed": num(r["sprint_speed"], 1), "speed_pct": num(r["speed_pct"], 0),
            "attempts": int(r["sb_attempts"]), "pop": num(r.get("pop_faced"), 2),
            "arm": num(r.get("arm_faced"), 1),
            "jump": num(r["jump_time"], 2), "jump_pct": num(r["jump_pct"], 0),
            "ground": num(r.get("ground"), 1), "ground_pct": num(r.get("ground_pct"), 0),
            "lead_rel": num(r.get("lead_rel"), 1),
            "steal_plus": num(r["steal_plus"], 1), "sp_pct": num(r["steal_plus_pct"], 0),
            "burst": num(r["burst_ft"], 1), "burst_pct": num(r["burst_pct"], 0),
            "sb": int(r["SB"]), "cs": int(r["CS"]), "net": int(r["net_sb"]),
            "netspeed": num(r["netspeed"], 1), "surplus": num(r["surplus"], 1),
            "success": (None if pd.isna(r["raw_succ"]) else int(round(r["raw_succ"] * 100))),
            "srv": num(r.get("sb_run_value"), 1),   # Statcast value — shown for reference only
        })
    return recs


# ── full run: score everyone, validate, write the artifacts ─────────────────
def build():
    era = load_era()
    fit = fit_league(era)
    full = score(era.copy(), fit)
    check_invariants(full)
    full["speed_pct"]  = percentile_rank(full["sprint_speed"], era["sprint_speed"].dropna().values)
    full["jump_pct"]   = 100 - percentile_rank(full["jump_time"], era["jump_time"].dropna().values)
    full["ground_pct"] = percentile_rank(full["ground"], era["ground"].dropna().values)
    return full, fit, validate(full)


PA_FRIENDLY = {"lead_at_firstmove_ft": "Lead at first move (ft)",
               "gain_to_release_ft": "Ground gained to release (ft)",
               "lead_at_release_ft": "Lead at release (ft)", "base_is_3b": "Stealing 3rd",
               "sprint_speed": "Sprint speed", "jump_time": "Jump time", "accel_gap": "Accel gap",
               "primary_lead": "Primary lead (career)", "lead_gain": "Lead gain (career)", "bolts": "Bolts"}


def run_perattempt(seed: int = 42):
    """Fit the per-attempt SB-success model on the tracked attempts (5-fold OOF AUC),
    write the AUC + importance tables and their two figures. Needs xgboost + sklearn;
    returns None (with a note) if they are unavailable rather than failing the run."""
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"per-attempt model skipped (missing {e.name}); season metrics unaffected")
        return None

    df = load_attempts()
    df = df[df["result"].isin(["SB", "CS"])].copy()
    df["y"] = (df["result"] == "SB").astype(int)
    df["base_is_3b"] = (df["base"].astype(str) == "3B").astype(int)

    sssi = pd.read_csv(RAW / "DF_v7_SSSI.csv")
    keep = ["runner_id", "season"] + [c for c in PA_RUNNER_FEATS if c in sssi.columns]
    df = df.merge(sssi[keep].drop_duplicates(["runner_id", "season"]), on=["runner_id", "season"], how="left")
    runner_cols = [c for c in PA_RUNNER_FEATS if c in df.columns]
    feats = PA_LEAD_FEATS + ["base_is_3b"] + runner_cols
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")

    y = df["y"].values
    prior = float(df["y"].mean())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def xgb():
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                             eval_metric="logloss", verbosity=0, random_state=seed, use_label_encoder=False)

    def encode_map(idx, key, smoothing=20.0):
        """Smoothed mean-target encoding learned from the rows in `idx`."""
        stats = df.iloc[idx].groupby(key)["y"].agg(["sum", "count"])
        return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)

    def apply_map(enc, idx, key):
        return df.iloc[idx][key].map(enc).fillna(prior).values

    def cv_auc(battery_keys=()):
        """Pooled out-of-fold AUC. Battery tendency is target-encoded, and the TRAINING rows
        are encoded with an inner K-fold so no row ever sees its own outcome. Encoding the
        train rows from the same rows leaks the label into the feature: the model over-trusts
        it in training, the clean val encoding then behaves differently, and AUC drops. The
        leak is ~1/(n+smoothing) per row, so it is worst for sparsely-seen pitchers."""
        oof = np.zeros(len(df))
        for tr, va in cv.split(df, y):
            Xtr, Xva = df.iloc[tr][feats].copy(), df.iloc[va][feats].copy()
            for key in battery_keys:
                col = key + "_enc"
                inner_enc = np.full(len(tr), prior)
                inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
                for i_tr, i_va in inner.split(df.iloc[tr], y[tr]):
                    inner_enc[i_va] = apply_map(encode_map(tr[i_tr], key), tr[i_va], key)
                Xtr[col] = inner_enc                                  # nested OOF -> no self-leak
                Xva[col] = apply_map(encode_map(tr, key), va, key)    # val: encoded from full train
            oof[va] = xgb().fit(Xtr.values, y[tr]).predict_proba(Xva.values)[:, 1]
        return roc_auc_score(y, oof)

    auc_leads   = cv_auc()                                        # leads + base + runner skill
    auc_catcher = cv_auc(["catcher_id"])                          # + catcher tendency
    auc_pitcher = cv_auc(["pitcher_id"])                          # + pitcher tendency
    auc_full    = cv_auc(["catcher_id", "pitcher_id"])            # + both
    pd.DataFrame([{"model": "per-attempt: leads+base+runner",            "auc": round(auc_leads, 4)},
                  {"model": "per-attempt: + catcher (nested OOF)",       "auc": round(auc_catcher, 4)},
                  {"model": "per-attempt: + pitcher (nested OOF)",       "auc": round(auc_pitcher, 4)},
                  {"model": "per-attempt: + both battery (nested OOF)",  "auc": round(auc_full, 4)}]
                 ).to_csv(res_path("DF_perattempt_AUC.csv"), index=False)

    imp = (pd.DataFrame({"feature": feats, "importance": xgb().fit(df[feats].values, y).feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    imp.to_csv(res_path("DF_perattempt_Importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    labels = ["Leads +\nrunner skill", "+ catcher\ntendency", "+ pitcher\ntendency", "+ both"]
    vals   = [auc_leads, auc_catcher, auc_pitcher, auc_full]
    ax.bar(labels, vals, color=["#10B981", "#2F6FB0", "#9CA3AF", "#1F2D3D"], width=0.55)
    ax.axhline(auc_leads, color="#10B981", lw=1, ls="--", zorder=0)
    ax.set_ylabel("CV AUC (nested out-of-fold)"); ax.set_ylim(0.5, 0.82)
    ax.set_title("Per-Attempt Model — leads carry most of it; the catcher adds the rest", fontsize=11.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=11)
    ax.text(0.5, -0.20, "Catchers are seen ~46 times each, so their tendency is estimable; pitchers a median of 6 times,\n"
            "and the runner's lead already absorbs most of the pitcher's effect.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555")
    fig.tight_layout(); fig.savefig(fig_path("Fig_AUC.png"), dpi=160); plt.close(fig)

    g = imp.iloc[::-1]
    ax = plt.subplots(figsize=(7.6, 4.6))[1]
    ax.barh([PA_FRIENDLY.get(f, f) for f in g["feature"]], g["importance"],
            color=["#0EA5E9" if f in PA_LEAD_FEATS else "#1F2D3D" for f in g["feature"]])
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Per-Attempt Model — what decides a steal (blue = per-pitch lead distances)", fontsize=11)
    plt.tight_layout(); plt.savefig(fig_path("Fig_Importance.png"), dpi=160); plt.close()
    print(f"per-attempt SB-success AUC (nested OOF): {auc_leads:.4f} leads+runner | "
          f"{auc_catcher:.4f} +catcher | {auc_pitcher:.4f} +pitcher | {auc_full:.4f} +both, n={len(df)}")
    return auc_leads, auc_full


def fit_calibrator(oof: np.ndarray, y: np.ndarray) -> dict | None:
    """THE SHIPPED CALIBRATION MAP — isotonic, serialized for the browser.

    The raw logistic is a good RANKER and a poor RATE. Out-of-fold it discriminates at AUROC
    0.756 but its expected calibration error is 0.021, with 5 of 10 deciles more than two
    binomial SE off: it reads ~0.876 where the observed rate is 0.919. The calculator prints that
    number as "chance the attempt is safe", so the error is the user-facing claim, not a footnote.

    WHY ISOTONIC AND NOT PLATT. The miscalibration is a WAVE, not a monotone S — over-confident in
    deciles 2-3, under-confident in 6-8, reversing again at 9 — and the signed gaps sum to
    -0.0000, i.e. the errors cancel and the model is unbiased in aggregate. A global shift or a
    single-slope sigmoid therefore cannot help; measured, Platt makes it WORSE (ECE 0.021 ->
    0.034, 8 bad deciles). Isotonic is free to bend with the wave: ECE 0.021 -> 0.009 with ZERO
    deciles beyond 2 SE, and Brier improves too (0.1359 -> 0.1350). The cost is 0.003 AUROC from
    the step map creating ties, which sits well inside the bootstrap CI (0.7416-0.7703).

    (An earlier note in this project declined to calibrate on the grounds that isotonic "would
    hide" a falling league base rate. That argument was written about v12's FORWARD holdout, where
    the drift is a genuine base-rate shift across seasons. It does not describe this table, which
    is a pooled random split whose signed gaps cancel — a shape problem, not a drift problem.)

    WHICH MAP SHIPS. Fit ONE isotonic on the full-data out-of-fold predictions. Fitting on OOF
    rather than in-sample is what keeps it honest; fitting a single final map on all of them uses
    every row. The performance NUMBERS quoted anywhere else come from a nested run (calibrator fit
    on inner training folds only) — never from this map scored on its own training data, which
    would be optimistic.

    Returned as {"x": [...], "y": [...]}: 68 breakpoints, ~1.3 KB. sklearn's predict() on an
    isotonic fit is exactly linear interpolation over these thresholds with end clipping — checked
    to a max difference of 0.0 across [0,1] — so the browser reproduces it with a plain interp."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return None
    ir = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    return {"x": [round(float(v), 6) for v in ir.X_thresholds_],
            "y": [round(float(v), 6) for v in ir.y_thresholds_]}


def nested_calibrated_oof(X, y, seed: int = 42) -> np.ndarray | None:
    """Out-of-fold predictions of the CALIBRATED model, with the calibrator fit only on inner
    training folds. This is the only honest way to score a calibrated model: fitting the map on
    the same rows you then score inflates the result, because isotonic can memorise them.

    Outer 5-fold -> within each training side, an inner 5-fold produces clean OOF predictions ->
    the isotonic map is fit on those -> it is applied to the held-out outer fold, which no part of
    the calibrator has seen. Returns predictions aligned to the input rows."""
    try:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        return None
    out = np.zeros(len(y))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        lr = LogisticRegression(max_iter=5000).fit(X[tr], y[tr])
        inner_oof = np.zeros(len(tr))
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1)
        for itr, ite in inner.split(X[tr], y[tr]):
            li = LogisticRegression(max_iter=5000).fit(X[tr][itr], y[tr][itr])
            inner_oof[ite] = li.predict_proba(X[tr][ite])[:, 1]
        ir = IsotonicRegression(out_of_bounds="clip").fit(inner_oof, y[tr])
        out[te] = ir.predict(lr.predict_proba(X[te])[:, 1])
    return out


def run_success_model(seed: int = 42):
    """P(safe) for ONE attempt from four numbers a coach already has: sprint speed, the lead he
    had when the pitcher committed, the ground he gained from there, and the pop time of the
    catcher he is running on. Deliberately a plain logistic regression on RAW units — it also
    BEAT XGBoost on the same features (0.726 vs 0.722), so nothing is sacrificed for the
    simplicity — and the coefficients read directly as 'per ft/s', 'per foot' and 'per second'.
    Writes 5-Live-Webapp/results/DF_success_model.csv and syncs the fit into docs/index.html."""
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print(f"success model skipped (missing {e.name})")
        return None

    at = load_attempts()
    at = at[at["result"].isin(["SB", "CS"])].copy()
    at["y"] = (at["result"] == "SB").astype(int)
    lb = pd.read_csv(res_path("DF_v15_leaderboard.csv"))[["runner_id", "season", "sprint_speed", "burst_ft"]]
    at = at.merge(lb, on=["runner_id", "season"], how="left")

    # The leaderboard only holds the 408 QUALIFIED runner-seasons (>=10 attempts, gated upstream
    # by the committed DF_v7_SSSI.csv), while the attempts table spans 1,079 — which used to drop
    # 3,619 attempts (35%) out of the calculator. Fill the rest in: sprint speed from the full
    # leaderboard, Burst recomputed from the leads already on disk against the same league line
    # fit_league() uses, so the definition is identical for everyone.
    sp_path = RAW / "sprint_speed.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        at = at.merge(sp, on=["runner_id", "season"], how="left")
        at["sprint_speed"] = at["sprint_speed"].fillna(at["sprint_speed_all"])

    meta = json.loads(res_path("v15_players.json").read_text(encoding="utf-8"))["meta"]
    b0, b1 = meta["ground_fit"]["b0"], meta["ground_fit"]["b1"]
    w_lead, w_gain = ground_weights(res_path("DF_success_model.csv"))   # same blend load_era() used
    gain = (w_lead * pd.to_numeric(at["lead_at_firstmove_ft"], errors="coerce")
            + w_gain * pd.to_numeric(at["gain_to_release_ft"], errors="coerce"))
    grp = at.assign(_g=gain).groupby(["runner_id", "season"])["_g"]
    ground, n_tracked = grp.transform("mean"), grp.transform("count")
    league_ground = float(gain.mean())
    # shrink thin samples toward the league line; below 3 tracked attempts, don't guess at all
    w = (n_tracked / (n_tracked + 10)).clip(upper=1.0)
    ground_shrunk = w * ground + (1 - w) * league_ground
    burst_all = (ground_shrunk - (b0 + b1 * at["sprint_speed"])).where(n_tracked >= 3)
    at["burst_ft"] = at["burst_ft"].fillna(burst_all)

    pop_path = RAW / "poptime.csv"
    if pop_path.exists():
        pop = pd.read_csv(pop_path)[["catcher_id", "season", "pop_2b_sba"]] \
                .rename(columns={"pop_2b_sba": "pop_faced"})
        at = at.merge(pop, on=["catcher_id", "season"], how="left")
    else:
        at["pop_faced"] = np.nan
    at[SIMPLE_FEATS] = at[SIMPLE_FEATS].apply(pd.to_numeric, errors="coerce")

    # Primary lead (the ground he has BEFORE the pitcher commits) is context, not an input:
    # runners converge on it, so it barely separates anyone. Quantify that for the page.
    pl = pd.to_numeric(at["lead_at_firstmove_ft"], errors="coerce")
    seas = at.assign(_pl=pl).dropna(subset=["_pl"]).groupby(["runner_id", "season"])["_pl"]
    means, counts = seas.mean(), seas.count()
    qual = means[counts >= 15]
    grand = float(pl.mean())
    ss_between = float((counts[counts >= 15] * (qual - grand) ** 2).sum())
    dq = at.assign(_pl=pl).dropna(subset=["_pl"])
    dq = dq[dq.set_index(["runner_id", "season"]).index.isin(qual.index)]
    ss_total = float(((dq["_pl"] - grand) ** 2).sum())
    primary_lead = {"mean": round(grand, 1),
                    "runner_min": round(float(qual.min()), 1),
                    "runner_max": round(float(qual.max()), 1),
                    "within_pct": round(100 * (1 - ss_between / ss_total), 1),
                    "n_runner_seasons": int(len(qual))}

    at = at.dropna(subset=SIMPLE_FEATS).reset_index(drop=True)
    X, y = at[SIMPLE_FEATS].values, at["y"].values

    pipe = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)

    # ── calibration ────────────────────────────────────────────────────────
    # Two fits, two purposes — see fit_calibrator(). `cal_oof` is NESTED (the calibrator never
    # sees the fold it scores) and is what the reported calibrated metrics come from; `calib` is
    # the single map that ships to the browser.
    cal_oof = nested_calibrated_oof(X, y, seed=seed)
    calib = fit_calibrator(oof, y)
    # AUPRC is meaningless without its no-skill floor (= the base rate), and F1 at a fixed 0.5
    # is vacuous on a skewed target — report the floor and the tuned threshold explicitly.
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
    auprc = float(average_precision_score(y, oof))
    brier = float(brier_score_loss(y, oof))
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in grid]
    best = int(np.argmax(f1s))
    pipe.fit(X, y)
    lr = pipe.named_steps["logisticregression"]
    coefs = dict(zip(SIMPLE_FEATS, lr.coef_[0]))
    ece_raw, bad_raw = expected_calibration_error(y, oof)
    payload = {"intercept": float(lr.intercept_[0]),
               "coef": {k: float(v) for k, v in coefs.items()},
               "auc": round(float(auc), 4), "n": int(len(at)),
               "auprc": round(auprc, 4), "auprc_floor": round(float(y.mean()), 4),
               "brier": round(brier, 4),
               "f1_best": round(float(f1s[best]), 4), "f1_threshold": round(float(grid[best]), 2),
               "base_rate": round(float(y.mean()), 4),
               "ece_raw": round(ece_raw, 4), "bad_deciles_raw": bad_raw,
               # 95% CI from the retired decompose.py's 1000-resample bootstrap (archived), and the label-permutation
               # null it is measured against — both quoted on the model card so the page states
               # its own uncertainty instead of a bare point estimate.
               "auc_ci": [0.7416, 0.7703], "perm_null": 0.4985,
               "primary_lead": primary_lead,
               # full observed span so the sliders cover everyone (incl. Naylor at 24.4 ft/s)
               "range": {f: [round(float(at[f].min()), 1),
                             round(float(at[f].max()), 1),
                             round(float(at[f].median()), 1)] for f in SIMPLE_FEATS}}
    if calib is not None:
        payload["calibration"] = calib
    if cal_oof is not None:
        ece_cal, bad_cal = expected_calibration_error(y, cal_oof)
        payload.update({"auc_cal": round(float(roc_auc_score(y, cal_oof)), 4),
                        "brier_cal": round(float(brier_score_loss(y, cal_oof)), 4),
                        "ece_cal": round(ece_cal, 4), "bad_deciles_cal": bad_cal})
    rows = [{"term": "intercept", "coefficient": round(payload["intercept"], 5), "odds_multiplier": ""}]
    rows += [{"term": f, "coefficient": round(v, 5), "odds_multiplier": round(float(np.exp(v)), 4)}
             for f, v in coefs.items()]
    rows.append({"term": "CV AUC (5-fold)", "coefficient": payload["auc"], "odds_multiplier": ""})
    pd.DataFrame(rows).to_csv(res_path("DF_success_model.csv"), index=False)

    # keep the browser calculator locked to this fit — never let the two drift apart
    site = SITE
    if site.exists():
        html = site.read_text(encoding="utf-8")
        a, b = "/*__SUCCESS_MODEL__*/", "/*__END_SUCCESS_MODEL__*/"
        if a in html and b in html:
            head, rest = html.split(a, 1)
            _, tail = rest.split(b, 1)
            site.write_text(head + a + json.dumps(payload, separators=(",", ":")) + b + tail,
                            encoding="utf-8")

    per_ft = np.exp(coefs["gain_to_release_ft"])
    print(f"4-input success model (logistic): AUROC {auc:.4f} | AUPRC {auprc:.4f} "
          f"(floor {y.mean():.4f}) | Brier {brier:.4f} | F1 {f1s[best]:.4f} @thr "
          f"{grid[best]:.2f} | n={len(at)}")
    if cal_oof is not None:
        print(f"  + isotonic (SHIPPED, nested OOF): AUROC {payload['auc_cal']:.4f} | "
              f"Brier {payload['brier_cal']:.4f} | ECE {ece_raw:.4f} -> "
              f"{payload['ece_cal']:.4f} | deciles beyond 2 SE {bad_raw} -> "
              f"{payload['bad_deciles_cal']}  [{len(calib['x'])}-point map -> browser]")
    print(f"  each +1 ft of ground gained multiplies the odds of being safe by {per_ft:.2f}x")
    fig_roc_calculator(y, oof, auc)
    return payload


def fig_roc_calculator(y, oof, auc):
    """The actual ROC curve behind the v14 calculator's AUROC, not just the number.

    An AUROC on its own doesn't show WHERE the model's discrimination comes from or what it costs
    to raise the true-positive rate. The curve does: at this base rate (~81% safe), a caught-
    stealing-averse threshold near the top-left knee trades roughly 1 point of true-positive rate
    for every ~2 points of false-positive rate it gives up, which is the shape a coach is actually
    choosing between when picking a threshold."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    fpr, tpr, _ = roc_curve(y, oof)
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=150)
    ax.plot(fpr, tpr, lw=2.6, color="#2F6FB0", label=f"v14 calculator (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1.2, ls="--", color="#9AA0A6", label="no-skill (AUROC 0.500)")
    ax.fill_between(fpr, tpr, fpr, alpha=0.08, color="#2F6FB0")
    ax.set_xlabel("false-positive rate (called safe, actually caught)")
    ax.set_ylabel("true-positive rate (called safe, actually safe)")
    ax.set_title(f"ROC — the calculator vs a coin flip\n5-fold out-of-fold, n={len(y):,} attempts",
                fontsize=11, fontweight="bold", color="#0C2340")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    fig.tight_layout(); fig.savefig(fig_path("Fig_ROC_Calculator.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── the odds calculator: the paper's success model ──────────────────────────────────────────────────────────────
# The calculator on the site runs the best predictor in the project: the paper's per-base success GLMM (paper.py,
# DF_paper_forward_all.csv), refit here on every 2023-26 attempt. A browser cannot know a player's own effect, so it
# runs the fixed part, every runner, pitcher and catcher at average. The specification is whichever won the paper's
# cross-validation (DF_paper_success_cv.csv); a new term needs one line in CALC_TERMS and no change to the page.
# (run_success_model above still fits the v15 four-input model on 2023-26: its two lead weights define the leaderboard's
# Ground and Burst. The 2015-22 era's calculator runs the same four inputs, fit on that era.)
CALC_TERMS = {   # model term -> how the page computes it from what the reader sets:
                 # (scale * x - centre) ** power / div, or 1 when the input equals `eq`
    "speed": dict(input="speed"), "lead": dict(input="lead"), "gain": dict(input="gain"),
    "pop": dict(input="pop", scale=10),                       # the model reads pop time in tenths of a second
    "gain2": dict(input="gain", power=2, div=10),
    "balls": dict(input="balls"), "strikes": dict(input="strikes"),
    "out1": dict(input="outs", eq=1), "out2": dict(input="outs", eq=2),
    "lhp": dict(input="p_throws", eq="L"), "rhb": dict(input="stand", eq="R"),
    "brk": dict(input="pitch", eq="breaking"), "off": dict(input="pitch", eq="offspeed")}


def calc_eta(model: dict, inputs: pd.DataFrame) -> np.ndarray:
    """The page's formula (calcEta in docs/index.html) in Python: log-odds of a safe steal from the reader's inputs."""
    eta = np.full(len(inputs), model["intercept"])
    for t in model["terms"]:
        x = inputs[t["input"]]
        v = (x == t["eq"]).astype(float) if "eq" in t else (t.get("scale", 1) * x - t["centre"]) ** t.get("power", 1) / t.get("div", 1)
        eta += t["beta"] * v.to_numpy(float)
    return eta


def check_page_formula(model: dict, inputs: pd.DataFrame, eta: np.ndarray) -> None:
    """Run the page's own calcEta, cut from docs/index.html, in node on these rows and compare with Python."""
    import re, shutil, subprocess
    node, m = shutil.which("node"), re.search(r"function calcEta\(.*?\n}\n", SITE.read_text(encoding="utf-8"), re.S)
    if not (node and m):
        print("  (node or calcEta not found: the page's formula was not checked)")
        return
    js = m.group(0) + ("let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{const {M,R}=JSON.parse(s);"
                       "process.stdout.write(JSON.stringify(R.map(r=>calcEta(M,r))));});")
    out = subprocess.run([node, "-e", js], input=json.dumps({"M": model, "R": inputs.to_dict(orient="records")}),
                         capture_output=True, text=True, check=True).stdout
    gap = float(np.abs(np.array(json.loads(out)) - eta).max())
    assert gap < 1e-9, f"the page's calcEta is off the model by {gap}"
    print(f"  page formula (node) matches the model on {len(inputs):,} attempts: max gap {gap:.1e}")


def run_calculator(n_boot: int = 500, seed: int = 0):
    """Ship the success GLMM's fixed part to the page: refit it per base on 2023-26, check that the page's formula
    reproduces the model on every attempt (in Python, then the page's own JavaScript in node), score the same model fit
    on 2023-25 on the paper's 2026 forward-test attempts, and write it into docs/index.html. Writes
    5-Live-Webapp/results/DF_calculator.csv."""
    from scipy.special import expit
    from paper import GLMM, SUCCESS_SPECS, design, success_rows
    need = [n for n in ("DF_paper_forward_all.csv", "DF_paper_success_cv.csv", "DF_paper_breakeven.csv") if not res_path(n).exists()]
    if need:
        print(f"calculator skipped: {need} missing; run `python3 2-Data-Analysis/stealiq.py decide success` first")
        return None
    F = pd.read_csv(res_path("DF_paper_forward_all.csv")).set_index("model")
    S = pd.read_csv(res_path("DF_paper_success_cv.csv"))
    spec = S[S.chosen].spec.iloc[0]; terms, name = SUCCESS_SPECS[spec], f"Success GLMM: {spec}"
    if F.auroc.idxmax() != name:
        print(f"NOTE: the best model in DF_paper_forward_all.csv is {F.auroc.idxmax()!r}; the calculator runs {name!r}")
    missing = [t for t in terms if t not in CALC_TERMS]
    if missing:
        raise KeyError(f"add {missing} to CALC_TERMS so the page can compute them")
    BE = pd.read_csv(res_path("DF_paper_breakeven.csv"))
    bases, scored, rows = {}, [], []
    for base in ["2B", "3B"]:
        a = success_rows(base)
        inputs = pd.DataFrame({"speed": a.speed_raw, "lead": a.lead_raw, "gain": a.gain_raw, "pop": a.pop_raw / 10,
                               "balls": a.balls_raw, "strikes": a.strikes_raw, "outs": a.outs_when_up,
                               "p_throws": a.p_throws, "stand": a.stand, "pitch": a.pitch_class})
        centre = {c: float((a[c + "_raw"] - a[c]).iloc[0]) for c in ("speed", "lead", "pop", "balls", "strikes", "gain")}
        X, codes, _ = design(a, terms); m = GLMM(X, codes, a.y.values).fit()
        model = {"intercept": float(m.beta[0]), "terms": [
            {"term": t, "beta": float(b), **CALC_TERMS[t], **({} if "eq" in CALC_TERMS[t] else {"centre": centre[CALC_TERMS[t]["input"]]})}
            for t, b in zip(terms, m.beta[1:])]}
        eta = calc_eta(model, inputs)
        gap = float(np.abs(eta - X @ m.beta).max())
        assert gap < 1e-9, f"{base}: the exported formula is off the model by {gap}"
        check_page_formula(model, inputs, eta)
        # sliders span every attempt; a squared term turns back up below its vertex, where few attempts are, so that
        # slider starts at the larger of the vertex and the 1st percentile and stops at the 99th
        rng = {k: [float(inputs[k].min()), float(inputs[k].max()), float(inputs[k].median())] for k in ("speed", "lead", "gain", "pop")}
        for t in model["terms"]:
            if t.get("power") == 2 and t["beta"] > 0:
                lin = sum(u["beta"] for u in model["terms"] if u["input"] == t["input"] and "eq" not in u and u.get("power", 1) == 1)
                vertex = (t["centre"] - lin * t.get("div", 1) / (2 * t["beta"])) / t.get("scale", 1)
                rng[t["input"]][:2] = [max(float(inputs[t["input"]].quantile(0.01)), vertex), float(inputs[t["input"]].quantile(0.99))]
        d = {k: 2 if k == "pop" else 1 for k in rng}                      # slider steps: 0.01 s, 0.1 ft and ft/s
        for k, (lo, hi, mid) in rng.items():                              # round inward, so every step is in the data
            lo, hi = np.ceil(lo * 10 ** d[k] - 1e-9) / 10 ** d[k], np.floor(hi * 10 ** d[k] + 1e-9) / 10 ** d[k]
            rng[k] = [float(lo), float(hi), float(min(max(round(mid, d[k]), lo), hi))]
        default = {k: v[2] for k, v in rng.items()} | {c: inputs[c].mode()[0] for c in ("balls", "strikes", "outs", "p_throws", "stand", "pitch")}
        be = BE[BE.base == base].set_index("outs").breakeven
        bases[base] = {"n": int(len(a)), **model, "range": rng, "default": default, "breakeven": [round(float(be[o]), 4) for o in range(3)]}
        rows += [dict(base=base, term="intercept", beta=model["intercept"])] + [dict(base=base, **t) for t in model["terms"]]
        # the paper's forward test, fixed part only: fit on 2023-25, score the 2026 attempts
        tr, te = a[a.season <= 2025], a[a.season == 2026]
        Xtr, ctr, lev = design(tr, terms); Xte, _, _ = design(te, terms, lev)
        scored.append(te.assign(p=expit(Xte @ GLMM(Xtr, ctr, tr.y.values).fit().beta)))
    T = pd.concat(scored)
    C = T[T.arm.notna() & T.catcher_pop_2b.notna()].reset_index(drop=True)      # the paper's common rows
    y, p = C.y.values, C.p.values
    grp = C.groupby("runner_id").indices; keys = np.array(list(grp)); r = np.random.default_rng(seed)
    boots = [np.concatenate([grp[k] for k in r.choice(keys, len(keys))]) for _ in range(n_boot)]
    b = [roc_auc_score(y[i], p[i]) for i in boots]
    test = {"auroc": round(float(roc_auc_score(y, p)), 4), "auroc_ci": [round(float(np.percentile(b, q)), 4) for q in (2.5, 97.5)],
            "n": int(len(y)), "mean_pred": round(float(p.mean()), 4), "observed": round(float(y.mean()), 4),
            "full_auroc": round(float(F.loc[name, "auroc"]), 4)}
    payload = {"kind": "glmm", "model": name, "fit": "2023-26", "test": test, "bases": bases}
    pd.DataFrame(rows + [dict(base="2026 forward test", term="AUROC (fixed part)", beta=test["auroc"])]).to_csv(
        res_path("DF_calculator.csv"), index=False)
    sync_site_payload(payload, "const CALC_MODEL = ")
    print(f"calculator: {name}, fixed part | 2026 forward test AUROC {test['auroc']:.4f} "
          f"(95% CI {test['auroc_ci'][0]:.4f}-{test['auroc_ci'][1]:.4f}) on {test['n']:,} attempts; with player effects "
          f"{test['full_auroc']:.4f}; mean prediction {test['mean_pred']:.3f} vs observed {test['observed']:.3f}")
    return payload


def run_v15():
    full, fit, val = build()
    lb_cols = ["runner_id", "season", "player_name", "team", "sprint_speed", "jump_time",
               "ground", "burst_ft", "SB", "CS", "net_sb", "raw_succ", "sb_attempts",
               "steal_plus", "steal_plus_pct", "burst_pct",
               "netspeed", "surplus", "sb_run_value"]
    full[lb_cols].sort_values("steal_plus", ascending=False).to_csv(
        res_path("DF_v15_leaderboard.csv"), index=False)
    val.to_csv(res_path("DF_v15_validation.csv"), index=False)

    league = fit["league"]
    payload = {
        # the site badge reads this — it is the MODEL version, not the era (the era dropdown
        # carries that separately), so both payloads report the same version
        "meta": {"version": "v15", "era": f"{ERA_MIN}-2026",
                 "n_player_seasons": len(full), "league_success_pct": round(league * 100, 1),
                 # league fits, so the report/site can draw the speed→expectation curves + Steal+ coefficients
                 "p_speed_fit": {"a0": fit["a0"], "a1": fit["a1"]},   # expected success = a0 + a1*speed
                 "ground_fit":  {"b0": fit["b0"], "b1": fit["b1"]}},  # expected ground = b0 + b1*speed
        "validation": val.to_dict(orient="records"),
        "players": to_records(full),
    }
    res_path("v15_players.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    sync_site_payload(payload, "const PAYLOAD = ")

    show = ["player_name", "season", "sprint_speed", "SB", "CS", "net_sb", "steal_plus", "burst_ft"]
    top12 = (full.dropna(subset=["steal_plus"])
             .sort_values("steal_plus", ascending=False).head(12)[show].round(2))
    print(f"{len(full)} runner-seasons ({ERA_MIN}-2026) | league SB% {league*100:.1f} | "
          f"p_speed = {fit['a0']:.3f} + {fit['a1']:.4f}*speed | ground = {fit['b0']:.2f} + {fit['b1']:.3f}*speed")
    print(top12.to_string(index=False))
    print(val.to_string(index=False))
    rel = reliability_audit(full, fit)
    print("\n=== HOW MUCH OF ONE SEASON'S Steal+ IS SIGNAL? ===")
    print(rel.to_string(index=False))
    run_perattempt()
    run_success_model()
    run_calculator()
    return full, val


if __name__ == "__main__":
    run_v15()
