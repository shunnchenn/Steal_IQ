#!/usr/bin/env python3
"""
literature.py — the numbers behind LITERATURE.md: published stolen-base models set against this project's data.

  1. Powers, Ramani, Hahn & Schaefer (2026, arXiv 2601.15608 v2) refit on our steals of second, 2023-26, with their
     filters and centring (github.com/jfhahn2/pickoff-game-theory, wrangle_data.R and estimate_runner_outcome_model.R):
     runner on first only, no 3-2 two-out counts, runners with 3+ attempts; lead - 10, sprint speed - 27, arm - 80;
     season factor; random intercepts for runner, pitcher and catcher. Then the same model + ground gained by release.
  2. Both fit on 2023-25 and scored on the paper's 2026 forward-test attempts (player effects for returning players),
     next to the rows of DF_paper_forward_all.csv, as AUROC and deviance explained (1 - log-loss / league-rate log-loss).
  3. Every model's weights in log-odds per SD of each input on our steals of second.

Usage:  python3 literature.py        (~3 minutes; prints only, writes nothing)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
import paper as P
from core import baseline_rows, res_path

YEARS = ["y2024", "y2025", "y2026"]
SPECS = {"Powers spec (lead, speed, arm)": ["lead_c", "speed_c", "arm_c"],
         "+ ground gained": ["lead_c", "speed_c", "arm_c", "gain_c"]}


def powers_rows(base):
    a = P.success_rows(base)
    a = a[a.arm.notna()].copy()
    a["lead_c"], a["speed_c"], a["arm_c"], a["gain_c"] = a.lead_raw - 10, a.speed_raw - 27, a.arm - 80, a.gain_raw - 10
    for s in (2024, 2025, 2026):
        a[f"y{s}"] = (a.season == s).astype(float)
    return a


def powers_filters(a):
    auto = (a.balls_raw == 3) & (a.strikes_raw == 2) & (a.outs_when_up == 2)
    a = a[a.on_2b.isna() & a.on_3b.isna() & ~auto]
    return a[a.groupby("runner_id").y.transform("size") >= 3].reset_index(drop=True)


def fit(a, terms):
    X, codes, lev = P.design(a, terms)
    return P.GLMM(X, codes, a.y.values).fit(), lev


def main():
    t0 = time.time()
    a = powers_filters(powers_rows("2B")); fits = {}
    print(f"1. Steals of second after Powers' filters: {len(a):,} (success {a.y.mean():.3f})")
    for name, terms in SPECS.items():
        m, _ = fit(a, YEARS + terms); fits[name] = dict(zip(["intercept"] + YEARS + terms, m.beta))
        print(f"   {name}: " + ", ".join(f"{t} {b:+.3f} (SE {s:.3f})" for t, b, s in zip(["intercept"] + YEARS + terms, m.beta, m.se)
                                        if t not in YEARS) + " | SD runner/pitcher/catcher " + "/".join(f"{s:.3f}" for s in m.sigma))

    F = pd.read_csv(res_path("DF_paper_forward_all.csv")).set_index("model"); ll0 = F.loc["League success rate", "log_loss"]
    scored = []
    for base in ["2B", "3B"]:
        b = powers_rows(base); tr, te = b[b.season <= 2025], b[b.season == 2026].copy()
        for name, terms in SPECS.items():
            m, lev = fit(tr, terms); Xte, cte, _ = P.design(te, terms, lev)
            te[name] = m.predict(Xte, cte)
        scored.append(te)
    C = pd.concat(scored); C = C[C.catcher_pop_2b.notna()]           # arm already required: the paper's common rows
    print(f"2. Forward test, {len(C):,} attempts of 2026 (AUROC, deviance explained)")
    for name in SPECS:
        p = np.clip(C[name].values, 1e-6, 1 - 1e-6)
        print(f"   {name} GLMM, new: {roc_auc_score(C.y, p):.4f}, {1 - log_loss(C.y, p) / ll0:.3f}")
    for model, r in F.iterrows():
        print(f"   {model}: {r.auroc:.4f}, {1 - r.log_loss / ll0:.3f}")

    s2 = P.success_rows("2B")
    sd = dict(lead=s2.lead_raw.std(), speed=s2.speed_raw.std(), arm=s2.arm.std(), pop=(s2.pop_raw / 10).std(),
              gain=s2.gain_raw.std(), jump=baseline_rows().jump_speed.std())
    D = pd.read_csv(res_path("DF_paper_decision_fixed.csv")); D = D[(D.base == "2B") & (D.model == "decision")].set_index("term").beta
    S = pd.read_csv(res_path("DF_paper_success_fixed.csv")); S = S[S.base == "2B"].set_index("term").beta
    PW, PG = fits["Powers spec (lead, speed, arm)"], fits["+ ground gained"]
    W = {"Powers et al. v2 (published)": dict(lead=0.14, speed=0.18, arm=-0.06),
         "Powers spec, refit here": dict(lead=PW["lead_c"], speed=PW["speed_c"], arm=PW["arm_c"]),
         "Pitcher's Dilemma generic (published)": dict(lead=0.2356, speed=0.2047, pop=1.5094, jump=0.4583),
         "Whitney-Epstein et al. 2025 (published)": dict(lead=0.0769, speed=0.0770, pop=5.2134),
         "StealIQ decision GLMM": dict(lead=D["lead"], speed=D["speed"], pop=10 * D["pop"]),
         "Powers spec + ground gained": dict(lead=PG["lead_c"], speed=PG["speed_c"], arm=PG["arm_c"], gain=PG["gain_c"]),
         "StealIQ success GLMM": dict(lead=S["lead"], speed=S["speed"], pop=10 * S["pop"], gain=S["gain"])}
    T = pd.DataFrame({k: {i: v * sd[i] for i, v in w.items()} for k, w in W.items()}).T.reindex(columns=list(sd))
    print("3. SDs on our steals of second: " + ", ".join(f"{k} {v:.3f}" for k, v in sd.items()))
    print("   log-odds per SD (per-unit weight x our SD; pop per second, so the SD is in seconds):")
    print(T.round(2).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
