#!/usr/bin/env python3
"""
paper.py — the paper's models (StealIQ_Paper.pdf): RE24 and break-even rates, the per-base success and decision GLMMs
(Laplace maximum likelihood), the 2026 forward test against v16 and the published models, the value of the calls, the
2027 green-light list, and the paper's figures. Run through stealiq.py (decide, success, successfit, popchoice,
paperfigs, glmmcheck). Outputs: results/ and figures/1-Paper-GLMM.
"""
from core import *  # noqa: F401,F403  (paths, loaders, shared models; see core.py)


# ════ 13. Decisions: run expectancy, per-base mixed models, break-even calls ═════════════════════════════════════
# The front office's question is "should this runner go, now, against this battery?" A steal of second and a steal of
# third are different plays (a foot of lead off second is not a foot off first) with different break-even rates, so
# every model here is fit separately by base, and only on what is known before the pitch:
#
#   decision model   logit P(safe) = sprint speed + lead at first move + catcher pop time + balls + strikes + outs
#                    + left-handed pitcher + right-handed batter + random intercepts for runner, pitcher and catcher
#   evaluation model the same + ground gained to release (the secondary lead, measured during the pitch): why it worked
#
# The random intercepts follow Powers et al.: each player's effect is shrunk toward the league according to how often
# the player was seen, which is what makes them usable as skill estimates. GLMM below fits them by maximum likelihood
# with the Laplace approximation (lme4's glmer method) in numpy/scipy; glmm_check() proves it on simulated data.
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import norm as _norm
norm_sf = _norm.sf

SC_GAMES = INGEST / "cache" / "statcast"
GROUPS = ["runner_id", "pitcher_id", "catcher_id"]
DEC_FIXED = ["speed", "lead", "pop", "balls", "strikes", "out1", "out2", "lhp", "rhb"]
TERM_LABEL = {"speed": "sprint speed (per ft/s)", "lead": "lead at first move (per ft)", "pop": "catcher pop time (per 0.1 s)",
              "balls": "balls in the count (per ball)", "strikes": "strikes in the count (per strike)",
              "out1": "one out (vs none)", "out2": "two outs (vs none)", "lhp": "left-handed pitcher",
              "rhb": "right-handed batter", "gain": "ground gained to release (per ft)"}


def re24_counts(innings: int = 8) -> pd.DataFrame:
    """Runs scored from the start of each plate appearance to the end of its half-inning, summed by season and base-out
    state, from the cached Statcast pitch files of every game with a tracked attempt (innings 1-8: later ones can end on
    a walk-off before three outs). Cached in DF_paper_re24.csv."""
    out = res_path("DF_paper_re24.csv")
    if out.exists():
        return pd.read_csv(out, dtype={"state": str})
    cols = ["game_type", "game_year", "inning", "inning_topbot", "at_bat_number", "pitch_number", "outs_when_up",
            "on_1b", "on_2b", "on_3b", "bat_score", "post_bat_score"]
    pa, games = [], {}
    for f in sorted(SC_GAMES.glob("*.csv.gz")):
        d = pd.read_csv(f, usecols=cols)
        d = d[(d.game_type == "R") & (d.inning <= innings)]
        if d.empty:
            continue
        games[int(d.game_year.iloc[0])] = games.get(int(d.game_year.iloc[0]), 0) + 1
        d["runs"] = d.groupby(["inning", "inning_topbot"]).post_bat_score.transform("max") - d.bat_score
        pa.append(d.sort_values("pitch_number").groupby("at_bat_number").head(1))
    pa = pd.concat(pa)
    pa["state"] = (np.where(pa.on_1b.notna(), "1", "_").astype(object) + np.where(pa.on_2b.notna(), "2", "_")
                   + np.where(pa.on_3b.notna(), "3", "_"))
    t = (pa.groupby(["game_year", "state", "outs_when_up"]).runs.agg(runs="sum", n="size").reset_index()
         .rename(columns={"game_year": "season", "outs_when_up": "outs"}))
    t["games"] = t.season.map(games)
    t.to_csv(out, index=False)
    print(f"RE24 from {sum(games.values()):,} games ({games}), {len(pa):,} plate appearances")
    return t


def re24(seasons) -> dict:
    """{(state, outs): expected runs to the end of the half-inning} pooled over the given seasons."""
    t = re24_counts(); t = t[t.season.isin(list(seasons))].groupby(["state", "outs"])[["runs", "n"]].sum()
    return (t.runs / t.n).to_dict()


def steal_values(a: pd.DataFrame, RE: dict) -> pd.DataFrame:
    """Expected runs now, after a stolen base and after a caught stealing, and the break-even success rate
    BE = (RE_now - RE_cs) / (RE_sb - RE_cs). A stolen base moves the runner up one base (a runner already on the target
    base moves up too, as in a double steal, scoring from third); a caught stealing removes the runner and adds an out."""
    occ = np.c_[a.on_1b.notna(), a.on_2b.notna(), a.on_3b.notna()]
    src = np.where(a.base.values == "2B", 0, 1)
    key = lambda b, o: ("".join(c if x else "_" for c, x in zip("123", b)), int(o))
    now, sb, cs = [], [], []
    for b, s, o in zip(occ, src, a.outs_when_up.values):
        b = b.copy(); b[s] = True                              # the runner is on the origin base
        n, runs = b.copy(), 0.0
        n[s] = False
        if n[s + 1]:                                           # double steal: the lead runner moves up as well
            if s + 2 == 3:
                runs += 1
            else:
                n[s + 2] = True
        n[s + 1] = True
        c = b.copy(); c[s] = False
        now.append(RE[key(b, o)]); sb.append(RE[key(n, o)] + runs); cs.append(0.0 if o == 2 else RE[key(c, o + 1)])
    v = pd.DataFrame({"re_now": now, "re_sb": sb, "re_cs": cs}, index=a.index)
    v["breakeven"] = (v.re_now - v.re_cs) / (v.re_sb - v.re_cs)
    return v


class GLMM:
    """Logistic GLMM with crossed random intercepts by maximum likelihood (Laplace approximation, as lme4's glmer).

    Random effects are u_g = sigma_g * v with v ~ N(0, I) (lme4's spherical form). For given parameters the conditional
    mode comes from penalized iteratively reweighted least squares; the objective is the Laplace deviance
        -2 log L  =  -2 sum log p(y | beta, u)  +  |v|^2  +  log det(Lam Z'W Z Lam + I).
    sigma is found with beta estimated inside the PIRLS step (lme4's nAGQ = 0); refine() then optimizes beta and sigma
    together (nAGQ = 1). Fixed-effect SEs are conditional on sigma, as lme4 reports them."""

    def __init__(self, X, codes, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        self.n, self.p = self.X.shape
        mats = [sp.csc_matrix((np.ones(self.n), (np.arange(self.n), c)), shape=(self.n, int(c.max()) + 1)) for c in codes]
        self.Z, self.sizes = sp.hstack(mats).tocsc(), [m.shape[1] for m in mats]
        self.gcol, self.q, self.G = np.repeat(np.arange(len(mats)), self.sizes), sum(self.sizes), len(mats)

    def _mode(self, lam, beta=None, start=None):
        """PIRLS: the conditional mode of (beta, v) (beta=None) or of v given beta."""
        ZL = self.Z @ sp.diags(lam)
        A = (sp.hstack([sp.csc_matrix(self.X), ZL]) if beta is None else ZL).tocsr()
        off = 0.0 if beta is None else self.X @ beta
        pen = np.r_[np.zeros(self.p), np.ones(self.q)] if beta is None else np.ones(self.q)
        g = np.zeros(A.shape[1]) if start is None else start.copy()
        pdev = lambda gg: -2 * np.sum(self.y * (A @ gg + off) - np.logaddexp(0, A @ gg + off)) + pen @ (gg * gg)
        d0 = pdev(g)
        for _ in range(200):          # the system is ~2,000 wide and fills in: dense Cholesky beats sparse LU here
            mu = expit(A @ g + off)
            H = (A.T @ sp.diags(mu * (1 - mu)) @ A).toarray()
            H[np.diag_indices_from(H)] += pen + 1e-10
            step, t = cho_solve(cho_factor(H, check_finite=False), A.T @ (self.y - mu) - pen * g, check_finite=False), 1.0
            while (d1 := pdev(g + t * step)) > d0 and t > 1e-8:
                t /= 2
            g, gain, d0 = g + t * step, d0 - d1, d1
            if gain < 1e-10:
                break
        return g

    def _deviance(self, lam, beta, v):
        ZL = self.Z @ sp.diags(lam); eta = self.X @ beta + ZL @ v; mu = expit(eta)
        M = (ZL.T @ sp.diags(mu * (1 - mu)) @ ZL).toarray()
        M[np.diag_indices_from(M)] += 1.0
        c, _ = cho_factor(M, lower=True, check_finite=False)
        return -2 * np.sum(self.y * eta - np.logaddexp(0, eta)) + v @ v + 2 * np.sum(np.log(np.diag(c)))

    def fit(self, start=None):
        lam = lambda ls: np.exp(ls)[self.gcol]
        memo = {"g": None}
        def f(ls):
            memo["g"] = self._mode(lam(ls), start=memo["g"])
            return self._deviance(lam(ls), memo["g"][:self.p], memo["g"][self.p:])
        r = minimize(f, np.full(self.G, np.log(0.3)) if start is None else start, method="L-BFGS-B",
                     bounds=[(np.log(1e-3), np.log(5.0))] * self.G, options=dict(eps=1e-4, ftol=1e-8, gtol=1e-3, maxiter=200))
        self.logsig, self.dev = r.x, r.fun
        g = self._mode(lam(r.x))
        self.beta, self.v = g[:self.p], g[self.p:]
        self._finish(lam(r.x))
        return self

    def refine(self):
        """nAGQ = 1: beta and log sigma optimized together, v at its conditional mode given beta."""
        lam = lambda ls: np.exp(ls)[self.gcol]
        memo = {"v": self.v}
        def f(par):
            memo["v"] = self._mode(lam(par[self.p:]), beta=par[:self.p], start=memo["v"])
            return self._deviance(lam(par[self.p:]), par[:self.p], memo["v"])
        r = minimize(f, np.r_[self.beta, self.logsig], method="L-BFGS-B", options=dict(eps=1e-5, maxiter=500))
        self.beta, self.logsig, self.dev = r.x[:self.p], r.x[self.p:], r.fun
        self.v = self._mode(lam(self.logsig), beta=self.beta, start=memo["v"])
        self._finish(lam(self.logsig))
        return self

    def _finish(self, lam):
        ZL = self.Z @ sp.diags(lam)
        A = sp.hstack([sp.csc_matrix(self.X), ZL]).tocsc()
        mu = expit(A @ np.r_[self.beta, self.v])
        H = (A.T @ sp.diags(mu * (1 - mu)) @ A).toarray()
        H[np.diag_indices_from(H)] += np.r_[np.zeros(self.p), np.ones(self.q)]
        Hi = cho_solve(cho_factor(H, check_finite=False), np.eye(len(H)), check_finite=False)
        self.sigma, self.se = np.exp(self.logsig), np.sqrt(np.diag(Hi)[:self.p])
        u, usd = lam * self.v, lam * np.sqrt(np.diag(Hi)[self.p:])
        cut = np.cumsum([0] + self.sizes)
        self.u = [u[a:b] for a, b in zip(cut[:-1], cut[1:])]; self.u_sd = [usd[a:b] for a, b in zip(cut[:-1], cut[1:])]
        self.loglik = -0.5 * self.dev

    def simulate(self, rng):
        """y drawn from the fitted model with fresh random effects (for the recovery check and the bootstrap)."""
        u = rng.normal(0, 1, self.q) * np.exp(self.logsig)[self.gcol]
        return (rng.random(self.n) < expit(self.X @ self.beta + self.Z @ u)).astype(float)

    def predict(self, X, codes):
        """P(safe) for new rows; codes are the training levels, -1 for a player not seen in training (effect 0)."""
        eta = np.asarray(X, float) @ self.beta
        for u, c in zip(self.u, codes):
            eta = eta + np.where(c >= 0, u[np.maximum(c, 0)], 0.0)
        return expit(eta)


def dec_rows(base: str, evaluation: bool = False) -> pd.DataFrame:
    """Steal attempts of one base with the decision inputs (centred on the 2023-26 means of this base; pop time in tenths
    of a second, to the base being stolen) and the break-even rate of each attempt's base-out state (RE24 2023-26).
    Steals of third use pop time to third: on the attempts with both, it fits them better than pop time to second
    (decision-model deviance 6.2 lower at equal parameters)."""
    m = load_meta()
    a = m[m.result.isin(["SB", "CS"]) & (m.base == base)].copy()
    popc = "catcher_pop_2b" if base == "2B" else "catcher_pop_3b"
    a = a.dropna(subset=["runner_sprint_speed", "lead_at_firstmove_ft", popc, "balls", "strikes",
                         "outs_when_up", "p_throws", "stand"] + (["gain_to_release_ft"] if evaluation else []))
    a["speed"], a["lead"], a["pop"] = a.runner_sprint_speed, a.lead_at_firstmove_ft, 10 * a[popc]
    a["out1"], a["out2"] = (a.outs_when_up == 1).astype(float), (a.outs_when_up == 2).astype(float)
    a["lhp"], a["rhb"] = (a.p_throws == "L").astype(float), (a.stand == "R").astype(float)
    a["gain"] = a.gain_to_release_ft
    for c in ["speed", "lead", "pop", "balls", "strikes", "gain"]:
        a[c + "_raw"] = a[c]; a[c] = a[c] - a[c].mean()
    return a.join(steal_values(a, re24(range(2023, 2027))))


def state_rows(base: str) -> pd.DataFrame:
    """Every tracked attempt of one base with a known pre-pitch base-out state: the denominator for observed success by
    state, which should not depend on which attempts have every model input."""
    m = load_meta()
    return m[m.result.isin(["SB", "CS"]) & (m.base == base)].dropna(subset=["outs_when_up"])


def design(a: pd.DataFrame, terms: list, levels: dict | None = None):
    """X with an intercept, and integer codes per grouping factor (training levels; -1 for players unseen in training)."""
    X = np.c_[np.ones(len(a)), a[terms].values]
    if levels is None:
        levels = {g: pd.Index(sorted(a[g].unique())) for g in GROUPS}
    codes = [levels[g].get_indexer(a[g]) for g in GROUPS]
    return X, codes, levels


def glmm_check(base="2B", n_sim=20, seed=0):
    """Proof of the GLMM code: fit the 2B decision model, simulate n_sim data sets from it (same design, same players,
    fresh random effects), refit each, and report the mean estimate and spread against the truth; plus statsmodels'
    variational-Bayes fit of the same model on the real data as an independent cross-check."""
    a = dec_rows(base); X, codes, _ = design(a, DEC_FIXED)
    t0 = time.time(); m = GLMM(X, codes, a.y.values).fit(); t_fit = time.time() - t0
    rng, est = np.random.default_rng(seed), []
    for k in range(n_sim):
        s = GLMM(X, codes, m.simulate(rng)).fit(start=m.logsig)
        est.append(np.r_[s.beta, s.sigma])
    est, truth = np.array(est), np.r_[m.beta, m.sigma]
    names = ["intercept"] + DEC_FIXED + [f"sigma {g.split('_')[0]}" for g in GROUPS]
    rec = pd.DataFrame({"param": names, "truth": truth, "mean_est": est.mean(0), "sd_est": est.std(0, ddof=1)})
    rec["bias_in_sd"] = (rec.mean_est - rec.truth) / (rec.sd_est / np.sqrt(n_sim))
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    vc = {g: f"0 + C({g})" for g in GROUPS}
    df = a[["y"] + DEC_FIXED + GROUPS].copy()
    vb = BinomialBayesMixedGLM.from_formula("y ~ " + " + ".join(DEC_FIXED), vc, df).fit_vb()
    rec["statsmodels_vb"] = np.r_[vb.fe_mean, np.exp(vb.vcp_mean)]
    rec.to_csv(res_path("DF_paper_glmm_check.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"{base}: n={len(a):,}, levels {dict(zip(GROUPS, m.sizes))}; one fit {t_fit:.1f} s")
    print(rec.round(4).to_string(index=False))
    return rec


def sigma_ci(m: GLMM):
    """95% likelihood-ratio intervals for the random-effect SDs: the values of sigma at which the profiled Laplace
    deviance rises 3.84 above its minimum, each SD varied with the others held at their estimates. Unlike a Wald interval
    on log sigma this stays sensible for small SDs; an SD whose lower limit reaches zero is reported as 0 to the upper."""
    lam = lambda ls: np.exp(ls)[m.gcol]
    def d(ls):
        g = m._mode(lam(ls), start=np.r_[m.beta, m.v])
        return m._deviance(lam(ls), g[:m.p], g[m.p:])
    out = np.zeros((m.G, 2))
    for gi in range(m.G):
        def rise(sig):
            x = m.logsig.copy(); x[gi] = np.log(sig)
            return d(x) - m.dev - 3.84
        s0 = m.sigma[gi]
        if rise(1e-3) < 0:
            lo = 0.0
        else:
            a, b = 1e-3, s0
            for _ in range(22):
                mid = np.sqrt(a * b); a, b = (mid, b) if rise(mid) > 0 else (a, mid)
            lo = b
        a, b = max(s0, 1e-3), max(4 * s0, 0.5)
        while rise(b) < 0 and b < 20:
            b *= 2
        for _ in range(22):
            mid = np.sqrt(a * b); a, b = (a, mid) if rise(mid) > 0 else (mid, b)
        out[gi] = [lo, b]
    return out


def first_last(name):
    """'Kelly, Merrill' -> 'Merrill Kelly' (Savant's pitcher and catcher names are last, first)."""
    return " ".join(reversed(str(name).split(", "))) if ", " in str(name) else str(name)


def run_decide(seed: int = 0):
    """Per base: the decision and evaluation GLMMs on 2023-26 (fixed effects, variance components, player effects);
    the forward test (fit 2023-25, score 2026) against simpler models; the value of the model's go / hold calls on the
    2026 attempts at each state's break-even rate; and the 2027 green-light list."""
    pp = lambda b0, d: 100 * (expit(b0 + d) - expit(b0))           # change in P(safe), points, from the league average
    fixed, var, players, fwd, cal, value, green, be_tab = [], [], [], [], [], [], [], []
    rng = np.random.default_rng(seed)
    RE_all, RE_tr = re24(range(2023, 2027)), re24(range(2023, 2026))
    for base in ["2B", "3B"]:
        src = "1__" if base == "2B" else "_2_"
        sv = steal_values(pd.DataFrame({"on_1b": [1 if base == "2B" else None] * 3, "on_2b": [1 if base == "3B" else None] * 3,
                                        "on_3b": [None] * 3, "base": [base] * 3, "outs_when_up": [0, 1, 2]}), RE_all)
        for o, r in zip([0, 1, 2], sv.itertuples()):
            be_tab.append(dict(base=base, state=src, outs=o, re_now=r.re_now, re_sb=r.re_sb, re_cs=r.re_cs, breakeven=r.breakeven))
        fits = {}
        for model, terms in [("decision", DEC_FIXED), ("evaluation", DEC_FIXED + ["gain"])]:
            a = dec_rows(base, evaluation=(model == "evaluation"))
            X, codes, lev = design(a, terms)
            m = GLMM(X, codes, a.y.values).fit()
            fits[model] = (a, m, lev, terms)
            b0, ci = m.beta[0], sigma_ci(m)
            for j, t in enumerate(["intercept"] + terms):
                z = m.beta[j] / m.se[j]
                binary = t in ("out1", "out2", "lhp", "rhb", "intercept")
                p10, p90 = (0, 1) if binary else a[t + "_raw"].quantile([0.1, 0.9])
                fixed.append(dict(base=base, model=model, term=t, label=TERM_LABEL.get(t, t), n=len(a), beta=m.beta[j], se=m.se[j],
                                  lo=m.beta[j] - 1.96 * m.se[j], hi=m.beta[j] + 1.96 * m.se[j], p=2 * norm_sf(abs(z)),
                                  pp_per_unit=np.nan if t == "intercept" else pp(b0, m.beta[j]),
                                  pp_lo=np.nan if t == "intercept" else pp(b0, m.beta[j] - 1.96 * m.se[j]),
                                  pp_hi=np.nan if t == "intercept" else pp(b0, m.beta[j] + 1.96 * m.se[j]),
                                  p10=p10, p90=p90, pp_10_90=np.nan if t == "intercept" else pp(b0, m.beta[j] * (p90 - p10))))
            for gi, g in enumerate(GROUPS):
                var.append(dict(base=base, model=model, group=g.split("_")[0], levels=m.sizes[gi], sigma=m.sigma[gi],
                                lo=ci[gi, 0], hi=ci[gi, 1], pp_plus_1sd=pp(b0, m.sigma[gi]), pp_minus_1sd=pp(b0, -m.sigma[gi]),
                                pp_plus_hi=pp(b0, ci[gi, 1]), pp_plus_lo=pp(b0, ci[gi, 0])))
                name = a.groupby(g)[g.replace("_id", "_name")].first().map(first_last)
                cnt = a.groupby(g).size()
                for k, pid in enumerate(lev[g]):
                    players.append(dict(base=base, model=model, group=g.split("_")[0], id=pid, name=name[pid], attempts=cnt[pid],
                                        effect=m.u[gi][k], effect_sd=m.u_sd[gi][k], pp_vs_avg=pp(b0, m.u[gi][k])))
            print(f"{base} {model}: n={len(a):,}, sigma runner/pitcher/catcher {np.round(m.sigma, 3)}, "
                  f"league P {expit(b0):.3f}; " + ", ".join(f"{t} {m.beta[j + 1]:+.3f}" for j, t in enumerate(terms)))

        # forward test: fit 2023-25, score 2026 (the use case: next season's calls from past seasons)
        a = fits["decision"][0]
        tr, te = a[a.season <= 2025], a[a.season == 2026]
        Xtr, ctr, lev = design(tr, DEC_FIXED); Xte, cte, _ = design(te, DEC_FIXED, lev)
        g = GLMM(Xtr, ctr, tr.y.values).fit()
        preds = {"league rate": np.full(len(te), tr.y.mean()),
                 "sprint speed only": sm.Logit(tr.y.values, sm.add_constant(tr[["speed"]].values)).fit(disp=0)
                     .predict(sm.add_constant(te[["speed"]].values)),
                 "decision inputs, no player effects": sm.Logit(tr.y.values, Xtr).fit(disp=0).predict(Xte),
                 "decision GLMM": g.predict(Xte, cte)}
        y, runners = te.y.values, te.runner_id.values
        ur = np.unique(runners); idx = {r: np.where(runners == r)[0] for r in ur}
        for name, pr in preds.items():
            au = roc_auc_score(y, pr) if np.ptp(pr) > 1e-9 else 0.5
            boots = []
            for _ in range(500 if np.ptp(pr) > 1e-9 else 0):
                ii = np.concatenate([idx[r] for r in rng.choice(ur, len(ur))])
                if y[ii].min() != y[ii].max():
                    boots.append(roc_auc_score(y[ii], pr[ii]))
            lg = np.log(pr / (1 - pr))
            cal_fit = sm.Logit(y, sm.add_constant(lg)).fit(disp=0).params if np.ptp(pr) > 1e-9 else [np.nan, np.nan]
            fwd.append(dict(base=base, model=name, n_train=len(tr), n_test=len(te), log_loss=log_loss(y, pr),
                            brier=brier_score_loss(y, pr), auroc=au, auroc_lo=np.percentile(boots, 2.5) if boots else np.nan,
                            auroc_hi=np.percentile(boots, 97.5) if boots else np.nan, cal_intercept=cal_fit[0], cal_slope=cal_fit[1],
                            new_runners=float(np.mean(cte[0] < 0)), new_pitchers=float(np.mean(cte[1] < 0))))
            q = pd.qcut(pr, 10, labels=False, duplicates="drop") if np.ptp(pr) > 1e-9 else np.zeros(len(pr), int)
            for b, ii in pd.Series(np.arange(len(pr))).groupby(q):
                cal.append(dict(base=base, model=name, bin=b, n=len(ii), mean_pred=pr[ii].mean(), observed=y[ii].mean()))
        # the calls: go when P(safe) >= the state's break-even (RE24 2023-25), scored on what 2026 attempts produced
        s26 = steal_values(te, RE_tr)
        pg = preds["decision GLMM"]
        realized = np.where(y == 1, s26.re_sb, s26.re_cs) - s26.re_now
        expected = pg * s26.re_sb + (1 - pg) * s26.re_cs - s26.re_now
        margin = 100 * (pg - s26.breakeven.values)
        bins = pd.cut(margin, [-100, -5, 0, 5, 10, 100], labels=["< -5", "-5 to 0", "0 to 5", "5 to 10", "> 10"])
        for lab, ii in pd.Series(np.arange(len(y))).groupby(bins, observed=True):
            ii = ii.values
            value.append(dict(base=base, margin_pts=str(lab), n=len(ii), success=y[ii].mean(), mean_p=pg[ii].mean(),
                              mean_be=s26.breakeven.values[ii].mean(), runs_per_attempt=realized.values[ii].mean(),
                              runs_se=realized.values[ii].std(ddof=1) / np.sqrt(len(ii)), expected_runs=expected.values[ii].mean(),
                              total_runs=realized.values[ii].sum()))
        for lab, ii in [("go (P >= break-even)", np.where(margin >= 0)[0]), ("hold (P < break-even)", np.where(margin < 0)[0])]:
            value.append(dict(base=base, margin_pts=lab, n=len(ii), success=y[ii].mean(), mean_p=pg[ii].mean(),
                              mean_be=s26.breakeven.values[ii].mean(), runs_per_attempt=realized.values[ii].mean(),
                              runs_se=realized.values[ii].std(ddof=1) / np.sqrt(len(ii)), expected_runs=expected.values[ii].mean(),
                              total_runs=realized.values[ii].sum()))

        # 2027 green light: each recent runner against an average battery (and a tough one), by outs
        a, m, lev, terms = fits["decision"]
        rec = a[a.season >= 2025]
        min_n = 10 if base == "2B" else 4
        b0 = m.beta[0]; B = dict(zip(terms, m.beta[1:]))
        be = {r["outs"]: r["breakeven"] for r in be_tab if r["base"] == base}
        for rid, d in rec.groupby("runner_id"):
            if len(d) < min_n or rid not in lev["runner_id"]:
                continue
            k = lev["runner_id"].get_loc(rid)
            u = m.u[0][k]
            sp_ = d.sort_values("season").speed.iloc[-1]; ld = d.lead.mean()
            eta = b0 + B["speed"] * sp_ + B["lead"] * ld + B["rhb"] * a.rhb.mean() + u   # average count and batter mix, vs a RHP
            row = dict(base=base, runner_id=rid, name=d.runner_name.iloc[0], attempts_2025_26=len(d), success_2025_26=d.y.mean(),
                       speed=d.sort_values("season").speed_raw.iloc[-1], lead=d.lead_raw.mean(), runner_effect=u, runner_effect_sd=m.u_sd[0][k])
            for o, ob in [(0, 0.0), (1, B["out1"]), (2, B["out2"])]:
                row[f"p_{o}out"] = expit(eta + ob); row[f"be_{o}out"] = be[o]
                row[f"p_tough_{o}out"] = expit(eta + ob - m.sigma[1] - m.sigma[2])
            row["green_outs"] = sum(row[f"p_{o}out"] >= be[o] for o in range(3))
            row["green_outs_tough"] = sum(row[f"p_tough_{o}out"] >= be[o] for o in range(3))
            green.append(row)

    out = {"DF_paper_breakeven.csv": be_tab, "DF_paper_decision_fixed.csv": fixed, "DF_paper_decision_variance.csv": var, "DF_paper_decision_players.csv": players,
           "DF_paper_decision_forward.csv": fwd, "DF_paper_decision_calibration.csv": cal, "DF_paper_decision_value.csv": value, "DF_paper_greenlight.csv": green}
    pd.set_option("display.width", 220)
    for f, rows in out.items():
        pd.DataFrame(rows).to_csv(res_path(f), index=False)
    print(pd.DataFrame(fwd).round(4).to_string(index=False))
    print(pd.DataFrame(value).round(4).to_string(index=False))
    print(pd.DataFrame(var).round(3).to_string(index=False))
    return out


# ── 13c. The success model: everything measured through release, per base, with player effects ───────────────────
# The decision model above is limited, by design, to what is known before the runner goes. The success model adds what
# the play itself measures up to the release of the pitch: the ground gained (with curvature, for diminishing returns)
# and the pitch class. It is the best predictor of whether a steal succeeds and the basis for rating runners, pitchers
# and catchers; it is not a pre-pitch decision tool. Its specification is chosen by cross-validation on 2023-25 only and
# tested once on 2026, against v16 (pooled logistic, the previous best), the published specifications and speed alone.
SUCCESS_SPECS = {"v16 inputs": ["speed", "lead", "gain", "pop", "brk", "off"],
                 "v16 inputs + count, outs, hands": DEC_FIXED + ["gain", "brk", "off"],
                 "+ curvature in ground gained": DEC_FIXED + ["gain", "gain2", "brk", "off"]}
TERM_LABEL.update({"gain2": "ground gained, squared (per 10 ft²)", "brk": "breaking ball (vs fastball)",
                   "off": "offspeed pitch (vs fastball)"})


def success_rows(base: str) -> pd.DataFrame:
    """Attempts of one base with every input through release: the decision inputs, ground gained (centred) and its
    square over 10, pitch class (two dummies, fastball the reference), and the published models' inputs (catcher arm,
    the Pitcher's Dilemma jump speed) for the comparison."""
    a = dec_rows(base, evaluation=True)
    pc = load_context()[["play_id", "pitch_code"]]
    a = a.merge(pc, on="play_id", how="left")
    a["pitch_class"] = a.pitch_code.map(PITCH_CLASS)
    a = a[a.pitch_class.notna()].copy()
    a["brk"], a["off"] = (a.pitch_class == "breaking").astype(float), (a.pitch_class == "offspeed").astype(float)
    a["gain2"] = a.gain ** 2 / 10
    pub = baseline_rows()[["play_id", "arm", "jump_speed"]]
    return a.merge(pub, on="play_id", how="left").reset_index(drop=True)


def success_cv(k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Each candidate specification, k-fold cross-validated on 2023-25 attempts only (log-loss, AUROC), per base."""
    out = []
    for base in ["2B", "3B"]:
        a = success_rows(base); a = a[a.season <= 2025].reset_index(drop=True)
        for name, terms in SUCCESS_SPECS.items():
            p = np.zeros(len(a))
            for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(a, a.y):
                Xtr, ctr, lev = design(a.iloc[tr], terms); Xte, cte, _ = design(a.iloc[te], terms, lev)
                p[te] = GLMM(Xtr, ctr, a.y.values[tr]).fit().predict(Xte, cte)
            out.append(dict(base=base, spec=name, n=len(a), log_loss=log_loss(a.y, p), auroc=roc_auc_score(a.y, p)))
            print(f"  cv {base} {name}: log-loss {out[-1]['log_loss']:.4f}, AUROC {out[-1]['auroc']:.4f}", flush=True)
    S = pd.DataFrame(out)
    pooled = S.assign(w=S.log_loss * S.n).groupby("spec")[["w", "n"]].sum()
    S["pooled_log_loss"] = S.spec.map(pooled.w / pooled.n)
    S["chosen"] = S.spec == (pooled.w / pooled.n).idxmin()
    return S


def run_success(n_boot: int = 500, seed: int = 0):
    """Choose the success model by cross-validation on 2023-25, test it on 2026 against every other model on the same
    attempts, fit it on 2023-26, and work out the ground a steal needs by release to clear each break-even rate."""
    S = success_cv(seed=seed); S.to_csv(res_path("DF_paper_success_cv.csv"), index=False)
    chosen = S[S.chosen].spec.iloc[0]; terms = SUCCESS_SPECS[chosen]
    print(f"chosen on 2023-25: {chosen}")
    # 2026 forward test, every model on the same attempts
    tests = []
    for base in ["2B", "3B"]:
        a = success_rows(base); tr, te = a[a.season <= 2025], a[a.season == 2026].copy()
        for col, tt in [("p_success", terms), ("p_decision", DEC_FIXED)]:
            Xtr, ctr, lev = design(tr, tt); Xte, cte, _ = design(te, tt, lev)
            te[col] = GLMM(Xtr, ctr, tr.y.values).fit().predict(Xte, cte)
        tests.append(te)
    T = pd.concat(tests).reset_index(drop=True)
    both = pd.concat([success_rows(b).assign(b3=float(b == "3B")) for b in ["2B", "3B"]])
    both["pop_s"] = both.catcher_pop_2b                      # v16 and the Pitcher's Dilemma: pop time to second, as published
    tr = both[(both.season <= 2025) & both.pop_s.notna()]; T = T.merge(both[["play_id", "b3", "pop_s"]], on="play_id", how="left")
    ok = T.arm.notna() & T.pop_s.notna()                     # the common rows: every model can score them
    V16 = ["speed_raw", "lead_raw", "gain_raw", "pop_s", "brk", "off", "b3"]          # FEATS order: 4 continuous, 3 binary
    import ast
    tuned = ast.literal_eval(pd.read_csv(res_path("DF_v16_forward.csv")).set_index("model").loc["v16 tuned on 2023-25 only", "tuned_params"])
    fitp = lambda est, cols, rows: est.fit(rows[cols].values.astype(float), rows.y.values)
    lr = lambda _: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    pub_tr = tr.dropna(subset=["arm"])
    T["p_v16"] = np.nan; T["p_v16_tuned"] = np.nan
    T.loc[ok, "p_v16"] = fitp(pipe(DEFAULT), V16, tr).predict_proba(T.loc[ok, V16].values.astype(float))[:, 1]
    T.loc[ok, "p_v16_tuned"] = fitp(pipe({**DEFAULT, **tuned}), V16, tr).predict_proba(T.loc[ok, V16].values.astype(float))[:, 1]
    T["p_speed"] = fitp(lr(None), ["speed_raw"], tr).predict_proba(T[["speed_raw"]].values)[:, 1]
    T["p_powers"] = np.nan; T["p_dilemma"] = np.nan
    T.loc[ok, "p_powers"] = fitp(lr(None), ["lead_raw", "speed_raw", "arm"], pub_tr).predict_proba(T.loc[ok, ["lead_raw", "speed_raw", "arm"]].values)[:, 1]
    T.loc[ok, "p_dilemma"] = fitp(lr(None), ["lead_raw", "speed_raw", "jump_speed", "pop_s"], pub_tr).predict_proba(
        T.loc[ok, ["lead_raw", "speed_raw", "jump_speed", "pop_s"]].values)[:, 1]
    T["p_league"] = tr.y.mean()
    models = [("p_league", "League success rate", "baseline"), ("p_speed", "Sprint speed only", "baseline"),
              ("p_powers", "Powers et al. specification (lead, speed, arm), refit", "published"),
              ("p_dilemma", "Pitcher's Dilemma specification (lead, speed, jump, pop), refit", "published"),
              ("p_decision", "Decision GLMM: before the pitch only", "this paper"),
              ("p_v16", "v16 logistic (pooled, default)", "v16"), ("p_v16_tuned", "v16 logistic (pooled, tuned on 2023-25)", "v16"),
              ("p_success", f"Success GLMM: {chosen}", "this paper")]
    C = T[ok].reset_index(drop=True)                       # the common rows: every model scores every attempt
    y = C.y.values; grp = C.groupby("runner_id").indices; keys = np.array(list(grp)); rng = np.random.default_rng(seed)
    boots = [np.concatenate([grp[k] for k in rng.choice(keys, len(keys))]) for _ in range(n_boot)]
    fwd, roc = [], []
    for col, name, fam in models:
        p = np.clip(C[col].values, 1e-6, 1 - 1e-6); flat = np.ptp(p) < 1e-9
        au = 0.5 if flat else roc_auc_score(y, p)
        b = [] if flat else [roc_auc_score(y[i], p[i]) for i in boots]
        cal = [np.nan, np.nan] if flat else sm.Logit(y, sm.add_constant(np.log(p / (1 - p)))).fit(disp=0).params
        d_v16 = [roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], C.p_v16_tuned.values[i]) for i in boots] if not flat else []
        fwd.append(dict(model=name, family=fam, n=len(y), n_2b=int((C.base == "2B").sum()), n_3b=int((C.base == "3B").sum()),
                        auroc=au, auroc_lo=np.percentile(b, 2.5) if b else np.nan, auroc_hi=np.percentile(b, 97.5) if b else np.nan,
                        auroc_2b=roc_auc_score(y[C.base == "2B"], p[C.base == "2B"]) if not flat else 0.5,
                        auroc_3b=roc_auc_score(y[C.base == "3B"], p[C.base == "3B"]) if not flat else 0.5,
                        log_loss=log_loss(y, p), brier=brier_score_loss(y, p), cal_intercept=cal[0], cal_slope=cal[1],
                        d_auroc_vs_v16_tuned=np.mean(d_v16) if d_v16 else np.nan,
                        d_lo=np.percentile(d_v16, 2.5) if d_v16 else np.nan, d_hi=np.percentile(d_v16, 97.5) if d_v16 else np.nan))
        if not flat:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y, p); g = np.linspace(0, 1, 201)
            roc += [dict(model=name, fpr=f_, tpr=np.interp(f_, fpr, tpr)) for f_ in g]
        q = pd.qcut(p, 10, labels=False, duplicates="drop") if not flat else np.zeros(len(p), int)
        for k_, ii in pd.Series(np.arange(len(p))).groupby(q):
            fwd[-1].setdefault("_cal", []).append((k_, len(ii), p[ii].mean(), y[ii].mean()))
    cal_rows = [dict(model=r["model"], bin=b_, n=n_, mean_pred=mp, observed=ob) for r in fwd for (b_, n_, mp, ob) in r.pop("_cal")]
    F = pd.DataFrame(fwd); F.to_csv(res_path("DF_paper_forward_all.csv"), index=False)
    pd.DataFrame(roc).to_csv(res_path("DF_paper_roc.csv"), index=False); pd.DataFrame(cal_rows).to_csv(res_path("DF_paper_calibration_all.csv"), index=False)
    pd.set_option("display.width", 230)
    print(F[["model", "n", "auroc", "auroc_lo", "auroc_hi", "auroc_2b", "auroc_3b", "log_loss", "cal_slope", "d_auroc_vs_v16_tuned", "d_lo", "d_hi"]].round(4).to_string(index=False))
    success_fit(terms)
    return F


def success_fit(terms=None):
    """The chosen success model on 2023-26 (estimates, variance components, player effects) and the ground a steal
    needs by release to clear each break-even rate. Reads the chosen specification from DF_paper_success_cv.csv."""
    pp = lambda b0, d: 100 * (expit(b0 + d) - expit(b0))
    if terms is None:
        S = pd.read_csv(res_path("DF_paper_success_cv.csv")); terms = SUCCESS_SPECS[S[S.chosen].spec.iloc[0]]
    fixed, var, players, ground = [], [], [], []
    BE = re24(range(2023, 2027))
    for base in ["2B", "3B"]:
        a = success_rows(base); X, codes, lev = design(a, terms); m = GLMM(X, codes, a.y.values).fit(); ci = sigma_ci(m); b0 = m.beta[0]
        for j, t in enumerate(["intercept"] + terms):
            binary = t in ("out1", "out2", "lhp", "rhb", "brk", "off", "intercept", "gain2")
            p10, p90 = (0, 1) if binary else a[t + "_raw"].quantile([0.1, 0.9])
            fixed.append(dict(base=base, term=t, label=TERM_LABEL.get(t, t), n=len(a), beta=m.beta[j], se=m.se[j],
                              p=2 * norm_sf(abs(m.beta[j] / m.se[j])), p10=p10, p90=p90))
        for gi, g in enumerate(GROUPS):
            var.append(dict(base=base, group=g.split("_")[0], levels=m.sizes[gi], sigma=m.sigma[gi], lo=ci[gi, 0], hi=ci[gi, 1],
                            pp_plus_1sd=pp(b0, m.sigma[gi]), pp_plus_lo=pp(b0, ci[gi, 0]), pp_plus_hi=pp(b0, ci[gi, 1])))
            name = a.groupby(g)[g.replace("_id", "_name")].first().map(first_last); cnt = a.groupby(g).size()
            for k_, pid in enumerate(lev[g]):
                players.append(dict(base=base, group=g.split("_")[0], id=pid, name=name[pid], attempts=cnt[pid],
                                    effect=m.u[gi][k_], effect_sd=m.u_sd[gi][k_], pp_vs_avg=pp(b0, m.u[gi][k_])))
        # ground needed by release to clear the break-even rate (average runner, pitcher, catcher and count, vs a RHP)
        B = dict(zip(terms, m.beta[1:]))
        mix = B.get("brk", 0) * a.brk.mean() + B.get("off", 0) * a.off.mean() + B.get("rhb", 0) * a.rhb.mean()
        lo_g, hi_g = a.gain.quantile([0.01, 0.99])
        src = {"2B": {"on_1b": 1, "on_2b": None, "on_3b": None}, "3B": {"on_1b": None, "on_2b": 1, "on_3b": None}}[base]
        sv = steal_values(pd.DataFrame([{**src, "base": base, "outs_when_up": o} for o in range(3)]), BE)
        for o, be in zip(range(3), sv.breakeven):
            ob = {0: 0.0, 1: B.get("out1", 0), 2: B.get("out2", 0)}[o]
            for cat, popc in [("average catcher", 0.0), ("quick catcher (10th pct pop)", a["pop"].quantile(0.1)),
                              ("slow catcher (90th pct pop)", a["pop"].quantile(0.9))]:
                f = lambda g_: expit(b0 + ob + mix + B["pop"] * popc + B["gain"] * g_ + B.get("gain2", 0) * g_ * g_ / 10) - be
                grid = np.linspace(lo_g, hi_g, 400); vals = np.array([f(g_) for g_ in grid])
                cross = np.where((vals[:-1] < 0) & (vals[1:] >= 0))[0]
                need = grid[cross[0]] + a.gain_raw.mean() - a.gain.mean() if len(cross) else (np.nan if vals.max() < 0 else grid[0] + a.gain_raw.mean())
                ground.append(dict(base=base, outs=o, breakeven=be, catcher=cat, ground_needed_ft=need,
                                   share_reaching=float((a.gain_raw >= need).mean()) if need == need else np.nan))
        print(f"{base} success GLMM: n={len(a):,}, sigma {np.round(m.sigma, 3)}, " + ", ".join(f"{t} {B[t]:+.3f}" for t in terms))
    pd.DataFrame(fixed).to_csv(res_path("DF_paper_success_fixed.csv"), index=False)
    pd.DataFrame(var).to_csv(res_path("DF_paper_success_variance.csv"), index=False)
    pd.DataFrame(players).to_csv(res_path("DF_paper_success_players.csv"), index=False)
    G = pd.DataFrame(ground); G.to_csv(res_path("DF_paper_ground.csv"), index=False)
    print(G.round(3).to_string(index=False))


# ── 13b. Paper figures: one journal style ───────────────────────────────────────────────────────────────────────
# Sized for the printed page (7 in wide), 8-9.5 pt type, white background, at most three colours: blue = steals of
# second / stolen bases, orange = caught stealing / below break-even, grey = steals of third / reference lines. Each
# figure makes one point; panel letters; the takeaway lives in the caption, not crammed into the plot.
J_BLUE, J_ORANGE, J_GREY, J_INK, J_MUTED = "#2a78d6", "#eb6834", "#8a8984", "#111111", "#55544f"


def jstyle():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3,
        "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": J_INK,
        "axes.labelcolor": J_INK, "xtick.color": J_INK, "ytick.color": J_INK, "text.color": J_INK,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titlepad": 8})
    import matplotlib.pyplot as plt
    return plt


def jsave(fig, name):
    import matplotlib.pyplot as plt
    fig.savefig(fig_path(name), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig); print(f"wrote {fig_path(name)}")


def letter(ax, s, x=-0.14):
    ax.text(x, 1.02, s, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="left")


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float); ph = k / n
    mid = (ph + z * z / (2 * n)) / (1 + z * z / n); half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


def fig_breakeven():
    """Fig 2: the bar each steal must clear (break-even by out count) against what teams achieved, by base."""
    plt = jstyle()
    BE = pd.read_csv(res_path("DF_paper_breakeven.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, base, lab, title in zip(axs, ["2B", "3B"], "AB", ["Steal of second, runner on first only", "Steal of third, runner on second only"]):
        a = state_rows(base)
        clean = (a.on_2b.isna() & a.on_3b.isna()) if base == "2B" else (a.on_1b.isna() & a.on_3b.isna())
        for o in [0, 1, 2]:
            d = a[clean & (a.outs_when_up == o)]
            k, n = d.y.sum(), len(d); lo, hi = wilson(k, n); r = k / n
            be = BE[(BE.base == base) & (BE.outs == o)].breakeven.iloc[0]
            ax.plot([o - 0.22, o + 0.22], [100 * be] * 2, color=J_INK, lw=2.4, solid_capstyle="butt", zorder=3)
            ax.errorbar(o, 100 * r, yerr=[[100 * (r - lo)], [100 * (hi - r)]], fmt="o", color=J_BLUE, ms=5.5, lw=1.3, zorder=4)
            close = abs(r - be) < 0.035
            ax.text(o + 0.27, 100 * r + (1.6 if close else 0), f"{100 * r:.0f}%", fontsize=7.5, color=J_BLUE, va="center")
            ax.text(o + 0.27, 100 * be - (1.6 if close else 0), f"{100 * be:.0f}%", fontsize=7.5, color=J_INK, va="center")
            ax.text(o, 51.5, f"n = {n:,}", ha="center", fontsize=7, color=J_MUTED)
        ax.set_xticks([0, 1, 2], ["0 outs", "1 out", "2 outs"]); ax.set_xlim(-0.5, 2.65); ax.set_ylim(50, 100)
        ax.set_title(f"{lab}   {title}")
    axs[0].set_ylabel("Success rate (%)")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=J_INK, lw=2.4), Line2D([], [], marker="o", ls="-", color=J_BLUE, lw=1.3)],
               ["break-even success rate (run expectancy, 2023-26)", "observed success rate, 2023-26 (95% CI)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), handlelength=1.8)
    jsave(fig, "Fig_paper_breakeven.png")


def fig_drivers():
    """Fig 3: what decides a steal of second: the success rate across fifths of sprint speed and of ground gained (same
    scale), and how often each speed fifth runs at all."""
    plt = jstyle()
    a = dec_rows("2B", evaluation=True)
    sel = pd.read_csv(res_path("DF_v13_SelectionEffect.csv")).set_index("sprint_speed_all")
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 0.9]))
    swing = {}
    for ax, col, lab, xl, c, k in [(axs[0], "speed_raw", "A", "Sprint speed (ft/s)", J_GREY, "speed"),
                                  (axs[1], "gain_raw", "B", "Ground gained to release (ft)", J_BLUE, "gain")]:
        q = pd.qcut(a[col], 5)
        g = a.groupby(q, observed=True).agg(x=(col, "median"), k=("y", "sum"), n=("y", "size"))
        lo, hi = wilson(g.k, g.n); r = g.k / g.n
        ax.errorbar(g.x, 100 * r, yerr=[100 * (r - lo), 100 * (hi - r)], fmt="o-", color=c, ms=4.5, lw=1.4, capsize=0)
        swing[k] = 100 * (r.iloc[-1] - r.iloc[0])
        ax.set_ylim(55, 100); ax.set_xlabel(xl)
    axs[0].set_ylabel("Success rate (%)")
    axs[0].set_title(f"A   Sprint speed: {swing['speed']:+.0f} pts", fontsize=8.5)
    axs[1].set_title(f"B   Ground gained: {swing['gain']:+.0f} pts", fontsize=8.5)
    ax = axs[2]
    order = ["slowest", "slow", "mid", "fast", "fastest"]
    ax.bar(range(5), sel.loc[order, "attempt_pct"], color=J_GREY, width=0.62)
    for i, v in enumerate(sel.loc[order, "attempt_pct"]):
        ax.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=7.2)
    ax.set_xticks(range(5), ["1", "2", "3", "4", "5"]); ax.set_xlabel("Speed fifth (1 = slowest)")
    ax.set_ylabel("Attempts per 100 chances")
    ax.set_title(f"C   Who runs: {sel.loc['fastest', 'attempt_pct'] / sel.loc['slowest', 'attempt_pct']:.0f}× as often", fontsize=8.5)
    jsave(fig, "Fig_paper_drivers.png")


INFLUENCE_LABEL = {"gain": "Ground gained by release (more)", "pop": "Catcher pop time (slower)",
                   "speed": "Sprint speed (faster)", "lead": "Lead at first move (longer)",
                   "brk": "Breaking ball (vs fastball)", "off": "Offspeed pitch (vs fastball)",
                   "rhb": "Right-handed batter", "lhp": "Left-handed pitcher", "balls": "Balls in the count (more)",
                   "strikes": "Strikes in the count (more)", "out1": "One out (vs none)", "out2": "Two outs (vs none)"}


def fig_influence():
    """Fig 5: what moves the odds: each input's weight in log-odds of a steal succeeding, per standard deviation of the
    input (indicators: 0 to 1), before the pitch (decision model) and given the jump and the pitch (success model), by
    base. Ground gained: its slope at the average (the squared term makes it steeper above average). Writes
    DF_paper_influence.csv."""
    plt = jstyle()
    D = pd.read_csv(res_path("DF_paper_decision_fixed.csv")); D = D[D.model == "decision"]
    S = pd.read_csv(res_path("DF_paper_success_fixed.csv"))
    rows = []
    for model, F, get in [("decision", D, dec_rows), ("success", S, success_rows)]:
        for base in ("2B", "3B"):
            a = get(base)
            for r in F[(F.base == base) & ~F.term.isin(["intercept", "gain2"])].itertuples():
                sd = a[r.term + "_raw"].std() if r.term + "_raw" in a else 1.0          # indicators: 0 to 1
                rows.append(dict(model=model, base=base, term=r.term, label=INFLUENCE_LABEL[r.term], sd=sd, beta=r.beta,
                                 se=r.se, per_sd=r.beta * sd, lo=(r.beta - 1.96 * r.se) * sd, hi=(r.beta + 1.96 * r.se) * sd))
    W = pd.DataFrame(rows); W.to_csv(res_path("DF_paper_influence.csv"), index=False)
    order = W[(W.model == "success") & (W.base == "2B")].assign(k=lambda d: d.per_sd.abs()).sort_values("k", ascending=False).term.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.06))
    for ax, model, title in [(axs[0], "decision", "A   Before the pitch (decision model)"),
                             (axs[1], "success", "B   Given the jump and the pitch (success model)")]:
        for base, col, dy, mfc in [("2B", J_BLUE, -0.17, J_BLUE), ("3B", J_GREY, 0.17, "white")]:
            d = W[(W.model == model) & (W.base == base)].set_index("term")
            for k, t in enumerate(order):
                if t not in d.index:
                    continue
                r = d.loc[t]
                ax.plot([r.lo, r.hi], [k + dy] * 2, color=col, lw=1.3, solid_capstyle="butt")
                ax.plot(r.per_sd, k + dy, "o", color=col, mfc=mfc, mew=1.1, ms=4.6,
                        label="steal of second" if base == "2B" else "steal of third")
        for k, t in enumerate(order):
            if t not in W[W.model == model].term.values:
                ax.text(0.03, k, "measured only during the pitch", fontsize=6.8, color=J_MUTED, va="center", style="italic")
        ax.axvline(0, color=J_INK, lw=0.7); ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", color="#ececec", lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("Log-odds of success per SD (indicators: 0 to 1)")
    axs[0].set_yticks(range(len(order)), [INFLUENCE_LABEL[t] for t in order]); axs[0].invert_yaxis()
    h, l = axs[1].get_legend_handles_labels(); keep = dict(zip(l, h))                # one entry per base
    fig.legend(list(keep.values()), list(keep.keys()), loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.1), fontsize=7.4)
    jsave(fig, "Fig_paper_influence.png")


def fig_speed_2b(bin_w: float = 0.5):
    """Fig 5: steals of second by sprint speed, season by season: how often each speed ran and was caught (A), and the
    success rate with the speed-only logistic fit and the break-even line (B)."""
    plt = jstyle()
    a = dec_rows("2B"); be = steal_values(a, re24(range(2023, 2027))).breakeven.mean()
    edges = np.arange(np.floor(a.speed_raw.min() / bin_w) * bin_w, a.speed_raw.max() + bin_w + 1e-9, bin_w)
    seasons, rows = sorted(a.season.unique()), []
    fig, axs = plt.subplots(len(seasons), 2, figsize=(7.0, 1.9 * len(seasons) + 0.3), sharex=True,
                            gridspec_kw=dict(hspace=0.45, wspace=0.24))
    for i, s_ in enumerate(seasons):
        d = a[a.season == s_]
        c = d.groupby(pd.cut(d.speed_raw, edges, right=False), observed=True).y.agg(["sum", "size"])
        x = np.array([iv.mid for iv in c.index]); sb, n = c["sum"].values, c["size"].values; cs = n - sb
        ax = axs[i, 0]                                            # stacked: the bar's height is the attempts
        ax.bar(x, sb, width=bin_w * 0.86, color=J_BLUE, label="stolen base")
        ax.bar(x, cs, bottom=sb, width=bin_w * 0.86, color=J_ORANGE, label="caught stealing")
        ax.set_ylabel("Attempts"); ax.tick_params(labelbottom=True)
        ax.text(0.98, 0.9, f"n = {len(d):,}", transform=ax.transAxes, ha="right", fontsize=7.2, color=J_MUTED)
        ax.text(-0.3, 0.5, str(s_), transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=9.5, fontweight="bold")
        ax = axs[i, 1]
        f = sm.Logit(d.y.values, sm.add_constant(d.speed_raw.values)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.runner_id.values})
        grid = np.linspace(*d.speed_raw.quantile([0.01, 0.99]), 100)
        G = sm.add_constant(grid); eta = G @ f.params; se = np.sqrt(np.einsum("ij,jk,ik->i", G, f.cov_params(), G))
        ax.fill_between(grid, 100 * expit(eta - 1.96 * se), 100 * expit(eta + 1.96 * se), color=J_BLUE, alpha=0.15, lw=0)
        ax.plot(grid, 100 * expit(eta), color=J_BLUE, lw=1.3, label="logistic regression on speed alone (95% CI)")
        lo, hi = wilson(sb, n); big = n >= 20
        ax.errorbar(x[big], 100 * sb[big] / n[big], yerr=[100 * (sb[big] / n[big] - lo[big]), 100 * (hi[big] - sb[big] / n[big])],
                    fmt="o", color=J_INK, ms=3.2, lw=0.8, capsize=0, label="observed, 0.5 ft/s bins (95% CI)")
        ax.plot(x[~big], 100 * sb[~big] / n[~big], "o", mfc="white", color=J_GREY, ms=3.2, label="bins with fewer than 20 attempts")
        ax.axhline(100 * be, color=J_GREY, lw=0.9, ls=(0, (4, 3)), label=f"break-even ({100 * be:.0f}%)")
        q10, q90 = d.speed_raw.quantile([0.1, 0.9])
        sw = 100 * (expit(f.params[0] + f.params[1] * q90) - expit(f.params[0] + f.params[1] * q10)); pv = f.pvalues[1]
        r_ = d.groupby("runner_id").agg(n=("y", "size"), safe=("y", "mean"), speed=("speed_raw", "mean")).query("n >= 10")
        rows.append(dict(season=s_, attempts=len(d), success=d.y.mean(), beta=f.params[1], se=f.bse[1], p=pv, speed_p10=q10,
                         speed_p90=q90, swing_pts=sw, auroc=roc_auc_score(d.y, d.speed_raw), runners_10=len(r_),
                         r_runner=np.corrcoef(r_.speed, r_.safe)[0, 1]))
        ax.text(0.03, 0.07, f"10th to 90th percentile: {sw:+.1f} pts (p {'< 0.001' if pv < 0.001 else f'= {pv:.3f}'})",
                transform=ax.transAxes, fontsize=7.2, color=J_MUTED, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_ylim(40, 104); ax.set_yticks([40, 60, 80, 100]); ax.set_ylabel("Success rate (%)"); ax.tick_params(labelbottom=True)
    for ax in axs[-1]:
        ax.set_xlabel("Runner sprint speed (ft/s)")
    axs[0, 0].set_title("A   Attempts by sprint speed"); axs[0, 1].set_title("B   Success rate by sprint speed")
    axs[0, 0].legend(loc="upper left", handlelength=1.2)
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.0), handlelength=1.6)
    fig.subplots_adjust(bottom=0.1)
    pd.DataFrame(rows).to_csv(res_path("DF_paper_speed_2b.csv"), index=False)
    jsave(fig, "Fig_paper_speed.png")


def fig_variance():
    """Fig 6: who controls the running game: a player one SD above average, in points of P(safe), by base, before the pitch
    (decision model: total influence) and given the jump and the pitch (success model: what is left)."""
    plt = jstyle()
    D_ = pd.read_csv(res_path("DF_paper_decision_variance.csv")); D_ = D_[D_.model == "decision"]
    S_ = pd.read_csv(res_path("DF_paper_success_variance.csv"))
    groups = ["runner", "pitcher", "catcher"]
    xmax = max(8.0, np.ceil(max(D_.pp_plus_hi.max(), S_.pp_plus_hi.max()) + 0.5))    # never clip an interval
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.08))
    for ax, V, title in [(axs[0], D_, "A   Before the pitch (decision model)"), (axs[1], S_, "B   Given the jump and the pitch (success model)")]:
        for base, col, dy, mfc in [("2B", J_BLUE, -0.15, J_BLUE), ("3B", J_GREY, 0.15, "white")]:
            d = V[V.base == base].set_index("group")
            for i, g in enumerate(groups):
                r = d.loc[g]
                ax.plot([r.pp_plus_lo, r.pp_plus_hi], [i + dy] * 2, color=col, lw=1.4, solid_capstyle="butt")
                ax.plot(r.pp_plus_1sd, i + dy, "o", color=col, mfc=mfc, mew=1.2, ms=5.3,
                        label=("steal of second" if base == "2B" else "steal of third") if i == 0 else None)
        ax.axvline(0, color=J_INK, lw=0.7); ax.set_xlim(-0.3, xmax); ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", color="#ececec", lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("Points of P(safe) per SD (95% CI)")
    axs[0].set_yticks(range(3), [f"{g.capitalize()}s" for g in groups]); axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.17), fontsize=7.4)
    jsave(fig, "Fig_paper_variance.png")


def fig_value():
    """Fig 8: the calls, scored on 2026: runs per attempt by how far the model's P(safe) cleared the break-even rate."""
    plt = jstyle()
    V = pd.read_csv(res_path("DF_paper_decision_value.csv"))
    order = ["< -5", "-5 to 0", "0 to 5", "5 to 10", "> 10"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, base, lab in zip(axs, ["2B", "3B"], "AB"):
        d = V[(V.base == base) & V.margin_pts.isin(order)].set_index("margin_pts").reindex(order).dropna(subset=["n"])
        x = np.arange(len(d))
        ax.bar(x, d.runs_per_attempt, color=[J_ORANGE if m in ("< -5", "-5 to 0") else J_BLUE for m in d.index], width=0.62)
        ax.errorbar(x, d.runs_per_attempt, yerr=1.96 * d.runs_se, fmt="none", ecolor=J_INK, lw=0.9)
        for xi, (_, r) in zip(x, d.iterrows()):
            ax.text(xi, -0.235, f"n = {int(r.n):,}", ha="center", fontsize=6.8, color=J_MUTED)
        ax.axhline(0, color=J_INK, lw=0.7); ax.axvline(1.5, color=J_GREY, lw=0.8, ls=(0, (4, 3)))
        ax.text(0.5, 0.185, "model says hold", ha="center", fontsize=7.5, color=J_ORANGE)
        ax.text(3.0, 0.185, "model says go", ha="center", fontsize=7.5, color=J_BLUE)
        ax.set_xticks(x, ["below –5", "–5 to 0", "0 to 5", "5 to 10", "above 10"])
        ax.set_xlabel("Model's probability minus break-even (points)")
        ax.set_title(f"{lab}   " + ("Steals of second, 2026" if base == "2B" else "Steals of third, 2026"))
        ax.set_ylim(-0.25, 0.21)
    axs[0].set_ylabel("Runs gained per attempt (95% CI)")
    jsave(fig, "Fig_paper_decision_value.png")


def fig_greenlight():
    """Fig 9: who to send in 2027: each recent base stealer's predicted success on a steal of second against an
    average battery with one out, against sprint speed, with the break-even line."""
    plt = jstyle()
    import matplotlib.patheffects as pe
    G = pd.read_csv(res_path("DF_paper_greenlight.csv")); G = G[G.base == "2B"].copy()
    be = G.be_1out.iloc[0]; go = G.p_1out >= be
    size = lambda n: 8 + 1.6 * n
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.scatter(G.speed[go], 100 * G.p_1out[go], s=size(G.attempts_2025_26[go]), color=J_BLUE, alpha=0.55, lw=0,
               label=f"clears break-even ({go.sum()} runners)")
    ax.scatter(G.speed[~go], 100 * G.p_1out[~go], s=size(G.attempts_2025_26[~go]), color=J_ORANGE, alpha=0.9, lw=0,
               label=f"below break-even ({(~go).sum()} runners)")
    ax.axhline(100 * be, color=J_GREY, lw=0.9, ls=(0, (4, 3)))
    ax.text(24.15, 100 * be + 0.35, f"break-even with one out: {100 * be:.0f}%", fontsize=7.2, color=J_MUTED)
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    top, bot = G.nlargest(3, "p_1out"), G.nsmallest(3, "p_1out")
    for (_, r), off in zip(top.iterrows(), [(-6, 6), (6, 2), (6, -8)]):
        ax.annotate(r["name"], (r.speed, 100 * r.p_1out), xytext=off, textcoords="offset points", fontsize=7,
                    ha="right" if off[0] < 0 else "left", path_effects=halo)
    for (_, r), off in zip(bot.iterrows(), [(-6, -2), (6, -8), (6, 4)]):
        ax.annotate(r["name"], (r.speed, 100 * r.p_1out), xytext=off, textcoords="offset points", fontsize=7,
                    ha="right" if off[0] < 0 else "left", path_effects=halo)
    for n_ in (10, 30, 60):
        ax.scatter([], [], s=size(n_), color=J_GREY, alpha=0.5, lw=0, label=f"{n_} attempts")
    ax.set_xlabel("Sprint speed, latest season (ft/s)"); ax.set_ylabel("Predicted success, steal of second (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=7.2, columnspacing=1.1, handletextpad=0.3)
    jsave(fig, "Fig_paper_greenlight.png")


def fig_prediction():
    """Fig 3: how well a steal can be predicted, 2026, every model fit on 2023-25: ROC curves (A) and calibration (B)."""
    plt = jstyle()
    F = pd.read_csv(res_path("DF_paper_forward_all.csv")).set_index("model")
    R, C = pd.read_csv(res_path("DF_paper_roc.csv")), pd.read_csv(res_path("DF_paper_calibration_all.csv"))
    succ = next(m for m in F.index if m.startswith("Success GLMM"))
    v16 = "v16 logistic (pooled, tuned on 2023-25)"
    lines = [(succ, J_BLUE, "-", 1.9, "success model"), (v16, J_INK, "-", 1.0, "v16 logistic"),
             ("Pitcher's Dilemma specification (lead, speed, jump, pop), refit", J_GREY, (0, (4, 2)), 1.1, "Pitcher's Dilemma spec."),
             ("Decision GLMM: before the pitch only", J_ORANGE, "-", 1.4, "before the pitch only"),
             ("Sprint speed only", J_GREY, ":", 1.3, "sprint speed only")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.35), gridspec_kw=dict(wspace=0.34))
    ax = axs[0]; ax.plot([0, 1], [0, 1], color="#d0d0d0", lw=0.8)
    for m, col, ls, lw, lab in lines:
        r = R[R.model == m]
        ax.plot(r.fpr, r.tpr, color=col, ls=ls, lw=lw, label=f"{lab}  {F.loc[m, 'auroc']:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
    ax.set_title("A   ROC curves, 2026 attempts"); ax.legend(loc="lower right", fontsize=6.7, title="model  AUROC", title_fontsize=6.7,
                                                          handlelength=1.8)
    ax = axs[1]; ax.plot([0.3, 1], [0.3, 1], color=J_GREY, lw=0.8, ls=(0, (4, 3)))
    for m, col, mk, mfc, lab in [(succ, J_BLUE, "o", J_BLUE, "success model"), (v16, J_INK, "s", "white", "v16 logistic")]:
        d = C[C.model == m]; k = d.observed * d.n; lo, hi = wilson(k, d.n)
        ax.errorbar(d.mean_pred, d.observed, yerr=[d.observed - lo, hi - d.observed], fmt=mk, color=col, mfc=mfc, ms=4.3, lw=0.9,
                    mew=1.1, label=f"{lab} (slope {F.loc[m, 'cal_slope']:.2f})")
    ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability of success"); ax.set_ylabel("Observed success rate")
    ax.set_title("B   Calibration, 2026 attempts"); ax.legend(loc="lower right", fontsize=7, handlelength=1.0)
    jsave(fig, "Fig_paper_prediction.png")


def fig_ground():
    """Fig 4: what it takes to be safe: the success model's probability against the ground gained by release (average
    runner, pitcher and count, one out) for an average, a quick and a slow catcher, with the break-even rate; by base."""
    plt = jstyle()
    Gd, Fx = pd.read_csv(res_path("DF_paper_ground.csv")), pd.read_csv(res_path("DF_paper_success_fixed.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, base, lab, title in zip(axs, ["2B", "3B"], "AB", ["Steal of second, one out", "Steal of third, one out"]):
        a = success_rows(base); B = Fx[Fx.base == base].set_index("term").beta
        mix = B.get("brk", 0) * a.brk.mean() + B.get("off", 0) * a.off.mean() + B.get("rhb", 0) * a.rhb.mean()
        off = (a.gain_raw - a.gain).iloc[0]                        # raw = centred + this base's mean
        gr = np.linspace(*a.gain_raw.quantile([0.05, 0.95]), 200); gc = gr - off     # where the data are dense
        for cat, popc, col, lw, ls in [("quick catcher (10th pct pop time)", a["pop"].quantile(0.1), J_GREY, 1.0, (0, (3, 2))),
                                       ("average catcher", 0.0, J_BLUE, 1.8, "-"),
                                       ("slow catcher (90th pct pop time)", a["pop"].quantile(0.9), J_GREY, 1.0, ":")]:
            eta = B["intercept"] + B.get("out1", 0) + mix + B["pop"] * popc + B["gain"] * gc + B.get("gain2", 0) * gc * gc / 10
            ax.plot(gr, 100 * expit(eta), color=col, lw=lw, ls=ls, label=cat)
        r = Gd[(Gd.base == base) & (Gd.outs == 1) & (Gd.catcher == "average catcher")].iloc[0]
        ax.axhline(100 * r.breakeven, color=J_INK, lw=0.9, ls=(0, (4, 3)))
        ax.text(gr[-1], 100 * r.breakeven - 4.0, f"break-even {100 * r.breakeven:.0f}%", fontsize=7.2, color=J_INK, ha="right")
        if r.ground_needed_ft == r.ground_needed_ft:
            ax.plot([r.ground_needed_ft] * 2, [40, 100 * r.breakeven], color=J_BLUE, lw=0.8)
            ax.plot(r.ground_needed_ft, 100 * r.breakeven, "o", color=J_BLUE, ms=5)
            ax.text(r.ground_needed_ft + 0.25, 42.5, f"needs {r.ground_needed_ft:.1f} ft", fontsize=7.4, color=J_BLUE)
        ax.set_ylim(40, 100); ax.set_xlabel("Ground gained, first move to release (ft)"); ax.set_title(f"{lab}   {title}")
    axs[0].set_ylabel("Probability of success (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), handlelength=2.2, fontsize=7.4)
    jsave(fig, "Fig_paper_ground.png")


def fig_anatomy():
    """Fig 1: the race a steal of second runs, drawn at the median of every input (steals of second, 2023-26). Pitch
    flight is estimated from release speed and extension (about 4% speed lost to drag); the rest is measured."""
    plt = jstyle()
    a = dec_rows("2B", evaluation=True)
    m = load_meta(); at = m[m.result.isin(["SB", "CS"]) & (m.base == "2B")]
    dl = at.loc[at.cv_qa == "PASS", "cv_delivery_s"].median()
    f_ = at.dropna(subset=["release_speed", "release_extension"])
    fl = ((60.5 - f_.release_extension) / (0.96 * f_.release_speed * 5280 / 3600)).median()   # ponytail: flat 4% drag
    lead, gain, spd, pop = a.lead_raw.median(), a.gain_raw.median(), a.speed_raw.median(), a.pop_raw.median() / 10
    end = dl + fl + pop
    pd.DataFrame([dict(lead_ft=lead, gain_ft=gain, sprint_ft_s=spd, pop_s=pop, delivery_s=dl, flight_s=fl, defense_s=end,
                       left_to_run_ft=90 - lead - gain)]).to_csv(res_path("DF_paper_anatomy.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.axvspan(-0.75, 0, color="#f3f3f1", lw=0); ax.axvspan(0, dl, color="#e8f0fb", lw=0)
    ax.text(-0.375, 1.62, "known before\nthe pitch", fontsize=6.8, color=J_MUTED, va="top", ha="center")
    ax.text(dl / 2, 1.62, "measured during\nthe delivery", fontsize=6.8, color=J_BLUE, va="top", ha="center")
    segs = [(0, dl, J_GREY, f"Pitcher's delivery {dl:.2f} s", "white"), (dl, fl, "#d4d3cf", "", J_INK),
            (dl + fl, pop, J_ORANGE, f"Catcher's pop time {pop:.2f} s", "white")]
    for x0, w, c, lab, tc in segs:
        ax.barh(1, w, left=x0, height=0.42, color=c, edgecolor="white", lw=1.2)
        if lab:
            ax.text(x0 + w / 2, 1, lab, ha="center", va="center", fontsize=7.2, color=tc)
    ax.text(dl + fl / 2, 1.3, f"pitch\n{fl:.1f} s", ha="center", va="bottom", fontsize=6.6, color=J_MUTED)
    ax.barh(0, dl, left=0, height=0.42, color=J_BLUE, edgecolor="white", lw=1.2)
    ax.text(dl / 2, 0, f"gains {gain:.1f} ft", ha="center", va="center", fontsize=7.2, color="white")
    ax.annotate("", xy=(end + 0.35, 0), xytext=(dl, 0), arrowprops=dict(arrowstyle="-|>", color="#a9c9ef", lw=9, shrinkA=0, shrinkB=0))
    ax.text((dl + end) / 2, 0, f"sprints the last {90 - lead - gain:.0f} ft (sprint speed {spd:.1f} ft/s)", ha="center", va="center",
            fontsize=7.2, color=J_INK)
    ax.plot(0, 0, "o", color=J_INK, ms=4); ax.text(-0.07, 0, f"{lead:.1f} ft off\nfirst base", ha="right", va="center", fontsize=7, color=J_INK)
    for t, lab in [(0, "pitcher's\nfirst move"), (dl, "release"), (dl + fl, "pitch\ncaught"), (end, "throw\nat second")]:
        ax.axvline(t, color=J_INK, lw=0.6, ls=(0, (2, 2)), ymin=0.2, ymax=0.83)
        ax.text(t, -0.62, lab, ha="center", va="top", fontsize=6.8, color=J_INK)
    ax.set_yticks([0, 1], ["Runner", "Pitcher and\ncatcher"]); ax.set_ylim(-1.05, 1.7); ax.set_xlim(-0.75, end + 0.45)
    ax.set_xlabel("Seconds from the pitcher's first move"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    jsave(fig, "Fig_paper_anatomy.png")


SPEED_EDGES = [-np.inf] + [25.5 + 0.5 * k for k in range(10)] + [np.inf]     # 0.5 ft/s bins; sparse tails merged


def speed_bins():
    """Steals of second by sprint speed (the runner's season sprint speed, 0.5 ft/s bins, tails merged below 25.5 and
    from 30.0): stolen bases and caught stealing by season, success, runner-seasons and attempts per runner-season.
    Writes DF_paper_speed_bins.csv (the table behind Figures 7 and 8)."""
    a = dec_rows("2B")
    lab = ["below 25.5"] + [f"{lo:.1f}–{lo + 0.4:.1f}" for lo in SPEED_EDGES[1:-2]] + ["30.0 and up"]
    a["bin"] = pd.cut(a.speed_raw, SPEED_EDGES, right=False, labels=lab)
    out = a.groupby("bin", observed=True).agg(attempts=("y", "size"), sb=("y", "sum"))
    for s_ in sorted(a.season.unique()):
        g = a[a.season == s_].groupby("bin", observed=True).y.agg(["sum", "size"])
        out[f"sb_{s_}"], out[f"cs_{s_}"] = g["sum"], g["size"] - g["sum"]
    rs = a.groupby(["runner_id", "season"]).agg(bin=("bin", "first"), n=("y", "size"))
    out["cs"] = out.attempts - out.sb; out["success"] = out.sb / out.attempts
    out["runner_seasons"] = rs.groupby("bin", observed=True).size()
    out["attempts_per_runner_season"] = out.attempts / out.runner_seasons
    out = out.fillna(0).reset_index()
    out.to_csv(res_path("DF_paper_speed_bins.csv"), index=False)
    return out


def fig_runners():
    """Fig 8: runner by runner: each runner-season's success rate on steals of second (10+ attempts) against his sprint
    speed, marker area by attempts, with the speed-only logistic fit on every attempt. Writes DF_paper_runner_speed.csv:
    the correlation across runner-seasons (with and without Josh Naylor's two seasons) and the same after removing
    binomial sampling noise from the season success rates (Spearman's correction for attenuation)."""
    plt = jstyle()
    speed_bins()
    a = dec_rows("2B"); be = a.breakeven.mean()
    rs = a.groupby(["runner_id", "season"]).agg(name=("runner_name", "first"), speed=("speed_raw", "first"),
                                                n=("y", "size"), safe=("y", "mean")).reset_index()
    q = rs[rs.n >= 10]
    r, r_wo = np.corrcoef(q.speed, q.safe)[0, 1], np.corrcoef(*q[q.name != "Josh Naylor"][["speed", "safe"]].values.T)[0, 1]
    noise = (q.safe.mean() * (1 - q.safe.mean()) / q.n).mean(); rel = 1 - noise / q.safe.var()    # signal share of the spread
    pd.DataFrame([dict(runner_seasons=len(q), r=r, r_without_naylor=r_wo, observed_sd=q.safe.std(), noise_var=noise,
                       reliability=rel, true_sd=np.sqrt(q.safe.var() - noise), r_true=r / np.sqrt(rel),
                       r_true_without_naylor=r_wo / np.sqrt(rel))]).to_csv(res_path("DF_paper_runner_speed.csv"), index=False)
    f = sm.Logit(a.y.values, sm.add_constant(a.speed_raw.values)).fit(disp=0)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.scatter(q.speed, 100 * q.safe, s=4 + 1.6 * q.n, color=J_BLUE, alpha=0.45, lw=0)
    gx = np.linspace(q.speed.min(), q.speed.max(), 100)
    ax.plot(gx, 100 * expit(f.params[0] + f.params[1] * gx), color=J_INK, lw=1.0, label="logistic regression on speed alone, every attempt")
    ax.axhline(100 * be, color=J_GREY, lw=0.9, ls=(0, (4, 3)), label=f"break-even ({100 * be:.0f}%)")
    nay = q[q.name == "Josh Naylor"].sort_values("season")
    if len(nay):                                                  # the slowest regular base stealer, beside his dots
        txt = "Josh Naylor\n" + "\n".join(f"{t.season}: {int(round(t.safe * t.n))} of {t.n}" for t in nay.itertuples())
        ax.text(nay.speed.max() + 0.14, 100 * nay.safe.max() + 1.2, txt, fontsize=7.2, color=J_INK, va="top")
    ax.text(0.99, 0.04, f"r = {r:+.2f} across {len(q):,} runner-seasons with 10+ attempts ({r_wo:+.2f} without Naylor)",
            transform=ax.transAxes, ha="right", fontsize=7.2, color=J_MUTED)
    for nn in (10, 30, 60):
        ax.scatter([], [], s=4 + 1.6 * nn, color=J_BLUE, alpha=0.45, lw=0, label=f"{nn} attempts")
    ax.set_xlabel("Runner's sprint speed that season (ft/s)"); ax.set_ylabel("Success rate on steals of second (%)")
    ax.set_ylim(35, 104)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.53, -0.1), fontsize=7, handlelength=1.6)
    jsave(fig, "Fig_paper_runners.png")


def pop_choice():
    """Steals of third: the decision model with the catcher's pop time to third against pop time to second, on the
    attempts that have both (same rows, same number of parameters; lower deviance fits better). Writes
    DF_paper_pop_choice.csv."""
    a = dec_rows("3B"); a = a[a.catcher_pop_2b.notna()].copy()
    a["pop2"] = 10 * a.catcher_pop_2b; a["pop2"] -= a.pop2.mean()
    rows = []
    for name, term in [("pop time to third", "pop"), ("pop time to second", "pop2")]:
        terms = [term if t == "pop" else t for t in DEC_FIXED]
        X, codes, _ = design(a, terms); m = GLMM(X, codes, a.y.values).fit(); j = terms.index(term) + 1
        rows.append(dict(input=name, n=len(a), deviance=m.dev, beta=m.beta[j], se=m.se[j]))
    out = pd.DataFrame(rows); out.to_csv(res_path("DF_paper_pop_choice.csv"), index=False); print(out.round(3).to_string(index=False))
    return out


def run_paperfigs():
    for f in (fig_anatomy, fig_breakeven, fig_drivers, fig_prediction, fig_influence, fig_ground, fig_speed_2b, fig_runners,
              fig_variance, fig_value, fig_greenlight):
        f()


