#!/usr/bin/env python3
"""
build_paper.py — renders the paper (StealIQ_Paper.pdf) and its figure-by-figure discussion (StealIQ_Figure_Discussion.pdf)
from the files paper.py and ingest.py wrote, then refreshes the catalog (build_catalog.py). Every number in the two PDFs
is read from results/ or recomputed here from the same rows. The earlier technical report (27 pages, model ladder and
public-model comparisons) is archived at 4-Archive/reports/StealIQ_Technical_Report_2026-10-06.pdf.

Usage:  python3 build_paper.py       the two PDFs, then CATALOG.md, 0-Catalog.pdf and the folder READMEs
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stealiq as IQ
from stealiq import res_path, fig_path

REPO = IQ.ROOT.parent
PAPER, DISCUSSION = IQ.ROOT / "StealIQ_Paper.pdf", IQ.ROOT / "StealIQ_Figure_Discussion.pdf"
INK, MUTED, RULE = colors.HexColor("#111111"), colors.HexColor("#55544f"), colors.HexColor("#111111")

# ── type: serif body, sans headings, captions and tables (journal style) ──────────────────────────────────────────
TITLE = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=INK, alignment=TA_CENTER, spaceAfter=4)
SUB = ParagraphStyle("Su", fontName="Helvetica", fontSize=11, leading=14, textColor=MUTED, alignment=TA_CENTER, spaceAfter=14)
H1 = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=INK, spaceBefore=12, spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1)
P = ParagraphStyle("P", fontName="Times-Roman", fontSize=10.3, leading=13.8, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
ABS = ParagraphStyle("A", parent=P, fontSize=9.8, leading=13, leftIndent=24, rightIndent=24)
CAP = ParagraphStyle("C", fontName="Helvetica", fontSize=8.2, leading=10.6, textColor=INK, spaceBefore=3, spaceAfter=12)
CELL = ParagraphStyle("Ce", fontName="Helvetica", fontSize=7.8, leading=9.6, textColor=INK)
CELLB = ParagraphStyle("Cb", parent=CELL, fontName="Helvetica-Bold")
NOTE = ParagraphStyle("N", fontName="Helvetica", fontSize=7.4, leading=9.4, textColor=MUTED, spaceAfter=10)
BUL = ParagraphStyle("B", parent=P, leftIndent=13, bulletIndent=3, spaceAfter=3)
EQ = ParagraphStyle("E", parent=P, alignment=TA_CENTER, fontName="Times-Italic", spaceBefore=2, spaceAfter=8)
REF = ParagraphStyle("R", parent=P, fontSize=9.2, leading=11.8, leftIndent=14, firstLineIndent=-14, alignment=0)
WIDTH = 6.5   # text width (in): letter with 1-inch margins


def para(t, s=P): return Paragraph(t, s)
def bullets(items, s=BUL): return [Paragraph(t, s, bulletText="•") for t in items]
def pct(x, d=0): return f"{100 * x:.{d}f}%"
def pts(x, d=1, sign=True): return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def figure(name, n, caption, width=WIDTH):
    """A figure at the text width with its numbered caption, kept on one page."""
    path = fig_path(name); w, h = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width * inch, height=width * inch * h / w),
                         para(f"<b>Figure {n}.</b> {caption}", CAP)])


def table(rows, widths, caption=None, note=None, bold_last=False, groups=(), keep=True):
    """Booktabs-style table: rules above and below the header and at the bottom, no shading. `groups`: row indices of
    group labels, set bold across the full width with a thin rule above, never left alone at the foot of a page.
    keep=False returns the flowables unwrapped, so a table longer than a page breaks across pages instead of jumping to the next."""
    data = [[c if not isinstance(c, str) else Paragraph(c, CELLB if i == 0 or i in groups or (bold_last and i == len(rows) - 1) else CELL)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.9, RULE), ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE),
                           ("LINEBELOW", (0, -1), (-1, -1), 0.9, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
                          + [("SPAN", (0, g), (-1, g)) for g in groups] + [("LINEABOVE", (0, g), (-1, g), 0.4, RULE) for g in groups if g > 1]
                          + [("NOSPLIT", (0, g), (-1, min(g + 1, len(rows) - 1))) for g in groups]))
    out = ([para(f"<b>Table {caption[0]}.</b> {caption[1]}", CAP)] if caption else []) + [t]
    if note:
        out.append(para(note, NOTE))
    else:
        out.append(Spacer(1, 10))
    return KeepTogether(out) if keep else out


# ── everything the paper quotes, loaded once ─────────────────────────────────────────────────────────────────────
class Data:
    def __init__(self):
        r = lambda n: pd.read_csv(res_path(n))
        self.BE, self.FX, self.VC = r("DF_paper_breakeven.csv"), r("DF_paper_decision_fixed.csv"), r("DF_paper_decision_variance.csv")
        self.PL, self.FW, self.VA = r("DF_paper_decision_players.csv"), r("DF_paper_decision_forward.csv"), r("DF_paper_decision_value.csv")
        self.GL, self.CK, self.SP = r("DF_paper_greenlight.csv"), r("DF_paper_glmm_check.csv"), r("DF_paper_speed_2b.csv")
        self.FA = r("DF_paper_forward_all.csv").set_index("model")
        self.SF, self.SV = r("DF_paper_success_fixed.csv"), r("DF_paper_success_variance.csv")
        self.GN, self.SCV = r("DF_paper_ground.csv"), r("DF_paper_success_cv.csv")
        self.IN, self.SB = r("DF_paper_influence.csv"), r("DF_paper_speed_bins.csv")
        self.RSP, self.AN = r("DF_paper_runner_speed.csv").iloc[0], r("DF_paper_anatomy.csv").iloc[0]
        self.PC = r("DF_paper_pop_choice.csv").set_index("input")
        self.succ = next(m for m in self.FA.index if m.startswith("Success GLMM"))
        self.v16, self.v16d = "v16 logistic (pooled, tuned on 2023-25)", "v16 logistic (pooled, default)"
        self.dec = "Decision GLMM: before the pitch only"
        self.SE = r("DF_cmp_seasons.csv").set_index("season")
        lg = pd.read_csv(IQ.RAW / "league_sb_rates.csv").set_index("season")
        self.games_played = int(lg.loc[2023:2026, "games"].sum())
        self.JT, self.sel = r("DF_add_jump_tests.csv"), r("DF_v13_SelectionEffect.csv").set_index("sprint_speed_all")
        self.V16CV = r("DF_v16_models.csv").set_index("model")
        self.RE = IQ.re24_counts()
        m = IQ.load_meta(); self.att = m[m.result.isin(["SB", "CS"])]
        self.rows = {b: IQ.dec_rows(b) for b in ["2B", "3B"]}
        self.srows = {b: IQ.success_rows(b) for b in ["2B", "3B"]}
        self.ev2 = IQ.dec_rows("2B", evaluation=True)
        self.gold = pd.read_csv(IQ.DELIVERY / "delivery_gold.csv")
        ok = self.gold[self.gold.qa == "PASS"]
        self.cv_mae = (((ok.release_frame - ok.lift_frame) - (ok.gold_release - ok.gold_lift)) / ok.fps).abs().mean()
        self.cv = {s: pd.read_csv(f) for s, f in ((int(f.stem[-4:]), f) for f in sorted(IQ.DELIVERY.glob("delivery_20[0-9][0-9].csv")))}
        o = pd.read_csv(IQ.RAW / "Raw_Opportunities.csv.gz", usecols=["is_lhp", "attempt"])
        self.att_rate_hand = o.groupby("is_lhp").attempt.mean() * 100
        pe = self.PL[(self.PL.base == "2B") & (self.PL.model == "decision") & (self.PL.group == "pitcher")].set_index("id").effect
        ct = pd.concat(d[d.qa == "PASS"] for d in self.cv.values()).groupby("pitcher_id").delivery_s.agg(["mean", "size"])
        ct = ct[ct["size"] >= 10].join(pe, how="inner")
        self.pitcher_time_r = (np.corrcoef(ct.effect, ct["mean"])[0, 1], len(ct))
        self.clean = {}                                  # observed success by state: every tracked attempt in it
        for b in ("2B", "3B"):
            a = IQ.state_rows(b)
            c = (a.on_2b.isna() & a.on_3b.isna()) if b == "2B" else (a.on_1b.isna() & a.on_3b.isna())
            for o_ in [0, 1, 2]:
                d = a[c & (a.outs_when_up == o_)]
                self.clean[(b, o_)] = (d.y.mean(), len(d))
        e = self.ev2
        self.fifths = {k: e.groupby(pd.qcut(e[c], 5), observed=True).y.mean() for k, c in [("speed", "speed_raw"), ("gain", "gain_raw")]}
        self.naylor = self._naylor()

    def _naylor(self):
        """The player the introduction opens with: official 2025 totals, sprint-speed rank, and his tracked steals of
        second against every runner-season with ten or more."""
        rs = pd.read_csv(IQ.RAW / "Raw_Season.csv"); s = rs[(rs.player_name == "Josh Naylor") & (rs.season == 2025)].iloc[0]
        sp = pd.read_csv(IQ.RAW / "sprint_speed.csv"); s25 = sp[sp.season == 2025].sprint_speed_all
        t = self.att[(self.att.runner_name == "Josh Naylor") & (self.att.season == 2025)]
        g = self.ev2.groupby(["runner_id", "season"]).agg(name=("runner_name", "first"), n=("y", "size"), safe=("y", "sum"),
                                                          lead=("lead_raw", "mean"), gain=("gain_raw", "mean")).reset_index()
        q = g[g.n >= 10]
        n25, n26 = (q[(q.name == "Josh Naylor") & (q.season == y)].iloc[0] for y in (2025, 2026))
        G = self.GL[self.GL.base == "2B"]; me = G[G.name == "Josh Naylor"].iloc[0]
        return dict(sb=int(s.SB), cs=int(s.CS), speed=s.sprint_speed, slower_than=(s25 > s.sprint_speed).mean(), timed=len(s25),
                    tracked=len(t), tracked_safe=int(t.y.sum()), n25=n25, n26=n26, seasons=len(q),
                    more_ground_25=(q.gain > n25.gain).mean(), most_ground_26=bool((q.gain <= n26.gain).all()),
                    gl=me, effect_rank=int((G.runner_effect > me.runner_effect).sum()) + 1, gl_n=len(G))

    # decision model
    def fx(self, base, term, model="decision"):
        return self.FX[(self.FX.base == base) & (self.FX.model == model) & (self.FX.term == term)].iloc[0]

    def vc(self, base, group, model="decision"):
        return self.VC[(self.VC.base == base) & (self.VC.model == model) & (self.VC.group == group)].iloc[0]

    def fw(self, base, model):
        return self.FW[(self.FW.base == base) & (self.FW.model == model)].iloc[0]

    def va(self, base, label):
        return self.VA[(self.VA.base == base) & (self.VA.margin_pts == label)].iloc[0]

    def be(self, base, outs):
        return self.BE[(self.BE.base == base) & (self.BE.outs == outs)].iloc[0].breakeven

    def swing(self, base, term):
        """Decision model: P at the 90th minus P at the 10th percentile of a continuous input (others at their averages),
        or 0 -> 1 for an indicator; with the coefficient's 95% CI."""
        r, b0 = self.fx(base, term), self.fx(base, "intercept").beta
        if term not in ("speed", "lead", "pop", "balls", "strikes"):
            return tuple(100 * (IQ.expit(b0 + r.beta + z * r.se) - IQ.expit(b0)) for z in (0, -1.96, 1.96))
        m = self.rows[base][term + "_raw"].mean()
        return tuple(100 * (IQ.expit(b0 + (r.beta + z * r.se) * (r.p90 - m)) - IQ.expit(b0 + (r.beta + z * r.se) * (r.p10 - m)))
                     for z in (0, -1.96, 1.96))

    # success model
    def sx(self, base, term):
        return self.SF[(self.SF.base == base) & (self.SF.term == term)].iloc[0]

    def sv(self, base, group):
        return self.SV[(self.SV.base == base) & (self.SV.group == group)].iloc[0]

    def sswing(self, base, term):
        """Success model, same convention; ground gained includes its curvature (point estimate only for the CI ends)."""
        r, b0 = self.sx(base, term), self.sx(base, "intercept").beta
        if term in ("out1", "out2", "lhp", "rhb", "brk", "off"):
            return tuple(100 * (IQ.expit(b0 + r.beta + z * r.se) - IQ.expit(b0)) for z in (0, -1.96, 1.96))
        a = self.srows[base]; m = (a[term + "_raw"] - a[term]).iloc[0]          # the centring constant of dec_rows
        lo_, hi_ = r.p10 - m, r.p90 - m
        if term == "gain":
            b2 = self.sx(base, "gain2").beta if (self.SF.term == "gain2").any() else 0.0
            eta = lambda bb, g: bb * g + b2 * g * g / 10
            return tuple(100 * (IQ.expit(b0 + eta(r.beta + z * r.se, hi_)) - IQ.expit(b0 + eta(r.beta + z * r.se, lo_))) for z in (0, -1.96, 1.96))
        return tuple(100 * (IQ.expit(b0 + (r.beta + z * r.se) * hi_) - IQ.expit(b0 + (r.beta + z * r.se) * lo_)) for z in (0, -1.96, 1.96))

    def ground(self, base, outs, catcher="average catcher"):
        return self.GN[(self.GN.base == base) & (self.GN.outs == outs) & (self.GN.catcher.str.startswith(catcher))].iloc[0]

    def w(self, model, base, term):
        """An input's weight in Figure 5: log-odds per SD (indicators 0 to 1)."""
        return self.IN[(self.IN.model == model) & (self.IN.base == base) & (self.IN.term == term)].iloc[0]


# ── the findings, one per figure, with the discussion bullets (used by the paper and the companion PDF) ────────────
def findings(D):
    assert all(get("3B", g).lo == 0 for get in (D.vc, D.sv) for g in ("runner", "pitcher", "catcher")), \
        "a steal-of-third SD interval now excludes zero: rewrite the variance finding and section 4.7"
    sp = D.SP; b2 = {o: D.be("2B", o) for o in range(3)}; b3 = {o: D.be("3B", o) for o in range(3)}
    c2 = {o: D.clean[("2B", o)][0] for o in range(3)}; c3 = {o: D.clean[("3B", o)][0] for o in range(3)}
    fs, fg = D.fifths["speed"], D.fifths["gain"]
    ratio = D.sel.loc["fastest", "attempt_pct"] / D.sel.loc["slowest", "attempt_pct"]
    nsig = int((sp.p < 0.05).sum())
    v2r, v2p, v2c = (D.vc("2B", g) for g in ("runner", "pitcher", "catcher"))
    s2p = D.sv("2B", "pitcher")
    v3p, v3c, v3r = (D.vc("3B", g) for g in ("pitcher", "catcher", "runner"))
    FA = D.FA; s, v, dc, spd = FA.loc[D.succ], FA.loc[D.v16], FA.loc[D.dec], FA.loc["Sprint speed only"]
    pub = FA[FA.family == "published"].auroc.max()
    g2, g2q, g3 = D.ground("2B", 1), D.ground("2B", 1, "quick"), D.ground("3B", 2)
    hold, go, top = D.va("2B", "hold (P < break-even)"), D.va("2B", "go (P >= break-even)"), D.va("2B", "> 10")
    G = D.GL[D.GL.base == "2B"]; be1 = G.be_1out.iloc[0]
    clear, tough = int((G.p_1out >= be1).sum()), int((G.p_tough_1out >= be1).sum())
    pop, lead, speed = D.swing("2B", "pop"), D.swing("2B", "lead"), D.swing("2B", "speed")
    wg, wb, ws0, ws1 = D.w("success", "2B", "gain"), D.w("success", "2B", "brk"), D.w("decision", "2B", "speed"), D.w("success", "2B", "speed")
    B = D.SB; fast, slow = B.iloc[-1], B.iloc[:3]
    R = D.RSP; ny = D.naylor
    return [
        dict(n=2, fig="Fig_paper_breakeven.png", title="The bar each steal must clear",
             bluf=f"A steal of second pays above {pct(min(b2.values()))}–{pct(max(b2.values()))} success; a steal of third needs "
                  f"{pct(b3[1])} with one out but {pct(b3[2])} with two.",
             x=f"Steals of second clear the bar at every out count; steals of third with two outs barely break even ({pct(c3[2], 1)} "
               f"against {pct(b3[2], 1)}).",
             y=f"Break-even from 2023–26 run expectancy ({D.RE.n.sum():,} plate appearances); observed success {pct(c2[0])}, "
               f"{pct(c2[1])} and {pct(c2[2])} on steals of second with first base only occupied.",
             z="Break-even computed for each base-out state from the RE24 table (equation 1); success measured in the same states.",
             use="Treat a two-out steal of third as a near coin toss in run value; reserve it for runners and batteries well above the bar."),
        dict(n=3, fig="Fig_paper_drivers.png", title="The jump decides; speed decides who runs",
             bluf=f"Ground gained during the delivery moves success {100 * (fg.iloc[-1] - fg.iloc[0]):.0f} points across its range; "
                  f"sprint speed moves it {100 * (fs.iloc[-1] - fs.iloc[0]):.0f}.",
             x="The ground a runner covers between the pitcher's first move and release is the main determinant of a steal of second.",
             y=f"Success {pct(fg.iloc[0])} in the bottom fifth of ground gained, {pct(fg.iloc[-1])} in the top; {pct(fs.iloc[0])} to "
               f"{pct(fs.iloc[-1])} across fifths of speed; the fastest fifth attempts {ratio:.0f}× as often as the slowest.",
             z="Success rates by fifths of each input on the same scale (95% Wilson intervals); attempt rates per opportunity.",
             use="Evaluate and coach the jump (secondary lead), not the sprint; speed tells you who will run, not who will be safe."),
        dict(n=4, fig="Fig_paper_prediction.png", title="How well a steal can be predicted",
             bluf=f"Given the ground gained and the pitch, the success model ranks 2026 steals at AUROC {s.auroc:.3f}, ahead of the "
                  f"v16 logistic ({v.auroc:.3f}) and published specifications (at most {pub:.3f}); before the pitch, the decision "
                  f"model reaches {dc.auroc:.3f}.",
             x=f"What separates a stolen base from a caught stealing is measured mainly during the delivery: the ground gained and the "
               f"pitch add {s.auroc - dc.auroc:.2f} of AUROC to the pre-pitch inputs.",
             y=f"{int(s.n):,} attempts of 2026: AUROC {s.auroc:.3f} (95% CI {s.auroc_lo:.3f}–{s.auroc_hi:.3f}); gain over v16 "
               f"{s.d_auroc_vs_v16_tuned:+.3f} (paired 95% CI {s.d_lo:+.3f} to {s.d_hi:+.3f}); sprint speed alone {spd.auroc:.3f}.",
             z="Every model fit on 2023–25 and applied once to the same 2026 attempts, the success model with each attempt's measured "
               "ground gained and pitch type; specification chosen by cross-validation on 2023–25; runner-clustered intervals.",
             use="Judge attempts and players after the fact with the success model; make the call with the pre-pitch model. The gap "
                 "between them is the value of the jump."),
        dict(n=5, fig="Fig_paper_influence.png", title="What moves the odds",
             bluf=f"Once the jump is known, one SD of ground gained multiplies the odds of success by {np.exp(wg.per_sd):.1f}, a "
                  f"breaking ball by {np.exp(wb.per_sd):.1f}; before the pitch, the catcher's pop time weighs most.",
             x="Before the pitch the battery and the lead matter more than speed; during the delivery the jump dwarfs everything.",
             y=f"Steals of second, log-odds per SD: ground gained {wg.per_sd:+.2f}; sprint speed {ws0.per_sd:+.2f} before the pitch and "
               f"{ws1.per_sd:+.2f} given the ground; pop time worth {pop[0]:.1f} points from a quick to a slow catcher, lead "
               f"{lead[0]:.1f}, speed {speed[0]:.1f}.",
             z="Coefficients of the decision and success models times each input's standard deviation, with 95% confidence intervals.",
             use="Run on slow-popping catchers and in pitchers' counts; ask for the extra foot of lead; then coach the jump, which "
                 "is worth more than everything else combined once the pitcher moves."),
        dict(n=6, fig="Fig_paper_ground.png", title="What it takes to be safe",
             bluf=f"A steal of second with one out needs about {g2.ground_needed_ft:.1f} ft of ground by release against an average "
                  f"catcher and {g2q.ground_needed_ft:.1f} ft against a quick one; a steal of third with two outs needs "
                  f"{g3.ground_needed_ft:.1f} ft.",
             x="The break-even rate translates into a ground target that runners and coaches can train and scout for.",
             y=f"{g2.share_reaching:.0%} of 2023–26 steals of second reached the one-out target against an average catcher; "
               f"{g2q.share_reaching:.0%} reached the target against a quick catcher.",
             z="Success model at the league average for runner, pitcher, count and pitch mix, one out, right-handed pitcher; the ground "
               "at which the predicted probability equals the break-even rate.",
             use="Send a runner when the ground expected against this pitcher and catcher clears the target; train the jump toward it."),
        dict(n=7, fig="Fig_paper_speed.png", title="Speed is significant, and small, every season",
             bluf=f"Speed is a significant predictor in {nsig} of {len(sp)} seasons, yet it moves success only "
                  f"{sp.swing_pts.min():.1f}–{sp.swing_pts.max():.1f} points from slow to fast runners.",
             x="Both statements hold: the effect is real and small. The many attempts of fast runners make a small slope precise.",
             y=f"Speed-only logistic slope {sp.beta.min():.2f}–{sp.beta.max():.2f} per ft/s; AUROC {sp.auroc.min():.2f}–{sp.auroc.max():.2f}; "
               f"counts by speed and season in Table 3.",
             z="Season-by-season logistic regressions on steals of second with runner-clustered standard errors; 0.5 ft/s bins.",
             use="Answer to 'speed is significant': yes, and it is worth about five points of success; size a decision by points, not p-values."),
        dict(n=8, fig="Fig_paper_runners.png", title="Fast runners run more, not better",
             bluf=f"Runner-seasons at 30 ft/s and up average {fast.attempts_per_runner_season:.1f} attempts of second against "
                  f"{slow.attempts_per_runner_season.min():.1f}–{slow.attempts_per_runner_season.max():.1f} below 26.5 ft/s; among "
                  "regular base stealers, a runner's success rate barely depends on his speed.",
             x="Speed decides how often a runner is sent, not how often he is safe; the slowest regular base stealer is among the best.",
             y=f"r = {R.r:+.2f} between sprint speed and success across {int(R.runner_seasons)} runner-seasons with 10+ attempts; "
               f"allowing for sampling noise, speed accounts for about {100 * R.r_true ** 2:.0f}% of the true differences "
               f"(about {100 * R.true_sd:.0f} points SD); Naylor: {int(ny['n25'].safe)} of {int(ny['n25'].n)} and "
               f"{int(ny['n26'].safe)} of {int(ny['n26'].n)}.",
             z="Each runner-season's success rate against his season sprint speed; Spearman's correction for the binomial noise in "
               "a season's success rate.",
             use="Do not hand out green lights by sprint speed; judge runners by their jump and their record, shrunk for sample size."),
        dict(n=9, fig="Fig_paper_variance.png", title="Who controls the running game",
             bluf=f"On steals of second a runner one SD above average adds {v2r.pp_plus_1sd:.1f} points, a pitcher "
                  f"{v2p.pp_plus_1sd:.1f} and a catcher {v2c.pp_plus_1sd:.1f}; on steals of third no group's differences can yet be "
                  "told apart from chance.",
             x="Pitchers influence steals of second mostly through the ground they allow and the pitch they throw.",
             y=f"Pitcher SD {v2p.sigma:.2f} before the pitch, {s2p.sigma:.2f} (95% CI {s2p.lo:.2f} to {s2p.hi:.2f}) once ground gained and "
               f"pitch type are known; steals of third: pitcher {pts(v3p.pp_plus_1sd)}, catcher {pts(v3c.pp_plus_1sd)}, runner "
               f"{pts(v3r.pp_plus_1sd)} pts per SD, every 95% interval reaching zero.",
             z="Random-intercept standard deviations from the decision and success models with 95% likelihood-ratio intervals, "
               "converted to points at the league average.",
             use="Grade pitchers on time to the plate, hold and pitch mix (the ground they give up); on steals of third, lean on "
                 "measured inputs such as pop time and the lead rather than on player reputations."),
        dict(n=10, fig="Fig_paper_decision_value.png", title="The calls, scored on 2026",
             bluf=f"The {hold.n:.0f} steals of second the pre-pitch model would have held lost {abs(hold.total_runs):.0f} runs; the ones "
                  f"it would have sent gained {go.total_runs:.0f}.",
             x="The model separates paying from losing attempts among those teams actually made, and the margin over the bar predicts the payoff.",
             y=f"Hold: success {pct(hold.success, 1)} against a bar of {pct(hold.mean_be, 1)}, {hold.runs_per_attempt:+.3f} runs per attempt. "
               f"Clearing the bar by 10+ points: {top.runs_per_attempt:+.3f} runs per attempt ({top.n:.0f} attempts).",
             z="Each 2026 attempt marked go when P(safe) ≥ the state's break-even (RE24 from 2023–25), credited with the runs it produced.",
             use="Hold attempts the model scores below the bar; the payoff of an attempt grows with its margin over the bar."),
        dict(n=11, fig="Fig_paper_greenlight.png", title="Who to send in 2027",
             bluf=f"Against an average battery {clear} of {len(G)} regular base stealers clear the one-out bar; against a battery one SD "
                  f"tougher, {tough} do.",
             x="Who to send depends as much on the battery as on the runner: half the green lights turn red against the toughest batteries.",
             y=f"Predicted success on a steal of second with one out, from the 2023–26 pre-pitch model; break-even {pct(be1)}; runners "
               f"with 10+ attempts in 2025–26; tiers in Table 4.",
             z="Runner's latest sprint speed and typical lead plus his random intercept; battery at the league average or one SD tougher.",
             use="Give a standing green light to runners who clear the bar even against tough batteries; make the rest battery-dependent."),
    ]


# ── the paper ────────────────────────────────────────────────────────────────────────────────────────────────────
def paper(D):
    SE, sp, F = D.SE, D.SP, findings(D)
    FA = D.FA; s, v, vd, dc, spd = FA.loc[D.succ], FA.loc[D.v16], FA.loc[D.v16d], FA.loc[D.dec], FA.loc["Sprint speed only"]
    pubs, powers = FA[FA.family == "published"], FA.loc["Powers et al. specification (lead, speed, arm), refit"]
    n2, n3 = len(D.rows["2B"]), len(D.rows["3B"])
    a2, a3 = (D.att.base == "2B").sum(), (D.att.base == "3B").sum()
    b2, b3 = {o: D.be("2B", o) for o in range(3)}, {o: D.be("3B", o) for o in range(3)}
    c2, c3 = {o: D.clean[("2B", o)][0] for o in range(3)}, {o: D.clean[("3B", o)][0] for o in range(3)}
    games, pas = int(D.RE.groupby("season").games.first().sum()), int(D.RE.n.sum())
    timed = sum((d.qa == "PASS").sum() for d in D.cv.values()); tot = sum(len(d) for d in D.cv.values())
    g2, n2m = D.fw("2B", "decision GLMM"), D.fw("2B", "decision inputs, no player effects")
    g3, n3m = D.fw("3B", "decision GLMM"), D.fw("3B", "decision inputs, no player effects")
    hold, go, top = D.va("2B", "hold (P < break-even)"), D.va("2B", "go (P >= break-even)"), D.va("2B", "> 10")
    top3 = D.va("3B", "> 10")
    G = D.GL[D.GL.base == "2B"]; be1 = G.be_1out.iloc[0]
    clear, tough = int((G.p_1out >= be1).sum()), int((G.p_tough_1out >= be1).sum())
    v2r, v2p, v2c = D.vc("2B", "runner"), D.vc("2B", "pitcher"), D.vc("2B", "catcher")
    s2r, s2p = D.sv("2B", "runner"), D.sv("2B", "pitcher")
    v3p, v3c, v3r = D.vc("3B", "pitcher"), D.vc("3B", "catcher"), D.vc("3B", "runner")
    pop, lead, speed, balls, rhb, lhp = (D.swing("2B", t) for t in ("pop", "lead", "speed", "balls", "rhb", "lhp"))
    fs, fg = D.fifths["speed"], D.fifths["gain"]
    lr, rr = D.att_rate_hand[1], D.att_rate_hand[0]
    ck = D.CK.set_index("param")
    gr = {(b, o, c): D.ground(b, o, c) for b in ("2B", "3B") for o in range(3) for c in ("average", "quick", "slow")}
    v16cv = D.V16CV.loc["v16: Optuna-tuned (nested)", "auroc"]
    A, R, ny, B = D.AN, D.RSP, D.naylor, D.SB
    w = lambda model, term: D.w(model, "2B", term)
    r_ga = D.JT.set_index("item").loc["attempt: ground gained vs sprint speed", "value"]
    W2 = D.IN[(D.IN.model == "decision") & (D.IN.base == "2B")]
    assert W2.loc[W2.per_sd.abs().idxmax(), "term"] == "pop", "pop time is no longer the top pre-pitch weight: rewrite 4.4"

    st = [Spacer(1, 18), para("StealIQ: When to Steal", TITLE),
          para("Predicting stolen-base success and the decision to run, 2023–2026", SUB),
          para("<b>Abstract.</b> Josh Naylor was one of the slowest runners in the major leagues in 2025, and he stole "
               f"{ny['sb']} bases in {ny['sb'] + ny['cs']} attempts. He did it with his jump: by the time the pitch left the "
               "pitcher's hand he had gained more ground than almost any base stealer. We study every Statcast-tracked attempt to "
               f"steal second ({a2:,}) and third ({a3:,}) in 2023–26 with two logistic mixed models per base, each giving every "
               "runner, pitcher and catcher a random intercept. The <i>success model</i> uses everything measured by the release of "
               "the pitch. Fit on 2023–25 and applied to each 2026 attempt with its measured ground gained and pitch type, it ranks "
               f"the outcomes at AUROC {s.auroc:.3f} (95% CI {s.auroc_lo:.3f}–{s.auroc_hi:.3f}), ahead of our earlier pooled logistic "
               f"model ({v.auroc:.3f}), published specifications refit to the same data ({pubs.auroc.min():.2f}–{pubs.auroc.max():.2f}) "
               f"and sprint speed alone ({spd.auroc:.2f}). The <i>decision model</i> uses only what is known before the pitch. It "
               f"ranks less well ({dc.auroc:.3f}) but is calibrated, so its probability can be set against the break-even rate of "
               "each base-out state from 2023–26 run expectancy; the "
               f"{hold.n / g2.n_test:.0%} of 2026 steals of second it would have held lost {abs(hold.total_runs):.0f} runs. A steal of "
               f"second with one out needs about {gr[('2B', 1, 'average')].ground_needed_ft:.1f} feet of ground by release. Against "
               f"an average battery {clear} of {len(G)} regular base stealers clear the one-out break-even rate; against a tough one, "
               f"{tough} do.", ABS),
          para("<b>Keywords:</b> stolen bases; outcome prediction; break-even analysis; run expectancy (RE24); generalized linear "
               "mixed models; Statcast.", ABS),
          Spacer(1, 6)]

    st += [para("1. Introduction", H1),
           para(f"In 2025 Josh Naylor stole {ny['sb']} bases and was caught {'twice' if ny['cs'] == 2 else str(ny['cs']) + ' times'}. "
                "He was among the slowest runners in "
                f"the major leagues: his sprint speed, {ny['speed']:.1f} feet per second, was slower than {ny['slower_than']:.0%} of "
                f"the {ny['timed']} players Statcast timed that season. He did not take an unusually long lead either. On his "
                f"tracked steals of second he stood {ny['n25'].lead:.1f} feet off first base when the pitcher moved, a little short "
                f"of the typical {A.lead_ft:.1f}. What set him apart happened in the next second: by the time the pitch left the "
                f"pitcher's hand he had gained another {ny['n25'].gain:.1f} feet, more than in all but "
                f"{max(1, round(100 * ny['more_ground_25'])):.0f}% of the {ny['seasons']} runner-seasons with ten or more attempts. "
                "He won the race in the part of it that sprint speed does not measure."),
           para("The 2023 rule changes (larger bases, a limit of two disengagements per plate appearance and the pitch timer) "
                f"brought the stolen base back. Attempts rose from {SE.loc[2022, 'attempts_per_game']:.2f} per game in 2022 to "
                f"{SE.loc[2023, 'attempts_per_game']:.2f} in 2023 and the league success rate from {SE.loc[2022, 'success_pct']:.1f}% "
                f"to {SE.loc[2023, 'success_pct']:.1f}%; by 2026 it had eased to {SE.loc[2026, 'success_pct']:.1f}% on "
                f"{SE.loc[2026, 'attempts_per_game']:.2f} attempts per game (Table 1)."),
           para("The usual shorthand for who should run is sprint speed, and it is not wrong: Powers et al. (2026) find sprint "
                "speed a significant predictor of success in a multilevel model of the running game. Naylor shows why it is not "
                "the whole story. A front office needs two things the shorthand does not give it: how well the outcome of an "
                "attempt can be predicted, and which attempts are worth making. An attempt is worth making when its chance of "
                "success clears the break-even rate of its base-out state (Tango et al., 2007), a bar that runs from "
                f"{pct(min(min(b2.values()), min(b3.values())))} to {pct(max(b3.values()))} depending on the base and the outs."),
           para("We answer both questions for steals of second and third separately: a foot of lead off second is not a foot off "
                "first, and the throw and the bar differ. Each base gets two models. A success model uses everything measured by "
                "the release of the pitch, including the ground gained, and rates attempts and players after the fact. A decision "
                "model uses only what is known before the pitch and makes the call. The gap between them measures how much of a "
                "steal is decided after the runner commits. Section 2 defines every measurement and why it matters, Section 3 "
                "sets out the methods, Section 4 reports the findings, from the bar a steal must clear to a 2027 green-light list, "
                "and Section 5 turns them into decisions.")]

    rows = [["Season", "Attempts of 2nd", "Success, 2nd", "Attempts of 3rd", "Success, 3rd", "In models (2nd / 3rd)",
             "MLB attempts per game", "MLB success"]]
    for s_ in range(2023, 2027):
        d = D.att[D.att.season == s_]
        rows.append([str(s_), f"{(d.base == '2B').sum():,}", pct(d[d.base == '2B'].y.mean(), 1), f"{(d.base == '3B').sum():,}",
                     pct(d[d.base == '3B'].y.mean(), 1),
                     f"{(D.rows['2B'].season == s_).sum():,} / {(D.rows['3B'].season == s_).sum():,}",
                     f"{SE.loc[s_, 'attempts_per_game']:.2f}", f"{SE.loc[s_, 'success_pct']:.1f}%"])
    rows.append(["2023–26", f"{a2:,}", pct(D.att[D.att.base == '2B'].y.mean(), 1), f"{a3:,}", pct(D.att[D.att.base == '3B'].y.mean(), 1),
                 f"{n2:,} / {n3:,}", "", ""])
    st += [para("2. Data and measurements", H1), para("2.1 Attempts", H2),
           para("Baseball Savant's running-game data give every stolen-base attempt of 2023–26 that Statcast tracked, with the "
                "runner, pitcher and catcher involved and the runner's lead at the pitcher's first move and at release. We join "
                "each attempt to its Statcast pitch record (count, outs, base state, handedness, pitch type), the runner's season "
                f"sprint speed and the catcher's season pop time. Of {a2 + a3:,} attempts, {a2:,} were steals of second and {a3:,} "
                f"of third; {n2:,} and {n3:,} have every input (the rest lack a lead measurement or a pop time). Run expectancy "
                f"comes from {pas:,} plate appearances in the {games:,} games of the same seasons with a tracked attempt "
                f"({games / D.games_played:.0%} of the {D.games_played:,} played)."),
           table(rows, [0.6, 0.85, 0.7, 0.85, 0.7, 1.05, 0.95, 0.8], caption=(1, "Stolen-base attempts by season and base."),
                 note="Attempts and success: Statcast-tracked attempts (stolen base or caught stealing). MLB columns: all regular-season "
                      "attempts (StatsAPI). Models use attempts with every input present.", bold_last=True),
           para("2.2 The race, and what we measure", H2),
           para("A steal is a race between two clocks (Figure 1). The runner's clock starts when the pitcher makes his first move, and "
                "by then he is already some feet off the base. While the pitcher delivers, the runner gains more ground; after the "
                "release he sprints the rest of the way. The defense's clock is the pitcher's delivery, the flight of the pitch and "
                f"the catcher's throw. In the typical steal of second the pitcher takes {A.delivery_s:.2f} s to deliver, the pitch "
                f"about {A.flight_s:.1f} s to arrive and the catcher {A.pop_s:.2f} s to get the ball to second base, "
                f"{A.defense_s:.1f} s in all; the runner starts {A.lead_ft:.1f} feet off first, has gained {A.gain_ft:.1f} feet by "
                f"the release and must sprint the last {A.left_to_run_ft:.0f}. Every input in this paper measures, or shapes, one leg of "
                "that race."),
           figure("Fig_paper_anatomy.png", 1, "<b>Anatomy of a steal of second.</b> The median attempt of 2023–26, in seconds from "
                  "the pitcher's first move. Runner (blue): his lead at the first move, the ground he gains by the release, and the "
                  "distance left to sprint. Defense: the pitcher's delivery (measured from video), the flight of the pitch "
                  "(estimated from release speed and extension) and the catcher's pop time. Inputs known before the pitch (lead at "
                  "the first move, sprint speed, pop time, count, outs, handedness) enter both models; those measured during the "
                  "delivery (ground gained, pitch type) enter only the success model."),
           para("<b>Lead at the pitcher's first move</b> (feet). The distance from the base to the runner's centre of mass when the "
                "pitcher makes his first movement, toward the plate or the base (Statcast's lead distance). It is the last "
                "distance the runner chooses before he commits, and it is known before the pitch: a foot more of lead is a foot "
                "less to run."),
           para("<b>Ground gained</b> (feet). The distance the runner covers between the pitcher's first move and the release of "
                "the pitch: his lead at release minus his lead at the first move (Baseball Savant's lead distance gained). It is "
                "the runner's jump, measured in feet, and two things set it: how quickly he reads the pitcher and gets going, and "
                "how long the pitcher takes to deliver, since a slow delivery hands the runner ground. Because it ends at the "
                "release, it says how far ahead the runner is at the moment the ball starts toward the catcher, which makes it "
                "the most direct measure tracking gives of how the race stands. It cannot be known before the pitch, because the "
                "runner earns it during the delivery, so it enters the success model only. The typical steal of second gains "
                f"{A.gain_ft:.1f} feet; Naylor averaged {ny['n25'].gain:.1f} in 2025."),
           para("<b>Sprint speed</b> (feet per second). The runner's speed in his fastest one-second window, averaged over the best "
                "two-thirds of his competitive runs that season (Statcast). It measures how fast he covers the distance left after "
                "the release, and it is a season number, the same for every attempt he makes that year."),
           para("<b>Catcher's pop time</b> (seconds). The time from the pitch reaching the catcher's mitt to the throw reaching the "
                "fielder at the base, averaged over the catcher's throws on steal attempts that season (Statcast). It is the "
                "defense's last leg. We use the pop time to the base being stolen, to second for steals of second and to third "
                "for steals of third; for steals of third it fits better than pop time to second (Appendix Table C3)."),
           para("<b>Count, outs and handedness.</b> The balls and strikes before the pitch, the outs, and the handedness of pitcher "
                "and batter, all known before the pitch. The count shapes which pitch is coming; the outs move the bar; a "
                "left-handed pitcher faces first base and can hold a runner there more closely; and a right-handed batter stands "
                "in the catcher's throwing lane to third."),
           para("<b>Pitch type.</b> Fastball (four-seam, sinker, cutter), breaking ball (slider, curveball, sweeper and the like) "
                "or offspeed pitch (changeup, splitter and the like), from Statcast's pitch classification. A slower pitch takes "
                "longer to reach the catcher and is harder to catch and throw from. The runner cannot know it before he goes, so "
                "it enters the success model only."),
           para("<b>Delivery time</b> (seconds). The pitcher's time from lifting his lead foot to releasing the ball, measured from "
                "broadcast video (Section 3.5). It is not a model input; it splits ground gained into the runner's part and the "
                "pitcher's part.")]

    st += [para("3. Methods", H1), para("3.1 The bar: break-even rates from RE24", H2),
           para("Run expectancy is the average number of runs a team scores from a given base-out state to the end of the inning. "
                "There are 24 such states, eight arrangements of runners by three out counts, and the table of their run "
                "expectancies, RE24, is the standard way to price a play that changes the state (Tango et al., 2007). We compute "
                "RE24 ourselves from the plate appearances of 2023–26 (innings 1–8, because an inning that can end on a walk-off "
                "is cut short; Appendix Table A1) rather than borrow a published table, so that the bar is set in the same run "
                "environment as the attempts it judges."),
           para("A steal attempt from state (b, o) leaves the team in (b′, o) if the runner is safe and in (b″, o + 1) if he is "
                "caught. With success probability p, going is worth p·RE(b′, o) + (1 − p)·RE(b″, o + 1) runs against RE(b, o) for "
                "staying, so it pays when"),
           para("p &gt; p* = [ RE(b, o) − RE(b″, o + 1) ] / [ RE(b′, o) − RE(b″, o + 1) ].        (1)", EQ),
           para(f"For a runner on first with one out, RE is {RE_(D, '1__', 1):.3f} now, {RE_(D, '_2_', 1):.3f} with a runner on "
                f"second and {RE_(D, '___', 2):.3f} with the bases empty and two outs, so p* = "
                f"{RE_(D, '1__', 1) - RE_(D, '___', 2):.3f} / {RE_(D, '_2_', 1) - RE_(D, '___', 2):.3f} = {pct(b2[1], 1)}. We compute "
                "p* for every attempt's own state; a runner already standing on the target base moves up as in a double steal, "
                "and a third out ends the inning (RE = 0). Baseball Savant's run value credits every advance with about +0.2 runs "
                "and every out with −0.45, whatever the base or the outs, so it cannot give a bar that depends on the situation; "
                f"and the bar does depend on it, from {pct(min(min(b2.values()), min(b3.values())))} to {pct(max(b3.values()))} "
                "(Section 4.1)."),
           para("3.2 Two models per base", H2),
           para("For each base we fit two logistic generalized linear mixed models (GLMMs) of the form"),
           para("logit P(safe) = β′x + u<sub>runner</sub> + u<sub>pitcher</sub> + u<sub>catcher</sub>,   "
                "u ~ N(0, σ²) for each group,", EQ),
           para("with crossed random intercepts for runner, pitcher and catcher, the structure of Powers et al. (2026). In the "
                "<i>decision model</i>, x holds what is known when the runner decides: sprint speed, lead at the first move, pop "
                "time, balls, strikes, outs and the handedness of pitcher and batter. The <i>success model</i> adds what the play "
                "measures by the release: ground gained, with a squared term for its curvature, and the pitch type. Its strongest "
                "input arrives after the runner has committed, so it rates attempts and players and sets targets; it never makes "
                "the call."),
           para("Why a mixed model. A player's random intercept is that player's estimated skill, shrunk toward the league average "
                "in proportion to how little the player has been seen: a runner who went five for five keeps little of that record, "
                "a runner seen eighty times keeps most of it. That shrinkage is what makes the player effects usable as skill "
                "estimates. A logistic regression without player terms ignores who is running; one with a dummy per player "
                "overfits the many players seen a handful of times."),
           para("3.3 Estimation and checks", H2),
           para("The models are fit by maximum likelihood with the Laplace approximation, the method of lme4's glmer (Bates et al., "
                "2015), implemented in Python with the rest of the analysis. The success model's specification was chosen among "
                "three candidates (v16's inputs; adding the count, outs and handedness; adding the curvature in ground gained) by "
                "five-fold cross-validated log-loss on 2023–25 attempts only (Appendix Table C2); 2026 was not used. Refitting the "
                "decision model for steals of second to 20 data sets simulated from its own estimates recovers every fixed effect "
                f"within Monte Carlo error and the runner SD almost exactly ({ck.loc['sigma runner', 'mean_est']:.3f} against "
                f"{ck.loc['sigma runner', 'truth']:.3f}); the pitcher and catcher SDs come back 11–14% low, the known small-sample "
                "bias of maximum-likelihood variance estimates. An independent variational-Bayes fit (statsmodels) agrees to "
                "within 0.04 on every fixed effect and 0.012 on every SD (Appendix Table C1)."),
           para("3.4 How the models are tested", H2),
           para("Every model is fit on 2023–25 and scored once on the 2026 attempts that all of them can score; players first seen "
                f"in 2026 get a zero effect (they made {g2.new_runners:.0%} of 2026 steals of second). Scoring the success model "
                "on 2026 means applying the 2023–25 fit to each 2026 attempt's own inputs, including the ground gained and the "
                "pitch type measured on that attempt. It answers the question <i>given how the runner broke and what was thrown, how "
                "likely was he to be safe</i>, not <i>should he go</i>. Alongside our models we fit v16, this project's earlier pooled logistic "
                "regression (both bases in one model, with base as an input, at default and tuned settings), the specifications "
                "of Powers et al. (2026) and The Pitcher's Dilemma (42 Analytics), and sprint speed alone."),
           para("Three measures judge them. AUROC is the chance that a randomly chosen stolen base gets a higher predicted "
                "probability than a randomly chosen caught stealing (0.5 is a coin flip); it measures ranking, and we give it a "
                "runner-clustered bootstrap interval and a paired difference from v16. Log-loss measures whether the probabilities "
                "themselves are right, attempt by attempt. The calibration slope measures whether they can be taken at face value: "
                "at 1, a predicted 80% succeeds 80% of the time. The last matters most for the call, which compares a probability "
                "with a break-even rate. To score the calls, each 2026 attempt is marked <i>go</i> when the decision model's "
                "probability is at least the break-even rate of its state (RE24 from 2023–25 only) and <i>hold</i> otherwise, and "
                "credited with the runs it produced: RE after the outcome minus RE before. Only attempts that were made are "
                "observed, so this measures how well the model separates good attempts from bad among those teams chose; it "
                "cannot price attempts never made."),
           para("3.5 Delivery time from video", H2),
           para("Ground gained mixes the runner's jump with the pitcher's delivery, and a front office wants to know which one it "
                "is buying. Statcast does not publish delivery times pitch by pitch, so we measured them from the broadcast clip "
                "of every attempt: the time from the pitcher's lead foot leaving the ground to the ball leaving his hand, found by "
                "object detectors trained on broadcast baseball (BaseballCV) and a pose model (RTMPose; Jiang et al., 2023), "
                "behind a camera check that rejects clips that cut away during the delivery. Against 50 hand-labelled clips the "
                f"error averages {1000 * D.cv_mae:.0f} ms, and {timed:,} attempts ({timed / tot:.0%}) were timed. With the time "
                "known, ground gained splits into the runner's speed over the jump (ground divided by time) and the pitcher's time "
                "(Appendix D). Delivery time is not a model input because "
                f"{1 - timed / tot:.0%} of attempts lack it.")]

    st += [para("4. Results", H1), para("4.1 The bar each steal must clear", H2),
           para(f"For a steal of second with only first base occupied the bar is {pct(b2[0])} with nobody out and {pct(b2[1])} with "
                f"one or two (Figure 2A). Teams cleared it comfortably, succeeding {pct(c2[0])}, {pct(c2[1])} and {pct(c2[2])} of the "
                f"time. A steal of third is a different play: the bar is {pct(b3[0])} with nobody out, falls to {pct(b3[1])} with one, "
                f"when a runner on third scores on so many more outs, and rises to {pct(b3[2])} with two (Figure 2B). Teams cleared "
                f"the one-out bar by {100 * (c3[1] - b3[1]):.0f} points {two_out(c3[2], b3[2])} ({pct(c3[2], 1)} against "
                f"{pct(b3[2], 1)}): a two-out steal of third is close to break-even."),
           figure("Fig_paper_breakeven.png", 2, "<b>The bar each steal must clear.</b> Break-even success rate from 2023–26 RE24 "
                  "(black) and observed success rate with 95% Wilson interval (blue), by out count, for steals of second with only "
                  "first base occupied (A) and steals of third with only second occupied (B)."),
           para("4.2 The jump decides; speed decides who runs", H2),
           para("Before any model, the raw rates make the case. Success on steals of second climbs from "
                f"{pct(fg.iloc[0])} in the bottom fifth of ground gained to {pct(fg.iloc[-1])} in the top fifth; across the fifths of "
                f"sprint speed it moves from {pct(fs.iloc[0])} to {pct(fs.iloc[-1])} (Figure 3A–B). Speed matters most before the "
                f"steal: the fastest fifth of runners attempts {D.sel.loc['fastest', 'attempt_pct']:.1f} steals per 100 chances and "
                f"the slowest {D.sel.loc['slowest', 'attempt_pct']:.1f} (Figure 3C). Speed decides who runs; the jump decides who is safe."),
           figure("Fig_paper_drivers.png", 3, "<b>The jump decides; speed decides who runs.</b> Success rate on steals of second by "
                  "fifths of sprint speed (A) and of ground gained between the pitcher's first move and release (B), on the same "
                  "scale, with 95% Wilson intervals at the median of each fifth. (C) Attempts per 100 opportunities by sprint-speed "
                  "fifth, every pitch with a runner on first and second base open, 2023–26."),
           para("4.3 How well a steal can be predicted", H2),
           para(f"Fit on 2023–25 and applied to the {int(s.n):,} attempts of 2026 with their measured ground gained and pitch type, "
                f"the success model ranks the outcomes at AUROC {s.auroc:.3f} (95% CI {s.auroc_lo:.3f}–{s.auroc_hi:.3f}; Figure 4A, "
                f"Table 2): a stolen base outranks a caught stealing {s.auroc:.0%} of the time. It improves on v16 ({v.auroc:.3f} "
                f"tuned, {vd.auroc:.3f} at default settings) by {s.d_auroc_vs_v16_tuned:+.3f} (paired 95% CI {s.d_lo:+.3f} to "
                f"{s.d_hi:+.3f}) and lowers log-loss from {v.log_loss:.3f} to {s.log_loss:.3f}. The gain comes from several "
                "changes at once (separate bases, player effects, the count and handedness, the curvature in ground gained) that "
                "this comparison does not separate. The published specifications, which do not use the ground gained on the "
                f"attempt itself, reach {pubs.auroc.min():.3f}–{pubs.auroc.max():.3f}, and sprint speed alone {spd.auroc:.3f}. The "
                f"success model is close to calibrated (slope {s.cal_slope:.2f}; Figure 4B)."),
           para(f"The decision model, which sees only what is known before the pitch, reaches {dc.auroc:.3f}, against "
                f"{powers.auroc:.3f} for the Powers et al. specification, whose inputs are also known before the pitch. The gap of "
                f"{s.auroc - dc.auroc:.2f} to the success model is what arrives during the delivery: the jump and the pitch. That "
                "gap is why the call rests on calibration rather than ranking. On 2026 the decision model's probabilities can be "
                f"taken at face value (slope {g2.cal_slope:.2f}, intercept {g2.cal_intercept:+.2f} on steals of second), so they can "
                "be set directly against the break-even rate. Its player effects "
                + (f"help a little on steals of second (log-loss {g2.log_loss:.3f} against {n2m.log_loss:.3f} without them)"
                   if g2.log_loss < n2m.log_loss else f"do not help on steals of second (log-loss {g2.log_loss:.3f} against {n2m.log_loss:.3f})")
                + (f" and not on steals of third ({g3.log_loss:.3f} against {n3m.log_loss:.3f}), where {g3.new_pitchers:.0%} of 2026 "
                   "attempts came against pitchers unseen in 2023–25." if g3.log_loss >= n3m.log_loss else
                   f" and on steals of third ({g3.log_loss:.3f} against {n3m.log_loss:.3f}).")),
           figure("Fig_paper_prediction.png", 4, "<b>How well a steal can be predicted.</b> (A) ROC curves on the 2026 attempts for "
                  "five of the models in Table 2, each fit on 2023–25, with AUROC in the legend. (B) Observed success rate against "
                  "predicted probability for the success model and v16, in tenths of predicted probability, with 95% Wilson "
                  "intervals; points on the diagonal are perfectly calibrated.")]
    rows = [["Model (fit on 2023–25, scored on 2026)", "AUROC (95% CI)", "2nd", "3rd", "Log-loss", "Calib. slope",
             "Δ AUROC vs v16 tuned (95% CI)"]]
    short = {"League success rate": "League success rate", "Sprint speed only": "Sprint speed only",
             "Powers et al. specification (lead, speed, arm), refit": "Powers et al. specification",
             "Pitcher's Dilemma specification (lead, speed, jump, pop), refit": "Pitcher's Dilemma specification",
             D.dec: "Decision GLMM, pre-pitch", D.v16d: "v16 logistic, default", D.v16: "v16 logistic, tuned", D.succ: "<b>Success GLMM</b>"}
    for m in ["League success rate", "Sprint speed only", "Powers et al. specification (lead, speed, arm), refit",
              "Pitcher's Dilemma specification (lead, speed, jump, pop), refit", D.dec, D.v16d, D.v16, D.succ]:
        r = FA.loc[m]; flat = r.auroc == 0.5
        rows.append([short[m], "0.500" if flat else f"{r.auroc:.3f} ({r.auroc_lo:.3f}–{r.auroc_hi:.3f})",
                     "—" if flat else f"{r.auroc_2b:.3f}", "—" if flat else f"{r.auroc_3b:.3f}", f"{r.log_loss:.3f}",
                     "—" if pd.isna(r.cal_slope) else f"{r.cal_slope:.2f}",
                     "—" if pd.isna(r.d_auroc_vs_v16_tuned) or m == D.v16 else f"{r.d_auroc_vs_v16_tuned:+.3f} ({r.d_lo:+.3f} to {r.d_hi:+.3f})"])
    st.append(table(rows, [2.05, 1.15, 0.42, 0.42, 0.55, 0.5, 1.41], caption=(2, "Predicting 2026 steals from 2023–25."),
                    note=f"{int(s.n):,} attempts of 2026 that every model can score ({int(s.n_2b):,} of second, {int(s.n_3b):,} of third). "
                         "The success GLMM and v16 use each attempt's measured ground gained and pitch type. AUROC intervals and paired "
                         "differences from 500 runner-clustered bootstrap resamples. Published specifications refit on 2023–25: Powers "
                         "et al. (lead, speed, catcher arm) and The Pitcher's Dilemma (lead, speed, season jump speed, pop time). v16: "
                         "pooled logistic regression on sprint speed, lead at first move, ground gained, pop time, pitch class and base; "
                         f"its cross-validated AUROC on 2023–26 is {v16cv:.3f}."))

    st += [para("4.4 What moves the odds", H2),
           para("Figure 5 puts every input on one scale: the change in the log-odds of success for a one-standard-deviation "
                "increase, or from 0 to 1 for an indicator. Before the pitch (Figure 5A), the catcher's pop time weighs most on "
                f"steals of second ({w('decision', 'pop').per_sd:+.2f} per SD): a slow catcher at the 90th percentile gives up "
                f"{pop[0]:.1f} points of success over a quick one at the 10th. A long lead at the first move is worth "
                f"{lead[0]:.1f} points over a short one ({w('decision', 'lead').per_sd:+.2f} per SD) and sprint speed "
                f"{speed[0]:.1f} ({w('decision', 'speed').per_sd:+.2f}). Two balls in the count cost {abs(balls[0]):.1f} "
                "points, consistent with pitchers throwing fastballs, the quickest pitch to the plate, when they are behind. A "
                f"right-handed batter adds {rhb[0]:.1f} points, for reasons these data do not show. Left-handed pitchers look easier "
                f"to run on ({pts(lhp[0])}), but this is selection: teams attempt {lr:.2f} steals per 100 pitches against "
                f"left-handers and {rr:.2f} against right-handers, so the attempts they do make are the ones they like."),
           para("Given the jump and the pitch (Figure 5B), ground gained dwarfs everything: one standard deviation "
                f"({w('success', 'gain').sd:.1f} feet) multiplies the odds of success by {np.exp(w('success', 'gain').per_sd):.1f}. A "
                f"breaking ball multiplies them by {np.exp(w('success', 'brk').per_sd):.1f} and an offspeed pitch by "
                f"{np.exp(w('success', 'off').per_sd):.1f} against a fastball. Sprint speed's weight rises from "
                f"{w('decision', 'speed').per_sd:+.2f} to {w('success', 'speed').per_sd:+.2f} per SD, because faster runners gain less "
                f"ground before release (r = {r_ga:+.2f} across attempts): once the ground is held fixed, the direct benefit of "
                "speed appears. Appendix B gives every estimate."),
           figure("Fig_paper_influence.png", 5, "<b>What moves the odds.</b> Change in the log-odds of a successful steal for a "
                  "one-standard-deviation increase in each input (indicators: 0 to 1), with 95% confidence interval: (A) the "
                  "decision model, inputs known before the pitch; (B) the success model, which adds the inputs measured during the "
                  "delivery. Ground gained is shown at its average; its squared term makes it steeper above average. Filled: steals "
                  "of second; open: steals of third."),
           para("4.5 What it takes to be safe", H2),
           para("The success model turns the break-even rate into a target a runner can train for: the ground that must be covered "
                "by the release of the pitch for the attempt to pay (Figure 6). For a steal of second with one out against an "
                f"average catcher the target is {gr[('2B', 1, 'average')].ground_needed_ft:.1f} ft; against a quick catcher (10th "
                f"percentile pop time) it rises to {gr[('2B', 1, 'quick')].ground_needed_ft:.1f} ft and against a slow one it falls "
                f"to {gr[('2B', 1, 'slow')].ground_needed_ft:.1f} ft. In 2023–26, {gr[('2B', 1, 'average')].share_reaching:.0%} of "
                f"steals of second reached the average-catcher target but only {gr[('2B', 1, 'quick')].share_reaching:.0%} reached "
                f"the quick-catcher one. A steal of third needs {gr[('3B', 1, 'average')].ground_needed_ft:.1f} ft with one out and "
                f"{gr[('3B', 2, 'average')].ground_needed_ft:.1f} ft with two."),
           figure("Fig_paper_ground.png", 6, "<b>What it takes to be safe.</b> Probability of success from the success model against "
                  "the ground gained between the pitcher's first move and release (drawn over its 5th to 95th percentile), steals of "
                  "second (A) and third (B) with one out, for an average runner, pitcher, count and pitch mix against a "
                  "right-handed pitcher: average catcher (blue), quick catcher (10th percentile pop time, grey dashed) and slow "
                  "catcher (90th percentile, grey dotted). Black dashed line: the one-out break-even rate; the marked point is the "
                  "ground needed against an average catcher.")]

    st += [para("4.6 Speed is significant, and small, every season", H2),
           para(f"On steals of second a speed-only logistic regression gives a positive slope in every season, significant at the 5% "
                f"level in {int((sp.p < 0.05).sum())} of {len(sp)} (Figure 7B). Significance reflects the sample size, not the size of "
                f"the effect: from the 10th to the 90th percentile of speed (about {sp.speed_p10.mean():.1f} to {sp.speed_p90.mean():.1f} "
                f"ft/s) predicted success rises by {sp.swing_pts.min():.1f}–{sp.swing_pts.max():.1f} points, and speed alone ranks "
                f"attempts barely better than a coin flip (AUROC {sp.auroc.min():.2f}–{sp.auroc.max():.2f}). Figure 7A and Table 3 "
                "show why the slope is precise although flat: fast runners make most of the attempts."),
           figure("Fig_paper_speed.png", 7, "<b>Speed is significant, and small, every season.</b> Steals of second, 2023–2026. (A) "
                  "Attempts per 0.5 ft/s of sprint speed, stolen bases (blue) and caught stealing (orange) stacked. (B) Success rate "
                  "per bin with 95% Wilson interval (open circles: fewer than 20 attempts) and a logistic regression on speed alone "
                  "with its 95% band (runner-clustered; a straight line on the log-odds scale). Both panels pool every attempt in a "
                  "bin; runner by runner, see Figure 8. Dashed: the break-even rate.")]
    cols = [c for c in B.columns if c.startswith("sb_")]
    rows = [["Sprint speed (ft/s)"] + [c[3:] for c in cols] + ["2023–26", "Success", "Runner-<br/>seasons", "Attempts per runner-season"]]
    for _, r in B.iterrows():
        rows.append([r["bin"]] + [f"{int(r[c])}–{int(r['cs_' + c[3:]])}" for c in cols]
                    + [f"{int(r.sb):,}–{int(r.cs):,}", pct(r.success, 1), f"{int(r.runner_seasons):,}", f"{r.attempts_per_runner_season:.1f}"])
    tsb, tcs, trs = int(B.sb.sum()), int(B.cs.sum()), int(B.runner_seasons.sum())
    rows.append(["All"] + [f"{int(B[c].sum()):,}–{int(B['cs_' + c[3:]].sum()):,}" for c in cols]
                + [f"{tsb:,}–{tcs:,}", pct(tsb / (tsb + tcs), 1), f"{trs:,}", f"{(tsb + tcs) / trs:.1f}"])
    st.append(table(rows, [0.8, 0.68, 0.68, 0.68, 0.68, 0.85, 0.55, 0.65, 0.93], caption=(3, "Steals of second by the runner's sprint "
                    "speed, 2023–26: stolen bases–caught stealing by season."),
                    note="Sprint speed: the runner's Statcast sprint speed that season, in 0.5 ft/s bins with the sparse tails merged. "
                         "A runner-season is one runner in one season; attempts per runner-season count his tracked steals of second.",
                    bold_last=True))
    mid = B.set_index("bin").loc[["26.5–26.9", "27.0–27.4", "27.5–27.9"]].success
    st += [para(f"Fast runners run more: runner-seasons at 30 ft/s and up averaged {B.iloc[-1].attempts_per_runner_season:.1f} "
                f"attempts of second, against {B.iloc[:3].attempts_per_runner_season.min():.1f}–"
                f"{B.iloc[:3].attempts_per_runner_season.max():.1f} below 26.5 ft/s (Table 3). Pooled over every attempt, success "
                f"also rises with speed, from {pct(mid.min())}–{pct(mid.max())} between 26.5 and 27.9 ft/s to "
                f"{pct(B.iloc[-1].success)} at 30 and up. But among regular base stealers speed barely separates one runner from "
                f"another: across the {int(R.runner_seasons)} runner-seasons with ten or more attempts, success rate and sprint "
                f"speed are nearly uncorrelated (r = {R.r:+.2f}; {R.r_without_naylor:+.2f} without Naylor's two seasons; Figure 8). "
                "A season's success rate over 10 to 60 attempts is noisy, but noise does not explain the result: after removing "
                f"the binomial sampling noise, these runner-seasons truly differ by about {100 * R.true_sd:.0f} points of success "
                f"(SD), and speed accounts for about {100 * R.r_true ** 2:.0f}% of that spread ({100 * R.r_true_without_naylor ** 2:.0f}% "
                "without Naylor). Naylor sits at the far left of Figure 8 and near "
                f"its top: {int(ny['n25'].safe)} of {int(ny['n25'].n)} in 2025 and {int(ny['n26'].safe)} of {int(ny['n26'].n)} in "
                f"2026, when his {ny['n26'].gain:.1f} feet gained per attempt was the most of any runner-season."),
           figure("Fig_paper_runners.png", 8, "<b>Fast runners run more, not better.</b> Each dot is one runner's season of steals "
                  "of second (10 or more attempts), success rate against his sprint speed that season; dot area shows attempts. "
                  "Line: logistic regression on speed alone, every attempt 2023–26. Dashed: the break-even rate."),
           para("4.7 Who controls the running game", H2),
           para("The random intercepts measure how much players differ once the measured inputs are accounted for (Figure 9). Before "
                f"the pitch, on steals of second, a runner one standard deviation better than average adds {v2r.pp_plus_1sd:.1f} "
                f"points, a pitcher {v2p.pp_plus_1sd:.1f} and a catcher {v2c.pp_plus_1sd:.1f}: the runner's is the largest single "
                f"effect, and the pitcher and catcher together matter about as much (combined SD {np.hypot(v2p.sigma, v2c.sigma):.3f} "
                f"against {v2r.sigma:.3f} for runners). The success model shows where the pitcher's influence comes from. Once the "
                f"ground gained and the pitch are known, the pitcher SD falls from {v2p.sigma:.2f} to {s2p.sigma:.2f} on the log-odds "
                f"scale (95% CI {s2p.lo:.2f} to {s2p.hi:.2f}), a fall of {1 - (s2p.sigma / v2p.sigma) ** 2:.0%} in the estimated "
                "pitcher variance: most of a pitcher's influence on steals of second runs through the ground he allows and the "
                f"pitch he throws. The runner SD {'rises' if s2r.sigma > v2r.sigma else 'falls'} from {v2r.sigma:.2f} to "
                f"{s2r.sigma:.2f}: with the jump held fixed, runners still differ. On steals of third the point estimates favour "
                f"the battery, pitchers ({pts(v3p.pp_plus_1sd)}) and catchers ({pts(v3c.pp_plus_1sd)}) against runners "
                f"({pts(v3r.pp_plus_1sd)}), but with {n3:,} attempts spread over {int(v3p.levels):,} pitchers, {int(v3c.levels):,} "
                f"catchers and {int(v3r.levels):,} runners, every 95% interval reaches zero in both models (Appendix Table B3): no "
                "group's differences on steals of third can yet be told apart from chance."),
           figure("Fig_paper_variance.png", 9, "<b>Who controls the running game.</b> Gain in the probability of success from a "
                  "player one standard deviation above average, with 95% likelihood-ratio interval: (A) before the pitch, decision "
                  "model; (B) given the ground gained and the pitch, success model. Filled: steals of second; open: steals of third."),
           para("4.8 The calls, scored on 2026", H2),
           para(f"Of {int(g2.n_test):,} steals of second in 2026, the decision model would have held {hold.n:.0f} "
                f"({hold.n / g2.n_test:.0%}), those it scored below the break-even rate of their state. They succeeded "
                f"{pct(hold.success, 1)} of the time against a bar of {pct(hold.mean_be, 1)} and cost "
                f"{abs(hold.runs_per_attempt):.3f} runs each, {abs(hold.total_runs):.1f} runs in all (Figure 10A). The attempts it "
                f"would have sent gained {go.runs_per_attempt:.3f} runs each, {go.total_runs:.1f} in all, and the further the "
                f"model's probability cleared the bar the more an attempt paid: {top.runs_per_attempt:+.3f} runs per attempt when "
                f"it cleared it by more than ten points ({top.n:,.0f} attempts). On steals of third the calls are less informative "
                "(Figure 10B): the samples are small, and only the attempts scored more than ten points above the bar clearly paid "
                f"({top3.runs_per_attempt:+.2f} runs each, {top3.n:.0f} attempts)."),
           figure("Fig_paper_decision_value.png", 10, "<b>The calls, scored on 2026.</b> Runs gained per attempt (RE24 after the "
                  "outcome minus before, with 95% interval) by how far the decision model's probability, fit on 2023–25, cleared "
                  "the break-even rate of each attempt's state. Orange: attempts the model would have held; blue: attempts it would "
                  "have sent."),
           para("4.9 Who to send in 2027", H2),
           para("Applying the 2023–26 decision model to every runner with ten or more attempts of second in 2025–26 gives the "
                "green-light list for 2027 (Figure 11). Each runner's predicted success with one out is set against the one-out "
                f"break-even rate, {pct(be1, 1)}, twice: against an average pitcher and catcher, and against a pitcher and catcher "
                "each one standard deviation harder to run on. That sorts the runners into three calls (Table 4). Of the "
                f"{len(G)}, {tough} clear the bar even against the tough battery: a standing green light. Another {clear - tough} "
                "clear it only against an average battery: their green light depends on who is pitching and catching. The other "
                f"{len(G) - clear}, mostly slower runners with short leads, fall short even against an average battery. Table 5 lists the pitchers and "
                "catchers who move the bar most."),
           figure("Fig_paper_greenlight.png", 11, "<b>Who to send in 2027.</b> Predicted success on a steal of second against an "
                  "average pitcher and catcher with one out, from the 2023–26 decision model, for every runner with ten or more "
                  "attempts of second in 2025–26, against sprint speed; marker size shows attempts. Dashed: the one-out break-even "
                  "rate.")]
    green, dep, red = G[G.p_tough_1out >= be1], G[(G.p_1out >= be1) & (G.p_tough_1out < be1)], G[G.p_1out < be1]
    ex = lambda d: "; ".join(f"{r_['name']} ({r_.attempts_2025_26:.0f}: {100 * (r_.p_1out - be1):+.1f} / {100 * (r_.p_tough_1out - be1):+.1f})"
                             for _, r_ in d.sort_values("attempts_2025_26", ascending=False).head(4).iterrows())
    rows = [["Call", "Rule", "Runners", "Most active runners (attempts 2025–26: points above the bar against an average / a "
             "tough battery)"],
            ["<b>Green light</b>", "Clears the bar even against a tough battery", f"{len(green)}", ex(green)],
            ["<b>Battery-dependent</b>", "Clears it against an average battery only", f"{len(dep)}", ex(dep)],
            ["<b>Hold</b>", "Falls short even against an average battery", f"{len(red)}", ex(red)]]
    st.append(table(rows, [1.2, 1.35, 0.62, 3.33], caption=(4, f"The 2027 green-light list for steals of second: three calls, with "
                    f"one out (break-even {pct(be1, 1)})."),
                    note="Points above the bar: the decision model's predicted probability of success, with the runner's latest sprint "
                         f"speed, his typical lead and his random intercept, minus {pct(be1, 1)}. Average battery: a league-average "
                         "pitcher and catcher; tough battery: each one "
                         "standard deviation harder to run on (about the 84th percentile). The full list is "
                         "results/1-Paper-GLMM/DF_paper_greenlight.csv."))
    P = D.PL[(D.PL.base == "2B") & (D.PL.model == "decision")]
    a = D.rows["2B"]; bpop = D.fx("2B", "pop").beta; b0 = D.fx("2B", "intercept").beta
    cpop = a.groupby("catcher_id")["pop"].mean()
    cat = P[(P.group == "catcher") & (P.attempts >= 50)].copy()
    cat["total"] = 100 * (IQ.expit(b0 + bpop * cat.id.map(cpop) + cat.effect) - IQ.expit(b0))
    pit = P[(P.group == "pitcher") & (P.attempts >= 30)].copy()
    rows, groups = [["Easiest to run on", "Attempts", "Pts", "Hardest to run on", "Attempts", "Pts"]], []
    for lab, d, col in [("Pitchers, 30+ attempts against (random intercept)", pit, "pp_vs_avg"),
                        ("Catchers, 50+ attempts against (pop time plus random intercept)", cat, "total")]:
        groups.append(len(rows)); rows.append([lab, "", "", "", "", ""])
        for (_, x), (_, y) in zip(d.nlargest(5, col).iterrows(), d.nsmallest(5, col).iterrows()):
            rows.append([x["name"], f"{x.attempts:.0f}", pts(x[col]), y["name"], f"{y.attempts:.0f}", pts(y[col])])
    st.append(table(rows, [1.9, 0.6, 0.5, 1.9, 0.6, 0.5], groups=groups, caption=(5, "Batteries on steals of second, 2023–26: change in "
                    "the probability of success against an average runner, in points."),
                    note="Shrunk estimates from the decision model."))

    st += [para("5. Discussion", H1),
           para("Each finding answers a question a front office asks before a series. The bullets give the bottom line of each "
                "figure and what to do with it; the companion document gives each figure with its full evidence.")]
    for f in F:
        st += bullets([f"<b>Figure {f['n']}, {f['title'].lower()}.</b> {f['bluf']} <i>Use:</i> {f['use']}"])
    st += [para("Taken together, the results separate two jobs that are often run together. Predicting whether a steal succeeds "
                "is largely a matter of measuring the jump: with the ground gained and the pitch in hand, a mixed model with player "
                f"effects reaches AUROC {s.auroc:.2f} a season ahead, better than the pooled and published models tested here. "
                "Deciding whether to run is a matter of the matchup: the lead the runner can take, the catcher behind the plate, the "
                "pitcher's time to the plate and the count, set against a bar that differs by base and by outs. A decision model "
                "built only on those pre-pitch facts is calibrated a season ahead, and the attempts it would have held in 2026 were "
                "the ones that lost runs.")]

    st += [para("6. Limitations", H1)] + bullets([
        "The success model uses the ground gained and the pitch, which arrive after the runner commits. It is for evaluating "
        "attempts and players and for setting targets; using it as a pre-pitch forecast would overstate what can be known.",
        "Only attempts that were made are observed. The value of the calls is measured among attempts teams chose; extending the "
        "model to every pitch with a runner on base needs each runner's lead on every pitch, which the public data do not give.",
        "Disengagements and pickoff attempts, the focus of Powers et al. (2026), are not yet inputs; they shape the lead the "
        "runner can take and belong in the next version.",
        "Sprint speed and pop time are season averages, and the lead at the first move is partly the pitcher's doing (the hold) "
        "as well as the runner's choice.",
        "The bar is set by run expectancy. Late in close games the break-even rate should come from win expectancy instead. RE24 "
        f"is computed from the {games / D.games_played:.0%} of games with a tracked attempt and has not been checked against "
        "every game.",
        "Steals of third are fewer than a fifth of attempts; no group's player effects there can yet be distinguished from zero, "
        "and the value of the calls is uncertain.",
        "Maximum-likelihood variance components are slightly low for groups with few observations each (Appendix C), so the "
        "spread among pitchers and catchers is, if anything, understated."])
    me = ny["gl"]
    st += [para("7. Conclusion", H1),
           para("A stolen base pays when its chance of success clears the break-even rate of its situation, and that chance is "
                "decided mostly in the second between the pitcher's first move and his release. Measured there, with each runner, "
                f"pitcher and catcher given his own effect, the outcome of a steal can be ranked a season ahead at AUROC "
                f"{s.auroc:.2f}, better than the pooled and published models tested here. Before the pitch the same question can "
                f"be answered only to {dc.auroc:.2f}, but honestly enough to separate the 2026 attempts that gained runs from those "
                "that lost them. Josh Naylor is the paper in one player: the slowest legs on the list, the most ground gained, and "
                "a decision model that cannot see his jump before the pitch but has learned it, rating him "
                f"{ordinal(ny['effect_rank'])} of {ny['gl_n']} runners. With his speed and his lead it still puts him at only "
                f"{pct(me.p_1out, 1)} with one out against an average battery ({pct(me.p_tough_1out, 1)} against a tough one): a "
                "green light that depends on who is pitching and catching."),
           para("References", H1)]
    st += [para(r, REF) for r in [
        "42 Analytics. The Pitcher's Dilemma (stolen-base success model).",
        "BaseballCV: computer vision models for baseball broadcast video. github.com/dylandru/BaseballCV.",
        "Bates, D., Mächler, M., Bolker, B., &amp; Walker, S. (2015). Fitting linear mixed-effects models using lme4. "
        "<i>Journal of Statistical Software</i>, 67(1), 1–48.",
        "Jiang, T., Lu, P., Zhang, L., et al. (2023). RTMPose: real-time multi-person pose estimation based on MMPose. arXiv:2303.07399.",
        "MLB Advanced Media. Baseball Savant: basestealing run value and running game leaderboards, sprint speed, catcher pop "
        "time and Statcast search. baseballsavant.mlb.com.",
        "MLB Advanced Media. Statcast glossary: lead distance, pop time, sprint speed. mlb.com/glossary/statcast.",
        "Powers, S., et al. (2026). Strategies under a pickoff limit in baseball: a zero-sum sequential game with multilevel models. "
        "arXiv:2601.15608.",
        "Tango, T. M., Lichtman, M. G., &amp; Dolphin, A. E. (2007). <i>The Book: Playing the Percentages in Baseball</i>. Potomac Books."]]
    st.append(PageBreak())
    return st + appendix(D)


def RE_(D, state, outs):
    """RE24 for one base-out state, 2023-26 (Appendix Table A1)."""
    return IQ.re24(range(2023, 2027))[(state, outs)]


def two_out(obs, bar):
    """How the observed two-out success rate stands against its bar, in words."""
    d = 100 * (obs - bar)
    return (f"but the two-out bar by only {d:.0f} point{'' if round(d) == 1 else 's'}" if d >= 0.5 else
            "but only matched the two-out bar" if d > -0.5 else f"but fell {abs(d):.0f} points short of the two-out bar")


def ordinal(k):
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(k, f"{k}th")


def appendix(D):
    st = [para("Appendix A. Run expectancy and break-even rates", H1)]
    RE = IQ.re24(range(2023, 2027)); n = D.RE.groupby(["state", "outs"]).n.sum()
    order = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
    name = {"___": "Bases empty", "1__": "First", "_2_": "Second", "__3": "Third", "12_": "First and second",
            "1_3": "First and third", "_23": "Second and third", "123": "Loaded"}
    rows = [["Runners on", "0 outs", "1 out", "2 outs", "Plate appearances"]]
    for s in order:
        rows.append([name[s]] + [f"{RE[(s, o)]:.3f}" for o in range(3)] + [f"{sum(n[(s, o)] for o in range(3)):,}"])
    st.append(table(rows, [1.6, 0.9, 0.9, 0.9, 1.3], caption=("A1", "Run expectancy (RE24) to the end of the half-inning, 2023–26."),
                    note="Runs scored from the start of each plate appearance to the end of its half-inning, innings 1–8 (later "
                         "innings can end on a walk-off), Statcast pitch data of every game with a tracked attempt."))
    rows = [["Attempt", "Outs", "RE now", "RE if safe", "RE if caught", "Break-even"]]
    for _, r in D.BE.iterrows():
        rows.append(["Steal of second (first only)" if r.base == "2B" else "Steal of third (second only)", str(r.outs),
                     f"{r.re_now:.3f}", f"{r.re_sb:.3f}", f"{r.re_cs:.3f}", pct(r.breakeven, 1)])
    st.append(table(rows, [2.1, 0.5, 0.8, 0.9, 0.9, 0.9], caption=("A2", "Break-even success rates (equation 1).")))

    st += [para("Appendix B. Model estimates", H1)]
    unit = {"speed": "sprint speed (ft/s)", "lead": "lead at first move (ft)", "pop": "catcher pop time (0.1 s)", "balls": "balls (each)",
            "strikes": "strikes (each)", "out1": "one out (vs none)", "out2": "two outs (vs none)", "lhp": "left-handed pitcher",
            "rhb": "right-handed batter", "gain": "ground gained to release (ft)", "gain2": "ground gained squared ÷ 10",
            "brk": "breaking ball (vs fastball)", "off": "offspeed pitch (vs fastball)"}
    rows = [["Input (unit)", "2nd: β (SE)", "2nd: change, pts (95% CI)", "3rd: β (SE)", "3rd: change, pts (95% CI)"]]
    terms = [t for t in D.SF[D.SF.base == "2B"].term if t != "intercept"]
    for t in ["gain", "gain2", "brk", "off"] + [t for t in terms if t not in ("gain", "gain2", "brk", "off")]:
        cells = [unit[t]]
        for b in ("2B", "3B"):
            r = D.sx(b, t)
            if t == "gain2":
                cells += [f"{r.beta:+.3f} ({r.se:.3f})", "(curvature)"]
            else:
                m_, lo_, hi_ = D.sswing(b, t)
                cells += [f"{r.beta:+.3f} ({r.se:.3f})", f"{m_:+.1f}" + ("" if t == "gain" else f" ({lo_:+.1f} to {hi_:+.1f})")]
        rows.append(cells)
    st.append(table(rows, [1.75, 0.95, 1.4, 0.95, 1.4], caption=("B1", "Success model estimates by base (2023–26)."),
                    note="Log-odds coefficients with standard errors. Changes in points: probability at the 90th minus the 10th percentile "
                         "of a continuous input (others at their averages), or 0 to 1 for an indicator; the ground-gained change includes "
                         "the squared term. Pop time is to second for steals of second and to third for steals of third. Steals of second "
                         f"n = {len(D.srows['2B']):,}; steals of third n = {len(D.srows['3B']):,}."))
    rows = [["Input (unit)", "2nd: β (SE)", "2nd: change, pts (95% CI)", "3rd: β (SE)", "3rd: change, pts (95% CI)"]]
    for t in IQ.DEC_FIXED:
        cells = [unit[t]]
        for b in ("2B", "3B"):
            r = D.fx(b, t); m, lo, hi = D.swing(b, t)
            cells += [f"{r.beta:+.3f} ({r.se:.3f})", f"{m:+.1f} ({lo:+.1f} to {hi:+.1f})"]
        rows.append(cells)
    st.append(table(rows, [1.75, 0.95, 1.4, 0.95, 1.4], caption=("B2", "Decision model estimates by base (2023–26)."),
                    note="Same conventions as Table B1. Steals of second n = "
                         f"{len(D.rows['2B']):,}; steals of third n = {len(D.rows['3B']):,}."))
    rows = [["Model", "Group", "2nd: SD (95% CI)", "2nd: pts/SD", "3rd: SD (95% CI)", "3rd: pts/SD"]]
    fmt = lambda r: f"{r.sigma:.3f} ({r.lo:.3f}–{r.hi:.3f})"
    for model, get in [("Decision", lambda b, g: D.vc(b, g)), ("Success", lambda b, g: D.sv(b, g))]:
        for g in ("runner", "pitcher", "catcher"):
            r2, r3 = get("2B", g), get("3B", g)
            rows.append([model, f"{g}s ({int(r2.levels):,} / {int(r3.levels):,})", fmt(r2), pts(r2.pp_plus_1sd), fmt(r3), pts(r3.pp_plus_1sd)])
    st.append(table(rows, [0.75, 1.55, 1.25, 0.85, 1.25, 0.85], caption=("B3", "Random-intercept standard deviations (log-odds scale)."),
                    note="95% likelihood-ratio intervals (each SD varied with the others at their estimates). Players in parentheses: "
                         "second / third."))

    st += [para("Appendix C. Checking and tuning the models", H1),
           para("The fitting code was checked in two ways before use. First, 20 data sets were simulated from the fitted decision "
                "model for steals of second, with the same players and inputs and fresh random effects, and each was refit. Second, "
                "the same model was fit to the real data by an independent method, statsmodels' variational Bayes. Table C1 "
                "compares them. A fixed effect is recovered when its mean estimate lies within about two Monte Carlo standard errors "
                "of the truth; the variance components show the downward bias expected of maximum likelihood with many small groups "
                "(the pitchers average under ten attempts each).")]
    lab = {"intercept": "intercept", **{k: IQ.TERM_LABEL[k] for k in IQ.DEC_FIXED},
           "sigma runner": "SD, runner intercepts", "sigma pitcher": "SD, pitcher intercepts", "sigma catcher": "SD, catcher intercepts"}
    rows = [["Parameter", "Truth (fit)", "Mean of 20 refits", "SD of refits", "Bias (MC s.e.)", "Variational Bayes"]]
    for _, r in D.CK.iterrows():
        rows.append([lab[r.param], f"{r.truth:+.4f}", f"{r.mean_est:+.4f}", f"{r.sd_est:.4f}", f"{r.bias_in_sd:+.2f}", f"{r.statsmodels_vb:+.4f}"])
    st.append(table(rows, [1.95, 0.85, 1.05, 0.85, 0.85, 0.95], caption=("C1", "Recovery and cross-check, decision model for steals of second.")))
    rows = [["Specification (success model)", "2nd: log-loss", "2nd: AUROC", "3rd: log-loss", "3rd: AUROC", "Pooled log-loss", "Chosen"]]
    for spec in IQ.SUCCESS_SPECS:
        r2 = D.SCV[(D.SCV.spec == spec) & (D.SCV.base == "2B")].iloc[0]; r3 = D.SCV[(D.SCV.spec == spec) & (D.SCV.base == "3B")].iloc[0]
        rows.append([spec, f"{r2.log_loss:.4f}", f"{r2.auroc:.4f}", f"{r3.log_loss:.4f}", f"{r3.auroc:.4f}", f"{r2.pooled_log_loss:.4f}",
                     "yes" if r2.chosen else ""])
    st.append(table(rows, [2.0, 0.75, 0.75, 0.75, 0.75, 0.85, 0.6], caption=("C2", "Choosing the success model: five-fold cross-validation "
                    "on 2023–25 only."), note="The specification with the lowest pooled log-loss was chosen before 2026 was scored."))
    pc3, pc2 = D.PC.loc["pop time to third"], D.PC.loc["pop time to second"]
    rows = [["Pop time in the steal-of-third decision model", "Attempts", "Deviance", "β per 0.1 s (SE)"],
            ["To third (used)", f"{int(pc3.n):,}", f"{pc3.deviance:.1f}", f"{pc3.beta:+.3f} ({pc3.se:.3f})"],
            ["To second", f"{int(pc2.n):,}", f"{pc2.deviance:.1f}", f"{pc2.beta:+.3f} ({pc2.se:.3f})"]]
    st.append(table(rows, [3.0, 0.9, 0.9, 1.3], caption=("C3", "Which pop time to use for steals of third."),
                    note=f"Same attempts (those with both measures) and the same number of parameters, so the lower deviance fits better: "
                         f"pop time to third by {pc2.deviance - pc3.deviance:.1f}."))

    timed = {s: ((d.qa == "PASS").sum(), len(d)) for s, d in D.cv.items()}
    T = D.JT.set_index(["test", "item"])
    st += [para("Appendix D. Delivery time from video", H1),
           para("Section 3.5 describes the measurement. Coverage by season: " + "; ".join(f"{s}: {k:,} of {n:,} ({k / n:.0%})" for s, (k, n) in timed.items()) + ". "
                "Splitting ground gained into the runner's jump speed and the pitcher's time shows the runner supplies about "
                f"{100 * T.loc[('decomposition', 'runner jump speed share of log ground-gained variance'), 'value']:.0f}% of the "
                "variance in ground gained; jump speed is a stable runner trait (split-half reliability "
                f"{T.loc[('split_half_reliability', 'runner jump speed'), 'value']:.2f}). The timing offset between Statcast's first "
                f"move and the video's lift-off is {T.loc[('setup', 'first-move to lift-off offset delta (s)'), 'value']:+.3f} s, "
                f"statistically zero. Pitchers' random intercepts in the decision model partly reflect their measured delivery "
                f"times (r = {D.pitcher_time_r[0]:+.2f} for the {D.pitcher_time_r[1]} pitchers with ten or more timed clips: slower "
                "deliveries are easier to run on); the rest is the hold, the pickoff move and noise."),
           para("Appendix E. Reproduce", H1),
           para("From the repository root: <font face='Courier'>python3 1-Data-Ingestion/ingest.py meta</font> (the per-season tables), "
                "<font face='Courier'>python3 2-Data-Analysis/stealiq.py decide success popchoice paperfigs</font> (models, the "
                "forward test and figures, about 7 minutes on an idle machine), <font face='Courier'>python3 2-Data-Analysis/"
                "stealiq.py glmmcheck</font> (Table C1, up to an hour), and <font face='Courier'>python3 2-Data-Analysis/build_paper.py</font> (this paper, "
                "the figure discussion and CATALOG.md). Results and figures for this paper are in <i>results/1-Paper-GLMM</i> and "
                "<i>figures/1-Paper-GLMM</i>.")]
    return st


# ── the companion: one page per figure, with BLUF and XYZ bullets ─────────────────────────────────────────────────
def discussion(D):
    st = [Spacer(1, 10), para("StealIQ: When to Steal — figure by figure", TITLE),
          para("A discussion section for the paper, one figure per page, numbered as in the paper. Each figure has its bottom line "
               "(BLUF) and three bullets in Google's XYZ form: what is now true (X), the evidence that shows it (Y) and how it was "
               "shown (Z), followed by what a front office does with it.", SUB)]
    for f in findings(D):
        path = fig_path(f["fig"]); w_, h = ImageReader(str(path)).getSize()
        width = min(WIDTH, 5.4 * w_ / h)                  # tall figures shrink so the bullets stay on the page
        st += [para(f"Figure {f['n']}. {f['title']}", H1), Image(str(path), width=width * inch, height=width * inch * h / w_), Spacer(1, 6),
               para(f"<b>BLUF.</b> {f['bluf']}")]
        st += bullets([f"<b>X — what is true.</b> {f['x']}", f"<b>Y — the evidence.</b> {f['y']}", f"<b>Z — how it was shown.</b> {f['z']}",
                       f"<b>Front office.</b> {f['use']}"])
        st.append(PageBreak())
    return st[:-1]                                         # no blank page after the last figure


def build():
    D = Data()

    def footer(c, doc):
        c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
        c.drawString(1.0 * inch, 0.55 * inch, "StealIQ: When to Steal"); c.drawRightString(7.5 * inch, 0.55 * inch, str(doc.page))
        c.restoreState()
    for out, story, title in [(PAPER, paper(D), "StealIQ: When to Steal"), (DISCUSSION, discussion(D), "StealIQ: figure by figure")]:
        SimpleDocTemplate(str(out), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.85 * inch,
                          bottomMargin=0.85 * inch, title=title, author="StealIQ").build(story, onFirstPage=footer, onLaterPages=footer)
        print(f"wrote {out}")


if __name__ == "__main__":
    build()
    import build_catalog
    build_catalog.main()
