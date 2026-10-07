#!/usr/bin/env python3
"""
stealiq.py — StealIQ's command line, and one import for everything else (`import stealiq as IQ` in build_paper.py,
build_catalog.py and the notebook). The code lives in four modules:

  core.py                   paths, output folders, loaders, shared models and features
  paper.py                  the paper: RE24, break-even, per-base GLMMs, forward test, calls, 2027 list
  research.py               the archived technical report's studies (v12/v13 engines, Powers et al., pre-pitch, v16, ...)
  ../5-Live-Webapp/webapp.py   the live site: season metrics, leaderboard, calculator

Usage:  python3 stealiq.py all                 every step, in order (decide ~2 min, success ~4 min, glmmcheck up to 1 h)
        python3 stealiq.py decide paperfigs    any subset of the steps below
        python3 stealiq.py v16 60 200          one step with its arguments (v16: outer / final Optuna trials;
                                              powers: bootstrap resamples)

  site       v15 (everything below, then calc) calc (the odds calculator only) .. 5-Live-Webapp/webapp.py
  research   v12 v13 powers prepitch needle context algo v16 v16checks baselines speed jump prior ... research.py
  paper      decide success successfit popchoice paperfigs glmmcheck ................................ paper.py
"""
import sys, time
from pathlib import Path
from core import *  # noqa: F401,F403
from research import *  # noqa: F401,F403
from paper import *  # noqa: F401,F403
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "5-Live-Webapp"))
from webapp import *  # noqa: E402,F401,F403


# ════ CLI ═══════════════════════════════════════════════════════════════════════════════════════════════════════
STEPS = {"v15": run_v15, "calc": run_calculator, "v12": run_v12, "v13": run_v13, "powers": run_powers, "prepitch": run_prepitch,
         "needle": run_needle, "context": run_context, "algo": run_algo, "v16": run_v16, "v16checks": run_v16_checks,
         "baselines": run_baselines, "speed": run_speed, "jump": run_jump, "prior": run_prior, "decide": run_decide,
         "success": run_success, "successfit": success_fit, "popchoice": pop_choice, "paperfigs": run_paperfigs,
         "glmmcheck": glmm_check}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = ([n for n in STEPS if n not in ("successfit", "calc")] + ["calc"]   # success runs the fit; calc reruns last, on
             if args == ["all"] else [a for a in args if a in STEPS])        # the paper results this run wrote
    nums = [int(a) for a in args if a.isdigit()]
    bad = [a for a in args if a not in STEPS and not a.isdigit() and a != "all"]
    if bad or not names:
        sys.exit(__doc__)
    for name in names:
        t0 = time.time(); print(f"\n══ {name} ══", flush=True)
        STEPS[name](*nums) if len(names) == 1 and nums else STEPS[name]()
        print(f"══ {name} done in {time.time() - t0:.0f} s", flush=True)
