#!/usr/bin/env python3
"""
ingest.py — every byte of data in this project enters here: scrape -> raw tables -> one meta table per season.

    scrape (network, cached, resumable)            build (offline)          meta (offline)
    discover / leads    Savant steal drawer    -+
    sprint / poptime / league   season tables   +-> data/raw/*.csv  ---->  data/meta/meta_<season>.csv
    context / opportunities     MLB feed        |                          + data_dictionary.csv
    statcast / people / docs    Savant, MLB    -+   (vision/delivery.py adds data/delivery/delivery_<season>.csv)

Subcommands, in run order (python3 ingest.py <cmd> -h for options):
  discover --start --end [--expand]  rank base-stealers (StatsAPI SB/CS + Savant sprint speed); --expand pulls leads
  leads RUNNER YEAR                  one runner-season's per-attempt drawer (lead distances, result)
  sprint | poptime | league          season tables: sprint speed, catcher pop time / arm, league SB/CS per season
  context                            MLB feed: hand, pitch code, count, outs, inning for every attempt pitch
  opportunities                      MLB feed: every pitch with a runner on 1B and 2B empty (the decision-model denominator)
  statcast [--stop-at HH:MM]         Savant: every Statcast column for every attempt pitch
  people                             MLB names for every runner and batter id
  docs                               Savant's Statcast CSV field definitions (feeds the data dictionary)
  assets                             headshots + team map for the site (docs/assets)
  build [--refresh 2026]             Raw_Attempts.csv + Raw_Season.csv from the caches
  meta                               the per-season meta tables: one row per tracked attempt, every field, the video link

A new season: discover --expand -> sprint / poptime / league -> context -> statcast -> people -> build -> meta.
Why two sources: Savant's basestealing leaderboard export ignores its year parameter, so year-correct SB/CS come from
StatsAPI; Savant is used where it IS year-correct (sprint speed, pop time, the per-attempt drawer, Statcast search).
The caches in data/cache/ are gitignored and regenerable; data/raw/ and data/meta/ are committed, so the meta tables
rebuild offline from a fresh clone.
"""
from __future__ import annotations
import argparse, csv, gzip, html, io, json, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent                      # 1-Data-Ingestion/
DATA = ROOT / "data"
RAW, CACHE, META, DELIVERY = DATA / "raw", DATA / "cache", DATA / "meta", DATA / "delivery"
SITE_ASSETS = ROOT.parent / "docs" / "assets"               # the served copy (GitHub Pages)

LEADS_DIR, DISC_DIR = CACHE / "leads_cache", CACHE / "discovery"
PBP_DIR, OPPS_DIR = CACHE / "pbp_cache", CACHE / "pbp_opps_cache"
SC_DIR, PI_DIR = CACHE / "statcast", CACHE / "pitch_index"
HEADSHOTS, LOGOS = SITE_ASSETS / "headshots", SITE_ASSETS / "logos"

ATTEMPTS, SEASONS = RAW / "Raw_Attempts.csv", RAW / "Raw_Season.csv"
CONTEXT, OPPS = RAW / "Raw_Attempt_Context.csv", RAW / "Raw_Opportunities.csv.gz"
PITCHES, PEOPLE, DOCS = RAW / "Raw_Statcast_Pitches.csv.gz", RAW / "people.csv", RAW / "statcast_csv_docs.csv"
POPTIME, SPRINT, LEAGUE, TEAM_MAP = RAW / "poptime.csv", RAW / "sprint_speed.csv", RAW / "league_sb_rates.csv", RAW / "team_map.csv"
SSSI, XSB = RAW / "DF_v7_SSSI.csv", RAW / "DF_v7_xSB_Outcome.csv"        # frozen legacy season features (v7)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

LEADS_URL = ("https://baseballsavant.mlb.com/leaderboard/services/"
             "basestealing-running-game/{rid}?season_start={y}&season_end={y}")
GAMELOG_URL = ("https://statsapi.mlb.com/api/v1/people/{rid}/stats"
               "?stats=gameLog&group=hitting&season={y}&gameType=R")
STATS_URL = ("https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
             "&season={y}&gameType=R&playerPool=All&sortStat=stolenBases&order=desc&limit=3000")
# `fields` shrinks playByPlay ~9x while keeping everything used. It is a FLAT key whitelist applied at every depth.
PBP_URL = ("https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay?fields="
           "allPlays,playEvents,playId,details,type,code,count,balls,strikes,outs,"
           "about,inning,halfInning,matchup,pitchHand,batSide,result,awayScore,homeScore")
PBP_OPPS_URL = ("https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay?fields="
                "allPlays,atBatIndex,about,inning,halfInning,playEvents,playId,isPitch,index,"
                "pitchNumber,details,type,code,count,balls,strikes,outs,matchup,pitchHand,"
                "batSide,postOnFirst,postOnSecond,postOnThird,id,runners,movement,originBase,"
                "start,end,outBase,isOut,event,eventType,playIndex,runner,result,awayScore,homeScore")
SCHED_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
SCHED_R_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&date={date}"
POPTIME_URL = "https://baseballsavant.mlb.com/leaderboard/poptime?year={y}&team=&min2sb=1&min3b=0&csv=true"
SPRINT_URL = ("https://baseballsavant.mlb.com/leaderboard/sprint_speed"
              "?attempts=1&min_season={s}&max_season={e}&position=&team=&csv=true")
LEAGUE_URL = "https://statsapi.mlb.com/api/v1/teams/stats?season={y}&sportId=1&group=hitting&stats=season&gameType=R"
STATCAST_GAME_URL = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&game_pk={pk}"
PITCH_INDEX_URL = ("https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay?fields="
                   "allPlays,atBatIndex,playEvents,playId,pitchNumber,isPitch")
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"
DOCS_URL = "https://baseballsavant.mlb.com/csv-docs"
VIDEO_URL = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"
GAMEFEED_URL = "https://baseballsavant.mlb.com/gamefeed?gamePk={pk}"

LEADS_COLS = ["runner_id", "season", "date", "play_id", "pitcher_id", "pitcher_name",
              "catcher_id", "catcher_name", "fielder_name", "base", "result", "run_value",
              "lead_at_firstmove_ft", "gain_to_release_ft", "lead_at_release_ft"]
DISC_COLS = ["runner_id", "name", "name_tag", "team", "position", "sb", "cs", "attempts",
             "success_pct", "sprint_speed_ftps", "sprint_pctile", "seasons"]


# ── fetch helpers (retry with linear backoff) ────────────────────────────────
def get_json(url, tries=4):
    """GET a URL and parse JSON; None after `tries` failed attempts."""
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=40); r.raise_for_status(); return r.json()
        except Exception as e:
            if attempt == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return None
            time.sleep(1.0 + attempt)
    return None


def get_csv_rows(url, tries=4):
    """GET a Savant CSV export as a list of dict rows (BOM-stripped)."""
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=40); r.raise_for_status()
            return list(csv.DictReader(io.StringIO(r.text.lstrip("﻿"))))
        except Exception as e:
            if attempt == tries - 1:
                print(f"  ! GET failed {url}: {e}")
                return []
            time.sleep(1.0 + attempt)
    return []


def as_float(x, ndigits=None):
    try:
        v = float(x)
        return round(v, ndigits) if ndigits is not None else v
    except (TypeError, ValueError):
        return None


def write_rows(path: Path, rows: list, cols: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


# ── 1 per-attempt leads: Savant's basestealing-running-game drawer ───────────
# Field mapping (verified against the public leaderboard drawer):
#   r_primary_lead -> lead_at_firstmove_ft (lead at the pitcher's first move), r_secondary_lead -> lead_at_release_ft
#   (lead at the pitcher's RELEASE), r_sec_minus_prim_lead -> gain_to_release_ft (ground gained between the two),
#   runner_moved_cd -> result, runs_stolen_on_running_act -> run_value. The drawer carries no ball-arrival time.
def fetch_leads(runner_id: int, year: int) -> list[dict]:
    d = get_json(LEADS_URL.format(rid=runner_id, y=year))
    data = d.get("data", []) if isinstance(d, dict) else (d or [])
    rows = [{"runner_id": runner_id, "season": year, "date": str(a.get("game_date", ""))[:10],
             "play_id": a.get("play_id"), "pitcher_id": a.get("pitcher_id"), "pitcher_name": a.get("pitcher_name"),
             "catcher_id": a.get("catcher_id"), "catcher_name": a.get("catcher_name"),
             "fielder_name": a.get("fielder_name"), "base": a.get("target_base"), "result": a.get("runner_moved_cd"),
             "run_value": as_float(a.get("runs_stolen_on_running_act"), 3),
             "lead_at_firstmove_ft": as_float(a.get("r_primary_lead"), 1),
             "gain_to_release_ft": as_float(a.get("r_sec_minus_prim_lead"), 1),
             "lead_at_release_ft": as_float(a.get("r_secondary_lead"), 1)} for a in data]
    rows.sort(key=lambda r: (r["date"] or "", str(r["pitcher_name"])))
    return rows


def statsapi_sb_cs(runner_id: int, year: int) -> tuple[int, int]:
    """A runner's official SB / CS season total (the coverage check)."""
    d = get_json(GAMELOG_URL.format(rid=runner_id, y=year))
    splits = d["stats"][0]["splits"] if d and d.get("stats") else []
    return (sum(int(s["stat"].get("stolenBases", 0) or 0) for s in splits),
            sum(int(s["stat"].get("caughtStealing", 0) or 0) for s in splits))


def write_leads(runner_id: int, year: int, out: Path) -> int:
    rows = fetch_leads(runner_id, year)
    if not rows:
        print(f"  ! no attempts returned for {runner_id} {year}")
        return 0
    write_rows(out, rows, LEADS_COLS)
    n_sb, n_cs = sum(r["result"] == "SB" for r in rows), sum(r["result"] == "CS" for r in rows)
    sb, cs = statsapi_sb_cs(runner_id, year)
    print(f"[write] {out.name}  tracked {n_sb} SB / {n_cs} CS  |  StatsAPI {year} total {sb} SB / {cs} CS "
          f"(gap = steals of home / non-2B-3B / untracked)")
    return len(rows)


# ── 2 discovery: who to scrape ───────────────────────────────────────────────
def fetch_sb_cs(start: int, end: int) -> dict:
    """Per-runner SB/CS summed across the season range (year-correct, StatsAPI)."""
    agg: dict = {}
    for y in range(start, end + 1):
        d = get_json(STATS_URL.format(y=y))
        for s in (d["stats"][0]["splits"] if d and d.get("stats") else []):
            stat, p = s["stat"], s["player"]
            sb, cs = int(stat.get("stolenBases", 0) or 0), int(stat.get("caughtStealing", 0) or 0)
            if sb + cs == 0:
                continue
            a = agg.setdefault(int(p["id"]), {"runner_id": int(p["id"]), "name": p["fullName"], "sb": 0, "cs": 0,
                                              "team": (s.get("team") or {}).get("abbreviation", ""),
                                              "position": (s.get("position") or {}).get("abbreviation", ""),
                                              "seasons": set()})
            a["sb"] += sb; a["cs"] += cs; a["seasons"].add(y)
    return agg


def fetch_sprint(start: int, end: int) -> dict:
    """Sprint speed + within-population percentile per runner (Savant)."""
    out, speeds = {}, []
    for r in get_csv_rows(SPRINT_URL.format(s=start, e=end)):
        try:
            pid, spd = int(r["player_id"]), float(r["sprint_speed"])
        except (KeyError, TypeError, ValueError):
            continue
        out[pid] = {"sprint_speed": spd, "team": r.get("team", ""), "position": r.get("position", "")}
        speeds.append((pid, spd))
    speeds.sort(key=lambda x: x[1])
    for rank, (pid, _) in enumerate(speeds):
        out[pid]["pctile"] = round(100.0 * rank / (len(speeds) - 1), 1) if len(speeds) > 1 else 0.0
    return out


def discover(start, end, min_attempts=1, max_sprint_pctile=None, top=None, sort="attempts"):
    sbcs, sprint = fetch_sb_cs(start, end), fetch_sprint(start, end)
    rows = []
    for pid, a in sbcs.items():
        att, sp = a["sb"] + a["cs"], sprint.get(pid, {})
        last = (str(a["name"]).split() or ["runner"])[-1]
        rows.append({"runner_id": pid, "name": a["name"], "name_tag": "".join(c for c in last.lower() if c.isalnum()) or "runner",
                     "team": a["team"] or sp.get("team", ""), "position": a["position"] or sp.get("position", ""),
                     "sb": a["sb"], "cs": a["cs"], "attempts": att, "success_pct": round(100.0 * a["sb"] / att, 1) if att else "",
                     "sprint_speed_ftps": sp.get("sprint_speed", ""), "sprint_pctile": sp.get("pctile", ""),
                     "seasons": "/".join(str(y) for y in sorted(a["seasons"]))})
    rows = [r for r in rows if r["attempts"] >= min_attempts]
    if max_sprint_pctile is not None:
        rows = [r for r in rows if r["sprint_pctile"] != "" and r["sprint_pctile"] <= max_sprint_pctile]
    if sort == "slow":                                   # prolific AND slow
        rows.sort(key=lambda r: (r["attempts"], -(r["sprint_pctile"] if r["sprint_pctile"] != "" else 100.0)), reverse=True)
    else:
        rows.sort(key=lambda r: (r["attempts"], r["sb"]), reverse=True)
    return rows[:top] if top else rows


# ── 3 season tables ──────────────────────────────────────────────────────────
def fetch_poptime(start: int, end: int) -> None:
    """Savant pop-time leaderboard per season -> poptime.csv (joins on catcher_id, season)."""
    keep = ["pop_2b_sba", "pop_3b_sba", "maxeff_arm_2b_3b_sba", "exchange_2b_3b_sba", "pop_2b_sba_count"]
    out = []
    for y in range(start, end + 1):
        rows = get_csv_rows(POPTIME_URL.format(y=y))
        out += [{"catcher_id": r.get("entity_id"), "season": y, "catcher_name": r.get("entity_name"),
                 **{k: as_float(r.get(k), 3) for k in keep}} for r in rows]
        print(f"  poptime {y}: {len(rows)} catchers")
    write_rows(POPTIME, out, ["catcher_id", "season", "catcher_name"] + keep)
    print(f"[write] {POPTIME.name}  ({len(out)} catcher-seasons)")


def fetch_league_rates(start: int, end: int) -> None:
    """MLB regular-season SB, CS and games per season (sum over the 30 teams) -> league_sb_rates.csv."""
    out = []
    for y in range(start, end + 1):
        splits = ((get_json(LEAGUE_URL.format(y=y)) or {}).get("stats") or [{}])[0].get("splits", [])
        sb = sum(t["stat"].get("stolenBases", 0) for t in splits)
        cs = sum(t["stat"].get("caughtStealing", 0) for t in splits)
        games = sum(t["stat"].get("gamesPlayed", 0) for t in splits) / 2
        out.append({"season": y, "SB": sb, "CS": cs, "attempts": sb + cs, "games": int(games),
                    "success_pct": round(100 * sb / (sb + cs), 2) if sb + cs else "",
                    "attempts_per_game": round((sb + cs) / games, 3) if games else ""})
        print(f"  league {y}: {sb} SB / {cs} CS over {int(games)} games")
    write_rows(LEAGUE, out, list(out[0]))
    print(f"[write] {LEAGUE.name}  ({len(out)} seasons)")


def fetch_sprint_leaderboard(start: int, end: int) -> None:
    """Full sprint-speed leaderboard per season (every runner, not only qualified ones) -> sprint_speed.csv."""
    out = []
    for y in range(start, end + 1):
        rows = get_csv_rows(SPRINT_URL.format(s=y, e=y))
        out += [{"runner_id": r.get("player_id"), "season": y, "sprint_speed_all": as_float(r.get("sprint_speed"), 1)}
                for r in rows]
        print(f"  sprint speed {y}: {len(rows)} players")
    write_rows(SPRINT, out, ["runner_id", "season", "sprint_speed_all"])
    print(f"[write] {SPRINT.name}  ({len(out)} player-seasons)")


# ── 4 MLB feed: per-pitch context and the opportunity denominator ────────────
BASES = ("1B", "2B", "3B")


def _apply_movement(state, r):
    """Apply one runner movement to the base state (outs come from the feed's own count)."""
    mv, det = r.get("movement") or {}, r.get("details") or {}
    rid = (det.get("runner") or {}).get("id")
    start, end = mv.get("originBase") or mv.get("start"), mv.get("end")
    if mv.get("isOut"):
        for b in BASES:
            if state[b] == rid:
                state[b] = None
        return
    if end and end != start:
        for b in BASES:
            if state[b] == rid:
                state[b] = None
        if end in BASES:
            state[end] = rid


def _game_opportunities(pk):
    """One row per pitch with a runner on 1B and 2B empty, with the base state and TRUE pre-pitch count. Returns
    (rows, plate_appearances_checked, mismatches) so the caller can gate on the base-state replay's accuracy.
    Two feed traps: a steal is its own non-pitch ACTION event AFTER the pitch it happened on (runners[].playIndex
    points past the pitch, so it is attributed to the preceding pitch), and playEvents[].count is the count AFTER
    that event (so the pre-pitch count is the previous event's, 0-0 at the start of each plate appearance)."""
    OPPS_DIR.mkdir(parents=True, exist_ok=True)
    cache = OPPS_DIR / f"{pk}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text()); return d["rows"], d["checked"], d["bad"]
        except Exception:
            pass
    d = get_json(PBP_OPPS_URL.format(pk=pk)) or {}
    rows, checked, bad = [], 0, 0
    state, outs_now, cur_half = {b: None for b in BASES}, 0, None
    for play in d.get("allPlays", []):
        about = play.get("about") or {}
        half, inning = about.get("halfInning"), about.get("inning")
        if (half, inning) != cur_half:
            state, outs_now, cur_half = {b: None for b in BASES}, 0, (half, inning)
        mu, res = play.get("matchup") or {}, play.get("result") or {}
        away, home = res.get("awayScore"), res.get("homeScore")
        diff = None if away is None or home is None else ((away - home) if half == "top" else (home - away))
        is_lhp = 1 if (mu.get("pitchHand") or {}).get("code") == "L" else 0
        bat_r = 1 if (mu.get("batSide") or {}).get("code") == "R" else 0
        moves = {}
        for r in play.get("runners", []):
            moves.setdefault((r.get("details") or {}).get("playIndex"), []).append(r)
        seen, balls, strikes, last_row = set(), 0, 0, None
        for ev in play.get("playEvents", []):
            i = ev.get("index")
            if ev.get("isPitch"):
                on1, on2 = state["1B"], state["2B"]
                if on1 and not on2:
                    last_row = {"game_pk": pk, "at_bat": play.get("atBatIndex"), "half": half, "inning": inning,
                                "pitch_index": i, "play_id": ev.get("playId"), "runner_1b": on1, "outs": outs_now,
                                "balls": balls, "strikes": strikes, "score_diff": diff, "is_lhp": is_lhp,
                                "bat_side_r": bat_r, "pitch_code": ((ev.get("details") or {}).get("type") or {}).get("code"),
                                "attempt": 0, "attempt_type": ""}
                    rows.append(last_row)
                else:
                    last_row = None
            for r in moves.get(i, []):
                seen.add(i)
                et = ((r.get("details") or {}).get("eventType") or "")
                if et in ("stolen_base_2b", "caught_stealing_2b", "pickoff_1b", "pickoff_caught_stealing_2b") \
                        and last_row is not None:
                    last_row["attempt_type"] = et
                    if et.startswith(("stolen_base", "caught_stealing")):
                        last_row["attempt"] = 1
                _apply_movement(state, r)
            c = ev.get("count") or {}
            balls, strikes, outs_now = c.get("balls", balls), c.get("strikes", strikes), c.get("outs", outs_now)
        for idx in sorted((k for k in moves if k not in seen), key=lambda v: (v is None, v)):
            for r in moves[idx]:
                _apply_movement(state, r)
        pa_outs = (play.get("count") or {}).get("outs", outs_now)
        if pa_outs is not None and pa_outs >= 3:       # a third out empties the bases regardless of who reached
            state = {b: None for b in BASES}
        checked += 1
        post = {b: (mu.get("postOn" + w) or {}).get("id") for b, w in zip(BASES, ("First", "Second", "Third"))}
        if any(state[b] != post[b] for b in BASES):
            bad += 1
            state = dict(post)                       # resync so one bad PA cannot poison the inning
        outs_now = (play.get("count") or {}).get("outs", outs_now)
    cache.write_text(json.dumps({"rows": rows, "checked": checked, "bad": bad}, separators=(",", ":")))
    return rows, checked, bad


def _schedule(dates, url) -> list:
    pks = []
    for i, dt in enumerate(dates, 1):
        pks += [g.get("gamePk") for d_ in (get_json(url.format(date=dt)) or {}).get("dates", []) for g in d_.get("games", [])]
        if i % 100 == 0:
            print(f"  schedule {i}/{len(dates)} -> {len(set(pks))} games")
    return sorted({p for p in pks if p})


def fetch_opportunities() -> None:
    """Raw_Opportunities.csv.gz — every pitch with a runner on 1B and 2B empty: the denominator of 'will he go?'."""
    dates = sorted({r["date"][:10] for r in csv.DictReader(open(ATTEMPTS)) if r.get("date")})
    pks = _schedule(dates, SCHED_R_URL)
    print(f"{len(pks)} regular-season games to read")
    cols = ["game_pk", "at_bat", "half", "inning", "pitch_index", "play_id", "runner_1b", "outs", "balls", "strikes",
            "score_diff", "is_lhp", "bat_side_r", "pitch_code", "attempt", "attempt_type"]
    n = checked = bad = 0
    with gzip.open(OPPS, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i, pk in enumerate(pks, 1):
            rows, c, b = _game_opportunities(pk)
            checked, bad, n = checked + c, bad + b, n + len(rows)
            w.writerows(rows)
            if i % 250 == 0:
                print(f"  games {i}/{len(pks)}  opportunities {n:,}  base-state accuracy {100 * (1 - bad / max(1, checked)):.2f}%")
    acc = 100 * (1 - bad / max(1, checked))
    print(f"[write] {OPPS.name}  ({n:,} opportunity pitches from {len(pks)} games); base-state replay matched MLB's "
          f"end-of-PA state on {acc:.2f}% of {checked:,} plate appearances ({'PASS' if acc >= 99 else 'FAIL'})")


def _game_context(pk: int) -> dict:
    """play_id -> pitch + situation for one game (cached). The count is the feed's POST-pitch count and score_diff
    the END-of-plate-appearance score: both are kept for the record and must never be model inputs."""
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    cache = PBP_DIR / f"{pk}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    ctx = {}
    for play in (get_json(PBP_URL.format(pk=pk)) or {}).get("allPlays", []):
        ab, mu, res = play.get("about", {}), play.get("matchup", {}), play.get("result", {})
        half, away, home = ab.get("halfInning"), res.get("awayScore"), res.get("homeScore")
        diff = ((away - home) if half == "top" else (home - away)) if away is not None and home is not None else None
        for ev in play.get("playEvents", []):
            pid = ev.get("playId")
            if not pid:
                continue
            cnt = ev.get("count", {}) or {}
            ctx[pid] = {"is_lhp": 1 if (mu.get("pitchHand", {}) or {}).get("code") == "L" else 0,
                        "bat_side_r": 1 if (mu.get("batSide", {}) or {}).get("code") == "R" else 0,
                        "pitch_code": ((ev.get("details", {}) or {}).get("type", {}) or {}).get("code"),
                        "balls": cnt.get("balls"), "strikes": cnt.get("strikes"), "outs": cnt.get("outs"),
                        "inning": ab.get("inning"), "score_diff": diff}
    cache.write_text(json.dumps(ctx, separators=(",", ":")))
    return ctx


def fetch_context() -> None:
    """Raw_Attempt_Context.csv — every attempt's pitch context, Savant play_id joined to the MLB play-by-play."""
    att = list(csv.DictReader(open(ATTEMPTS)))
    pks = _schedule(sorted({r["date"][:10] for r in att if r.get("date")}), SCHED_URL)
    ctx = {}
    for i, pk in enumerate(pks, 1):
        ctx.update(_game_context(pk))
        if i % 250 == 0:
            print(f"  games {i}/{len(pks)}  ({len(ctx)} pitches indexed)")
    cols = ["play_id", "is_lhp", "bat_side_r", "pitch_code", "balls", "strikes", "outs", "inning", "score_diff"]
    rows = [{"play_id": r["play_id"], **{k: ctx[r["play_id"]].get(k) for k in cols[1:]}} for r in att if r.get("play_id") in ctx]
    write_rows(CONTEXT, rows, cols)
    print(f"[write] {CONTEXT.name}  ({len(rows)}/{len(att)} attempts matched, {100 * len(rows) / max(1, len(att)):.1f}%)")


# ── 5 Statcast: every Statcast column for every attempt pitch ────────────────
# Savant's per-game CSV export has every column but no play_id; the MLB feed supplies play_id -> (at_bat_number,
# pitch_number), the export's own pitch key. Full games are cached (gzipped); only the attempt pitches go to raw.
def attempt_games() -> dict:
    """play_id -> game_pk for every tracked attempt, from the per-game context cache (no network)."""
    want, out = set(pd.read_csv(ATTEMPTS, usecols=["play_id"]).play_id), {}
    for f in PBP_DIR.glob("*.json"):
        for pid in want.intersection(json.loads(f.read_text())):
            out[pid] = int(f.stem)
    return out


def fetch_statcast_game(pk: int) -> str:
    """Cache one game's Savant CSV (gzipped) and its play_id index. Resumable; returns a status word."""
    SC_DIR.mkdir(parents=True, exist_ok=True); PI_DIR.mkdir(parents=True, exist_ok=True)
    sc, pi = SC_DIR / f"{pk}.csv.gz", PI_DIR / f"{pk}.json"
    if not sc.exists():
        for k in range(4):
            try:
                r = requests.get(STATCAST_GAME_URL.format(pk=pk), headers={"User-Agent": UA}, timeout=90)
                r.raise_for_status(); txt = r.text.lstrip("﻿")
                if "pitch_type" not in txt[:40]:
                    raise ValueError("not a Statcast CSV")
                tmp = sc.with_suffix(".tmp"); tmp.write_bytes(gzip.compress(txt.encode())); tmp.rename(sc)
                break
            except Exception:
                if k == 3:
                    return "statcast failed"
                time.sleep(2 + 2 * k)
    if not pi.exists():
        try:
            r = requests.get(PITCH_INDEX_URL.format(pk=pk), headers={"User-Agent": UA}, timeout=40); r.raise_for_status()
            idx = {ev["playId"]: [p["atBatIndex"] + 1, ev.get("pitchNumber")]
                   for p in r.json().get("allPlays", []) for ev in p.get("playEvents", []) if ev.get("playId") and ev.get("isPitch")}
            pi.write_text(json.dumps(idx, separators=(",", ":")))
        except Exception:
            return "index failed"
    return "ok"


def fetch_statcast(workers: int = 3, stop_at: str | None = None) -> None:
    """Download every game with a tracked attempt into the cache, then write Raw_Statcast_Pitches.csv.gz.
    stop_at='HH:MM' ends the download cleanly at that local time (e.g. before a metered connection); re-run to resume."""
    import datetime as dt
    from concurrent.futures import ThreadPoolExecutor
    games = sorted(set(attempt_games().values()))
    todo = [pk for pk in games if not ((SC_DIR / f"{pk}.csv.gz").exists() and (PI_DIR / f"{pk}.json").exists())]
    stop = None
    if stop_at:
        h, m = map(int, stop_at.split(":")); now = dt.datetime.now()
        stop = now.replace(hour=h, minute=m, second=0)
        stop += dt.timedelta(days=1) if stop <= now else dt.timedelta(0)
    print(f"{len(games)} games with attempts, {len(todo)} to fetch" + (f", stopping at {stop:%H:%M}" if stop else ""), flush=True)
    t0, n, bad = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        for pk, st in zip(todo, ex.map(lambda pk: "stopped" if stop and dt.datetime.now() >= stop else fetch_statcast_game(pk), todo)):
            n += 1; bad += st not in ("ok", "stopped")
            if st not in ("ok", "stopped"):
                print(f"  ! {pk}: {st}", flush=True)
            if n % 200 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  [{n}/{len(todo)}] {el / n:.2f} s/game, ETA {el / n * (len(todo) - n) / 60:.0f} min, failed {bad}", flush=True)
    extract_pitches()


def extract_pitches() -> pd.DataFrame:
    """Raw_Statcast_Pitches.csv.gz: the Statcast row of every attempt pitch, play_id first (cache only, no network)."""
    want = set(pd.read_csv(ATTEMPTS, usecols=["play_id"]).play_id)
    parts, missing = [], 0
    for pk in sorted(set(attempt_games().values())):
        sc, pi = SC_DIR / f"{pk}.csv.gz", PI_DIR / f"{pk}.json"
        if not (sc.exists() and pi.exists()):
            missing += 1; continue
        idx = pd.DataFrame([(p, a, b) for p, (a, b) in json.loads(pi.read_text()).items() if p in want],
                           columns=["play_id", "at_bat_number", "pitch_number"])
        if len(idx):
            parts.append(idx.merge(pd.read_csv(sc, low_memory=False), on=["at_bat_number", "pitch_number"], how="inner"))
    out = pd.concat(parts, ignore_index=True)
    out = out[["play_id"] + [c for c in out.columns if c != "play_id"]]
    assert not out.play_id.duplicated().any(), "a play_id matched two Statcast rows"
    out.to_csv(PITCHES, index=False)
    print(f"[write] {PITCHES.name}  ({len(out):,} of {len(want):,} attempt pitches; {missing} games not cached yet)")
    return out


# ── 6 names and field definitions ────────────────────────────────────────────
def fetch_people() -> None:
    """people.csv — MLB full name for every runner and batter id (StatsAPI, 100 ids per request, cumulative)."""
    have = pd.read_csv(PEOPLE) if PEOPLE.exists() else pd.DataFrame(columns=["player_id", "name"])
    ids = set(pd.read_csv(ATTEMPTS, usecols=["runner_id"]).runner_id)
    if PITCHES.exists():
        ids |= set(pd.read_csv(PITCHES, usecols=["batter"]).batter.dropna().astype(int))
    todo = sorted(ids - set(have.player_id))
    new = []
    for i in range(0, len(todo), 100):
        new += [{"player_id": p["id"], "name": p["fullName"]}
                for p in (get_json(PEOPLE_URL.format(ids=",".join(map(str, todo[i:i + 100])))) or {}).get("people", [])]
    out = pd.concat([have, pd.DataFrame(new, columns=["player_id", "name"])]).drop_duplicates("player_id").sort_values("player_id")
    out.to_csv(PEOPLE, index=False)
    print(f"[write] {PEOPLE.name}  ({len(out):,} players; {len(new)} new, {len(todo) - len(new)} not found)")


def fetch_docs() -> None:
    """statcast_csv_docs.csv — Savant's own definition of every Statcast CSV field (baseballsavant.mlb.com/csv-docs)."""
    page = CACHE / "statcast_csv_docs.html"
    if not page.exists():
        page.write_text(SESSION.get(DOCS_URL, timeout=30).text)
    cards = re.findall(r'<div id="docs-([^"]+)" class="savant-docs-card">.*?<p>(.*?)</p>', page.read_text(), flags=re.S)
    rows = [{"field": k, "description": " ".join(html.unescape(re.sub("<[^>]+>", " ", v)).split()), "source": DOCS_URL,
             "fetched": time.strftime("%Y-%m-%d", time.localtime(page.stat().st_mtime))} for k, v in cards]
    write_rows(DOCS, rows, ["field", "description", "source", "fetched"])
    print(f"[write] {DOCS.name}  ({len(rows)} fields)")


# ── 7 site assets: headshots + team map ──────────────────────────────────────
ABBR_FIX = {"ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH", "OAK": "ATH",
            "AthLetics": "ATH"}                     # StatsAPI abbreviation -> logo filename in docs/assets/logos
HEADSHOT_URL = ("https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/"
                "w_213,q_auto:best/v1/people/{pid}/headshot/67/current")
PERSON_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season&season={yr}&group=hitting"


def fetch_assets(source=SSSI):
    """Cache MLB headshots per runner into docs/assets/headshots and resolve each runner-season's team -> team_map.csv."""
    HEADSHOTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source, usecols=["runner_id", "season"]).drop_duplicates()
    logo_names = {p.stem for p in LOGOS.glob("*.png")}
    id2abbr = {t["id"]: t["abbreviation"] for t in (get_json("https://statsapi.mlb.com/api/v1/teams?sportId=1") or {}).get("teams", [])
               if t.get("id") and t.get("abbreviation")}
    for pid in set(df["runner_id"].astype(int)):
        out = HEADSHOTS / f"{pid}.png"
        if out.exists():
            continue
        try:
            r = SESSION.get(HEADSHOT_URL.format(pid=pid), timeout=20)
            if r.ok and r.content and len(r.content) > 1000:
                out.write_bytes(r.content)
        except Exception:
            pass
    rows, missing = [], set()
    for pid, yr in df[["runner_id", "season"]].astype(int).itertuples(index=False):
        d = get_json(PERSON_STATS_URL.format(pid=pid, yr=yr))
        splits = (d or {}).get("stats", [{}])[0].get("splits", []) if d else []
        abbr = id2abbr.get(splits[-1].get("team", {}).get("id") if splits else None)   # last stint = latest team
        logo = ABBR_FIX.get(abbr, abbr) if abbr else None
        if logo and logo not in logo_names:
            missing.add(logo)
        rows.append({"runner_id": pid, "season": yr, "team": logo or ""})
        time.sleep(0.05)
    tm = pd.DataFrame(rows).drop_duplicates(["runner_id", "season"]); tm.to_csv(TEAM_MAP, index=False)
    print(f"[write] {TEAM_MAP.name}  ({len(tm)} rows, {tm['team'].ne('').sum()} with team; "
          f"{len(list(HEADSHOTS.glob('*.png')))} headshots)" + (f"; no logo for {sorted(missing)}" if missing else ""))


# ── 8 build: the raw modelling tables from the caches (no network) ───────────
XSB_ONLY = ["runner_id", "season", "net_sb", "z_net_sb", "z_sprint", "xsb_outcome", "sb_potential_gap", "quadrant"]
REF_ATTEMPTS, MIN_ATTEMPTS = 30, 10        # legacy season-level Steal+ scale; qualified runner-season
LEADS_FILE = re.compile(r"(\d+)_(\d{4})\.csv$")


def _iter_leads():
    """(runner_id, season, frame) for every readable, non-empty leads file, in file-name order."""
    for f in sorted(LEADS_DIR.glob("*.csv")):
        m = LEADS_FILE.search(f.name)
        if not m:
            continue
        try:
            a = pd.read_csv(f)
        except Exception:
            continue
        if not a.empty:
            yield int(m.group(1)), int(m.group(2)), a


def sb_run_value_map():
    """Statcast SB run value per runner-season = sum of the per-attempt run_value (reference only)."""
    return pd.DataFrame([{"runner_id": r, "season": s, "sb_run_value": round(float(pd.to_numeric(a["run_value"], errors="coerce").sum()), 3)}
                         for r, s, a in _iter_leads() if "run_value" in a.columns])


def add_legacy_metrics(df):
    """Legacy season-level Steal+ (sb_residual x 30) and the net-bases baseline: reference columns only."""
    q = df[df["sb_attempts"] >= MIN_ATTEMPTS]
    league_net_rate = float((q["SB"] - q["CS"]).sum() / max(1, q["sb_attempts"].sum()))
    df["league_net_rate"] = round(league_net_rate, 4)
    df["steal_plus"] = (df["sb_residual"] * REF_ATTEMPTS).round(2)
    df["exp_net"] = (league_net_rate * df["sb_attempts"]).round(2)
    return df


def refresh_live_sbcs(df, years, min_att=1):
    """Overlay year-correct StatsAPI SB/CS + sprint onto the given seasons (an in-progress season) without touching the
    frozen v7 table; new runner-seasons carry NaN for the v7 running-mechanics columns."""
    sp = pd.read_csv(SPRINT).rename(columns={"sprint_speed_all": "sprint_speed"})
    updated = [df[~df["season"].isin(years)]]
    for y in years:
        live, cur, rows = fetch_sb_cs(y, y), df[df["season"] == y].set_index("runner_id"), []
        for pid, a in live.items():
            att = a["sb"] + a["cs"]
            if att < min_att:
                continue
            base = cur.loc[pid].to_dict() if pid in cur.index else {c: np.nan for c in df.columns}
            base.update({"runner_id": pid, "season": y, "SB": a["sb"], "CS": a["cs"], "sb_attempts": att,
                         "player_name": a["name"] or base.get("player_name")})
            spd = sp[(sp.runner_id == pid) & (sp.season == y)]["sprint_speed"]
            if len(spd):
                base["sprint_speed"] = float(spd.iloc[0])
            rows.append(base)
        got = pd.DataFrame(rows)
        print(f"  refreshed {y}: {len(got)} runner-seasons (was {len(cur)}), total SB {int(got['SB'].sum()) if len(got) else 0}")
        updated.append(got)
    return pd.concat(updated, ignore_index=True)


def build_season(refresh_years=None, refresh_min_att=10):
    """Raw_Season.csv: one row per runner-season (v7 features + xSB + team + Statcast SB run value)."""
    df = pd.read_csv(SSSI).merge(pd.read_csv(XSB)[XSB_ONLY], on=["runner_id", "season"], how="left")
    if refresh_years:
        df = refresh_live_sbcs(df, refresh_years, refresh_min_att)
    df = df.merge(pd.read_csv(TEAM_MAP), on=["runner_id", "season"], how="left") if TEAM_MAP.exists() else df.assign(team="")
    df = add_legacy_metrics(df).merge(sb_run_value_map(), on=["runner_id", "season"], how="left")
    df.to_csv(SEASONS, index=False)
    print(f"[write] {SEASONS.name}  ({len(df)} rows x {df.shape[1]} cols; Statcast run value on {int(df['sb_run_value'].notna().sum())} rows)")


def build_attempts():
    """Raw_Attempts.csv: one row per tracked running event 2023+, in leads-file order (the canonical attempt order).
    The cache also holds 2015-22 leads; the 2023 rules changed the game, so the modern table starts in 2023."""
    frames = []
    for runner_id, season, a in _iter_leads():
        if season < 2023:
            continue
        if "runner_id" not in a.columns:            # older cache files carry no id/season columns
            a.insert(0, "runner_id", runner_id); a.insert(1, "season", season)
        frames.append(a)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = (df["result"].astype(str).str.upper() == "SB").astype(int)
    df.to_csv(ATTEMPTS, index=False)
    print(f"[write] {ATTEMPTS.name}  ({len(df)} rows x {df.shape[1]} cols, {df['y'].mean() * 100:.1f}% SB)")


# ── 9 meta: one comprehensive table per season ───────────────────────────────
DEPRECATED = ["spin_dir", "spin_rate_deprecated", "break_angle_deprecated", "break_length_deprecated", "tfs_deprecated",
              "tfs_zulu_deprecated", "umpire", "sv_id"]       # dropped only if empty for every attempt pitch
CV_COLS = {"qa": "cv_qa", "delivery_s": "cv_delivery_s", "confidence": "cv_confidence", "why": "cv_why", "feed": "cv_feed",
           "home_qa": "cv_home_qa", "fps": "cv_fps", "lift_frame": "cv_lift_frame", "release_frame": "cv_release_frame",
           "lead_foot": "cv_lead_foot", "release_to_wrist": "cv_release_to_wrist", "lift_peak_in": "cv_lift_peak_in",
           "ball_pts": "cv_ball_pts"}
POP_COLS = {"pop_2b_sba": "catcher_pop_2b", "pop_3b_sba": "catcher_pop_3b", "maxeff_arm_2b_3b_sba": "catcher_arm_mph",
            "exchange_2b_3b_sba": "catcher_exchange_s", "pop_2b_sba_count": "catcher_pop_2b_n"}
FRONT = ["season", "date", "game_pk", "play_id", "video_url", "video_url_away", "gamefeed_url", "qa_flags",
         "inning", "inning_topbot", "outs_when_up", "balls", "strikes", "on_1b", "on_2b", "on_3b", "home_team", "away_team",
         "bat_score", "fld_score",
         "runner_id", "runner_name", "base", "result", "y", "run_value", "lead_at_firstmove_ft", "gain_to_release_ft",
         "lead_at_release_ft", "runner_sprint_speed",
         "pitcher_id", "pitcher_name", "p_throws", "cv_delivery_s", "cv_qa", "cv_confidence", "cv_feed",
         "catcher_id", "catcher_name", "fielder_name", "catcher_pop_2b", "catcher_pop_3b", "catcher_arm_mph", "catcher_exchange_s",
         "batter", "batter_name", "stand", "pitch_type", "pitch_name", "release_speed", "zone", "description", "events"]
# what each project column means; Statcast columns are described from Savant's own docs (statcast_csv_docs.csv)
DERIVED = {
    "season": ("attempt", "Savant drawer", "Season of the attempt."),
    "date": ("attempt", "Savant drawer", "Game date (YYYY-MM-DD)."),
    "game_pk": ("identity", "MLB feed", "MLB game id."),
    "play_id": ("identity", "Savant drawer", "Savant/MLB pitch id of the pitch the runner went on. Join key for every table."),
    "video_url": ("link", "derived", "Baseball Savant video of the play (home broadcast)."),
    "video_url_away": ("link", "derived", "The same play on the away broadcast."),
    "gamefeed_url": ("link", "derived", "Savant game feed for the whole game."),
    "qa_flags": ("qa", "derived", "ok, or a semicolon list of cross-source disagreements for this row: "
                 "no_statcast_row, no_feed_context, pitcher_mismatch (Statcast pitcher != drawer pitcher), catcher_mismatch "
                 "(Statcast fielder_2 != drawer catcher), runner_not_on_base (runner not on the base he left, before the "
                 "pitch), date_mismatch, pitch_type_mismatch (MLB feed code != Statcast pitch_type)."),
    "runner_id": ("runner", "Savant drawer", "MLBAM id of the runner."),
    "runner_name": ("runner", "MLB StatsAPI people", "Runner's name."),
    "base": ("attempt", "Savant drawer", "Base the runner went for: 2B or 3B."),
    "result": ("attempt", "Savant drawer (runner_moved_cd)", "SB stolen base, CS caught stealing. PK, BK and FB are running "
               "events that did not happen on a pitch (checked on 12 sampled plays each against the MLB feed: PK on a pickoff "
               "attempt, BK on a balk, FB on a disengagement, i.e. a pickoff attempt or step-off), so they have no Statcast "
               "row. Models use SB/CS only."),
    "y": ("attempt", "derived", "1 if result == SB, else 0 (the modelling target)."),
    "run_value": ("attempt", "Savant drawer", "Statcast runs stolen on the running act (Savant's own credit)."),
    "lead_at_firstmove_ft": ("attempt", "Savant drawer (r_primary_lead)", "Runner's lead (ft) at the pitcher's first move."),
    "gain_to_release_ft": ("attempt", "Savant drawer (r_sec_minus_prim_lead)", "Ground gained (ft) from the pitcher's first "
                           "move to release: the secondary lead minus the primary lead."),
    "lead_at_release_ft": ("attempt", "Savant drawer (r_secondary_lead)", "Runner's lead (ft) at the pitcher's release "
                           "(= first-move lead + ground gained)."),
    "runner_sprint_speed": ("runner", "Savant sprint-speed leaderboard", "Runner's sprint speed that season (ft/s)."),
    "pitcher_id": ("pitcher", "Savant drawer", "MLBAM id of the pitcher."),
    "pitcher_name": ("pitcher", "Savant drawer", "Pitcher's name."),
    "catcher_id": ("catcher", "Savant drawer", "MLBAM id of the catcher."),
    "catcher_name": ("catcher", "Savant drawer", "Catcher's name."),
    "fielder_name": ("catcher", "Savant drawer", "Fielder named on the play in the drawer."),
    "catcher_pop_2b": ("catcher", "Savant pop-time leaderboard", "Catcher's season pop time to 2B on steal attempts (s)."),
    "catcher_pop_3b": ("catcher", "Savant pop-time leaderboard", "Catcher's season pop time to 3B (s)."),
    "catcher_arm_mph": ("catcher", "Savant pop-time leaderboard", "Catcher's max-effort arm strength (mph)."),
    "catcher_exchange_s": ("catcher", "Savant pop-time leaderboard", "Catcher's exchange time (s)."),
    "catcher_pop_2b_n": ("catcher", "Savant pop-time leaderboard", "Throws behind the catcher's pop time to 2B."),
    "batter_name": ("batter", "Statcast player_name", "Batter's name (Savant's player_name on pitch-level rows is the batter)."),
    "cv_qa": ("pitcher (CV)", "vision/delivery.py", "PASS, or why the clip could not be timed (camera cut, no release found, ...)."),
    "cv_delivery_s": ("pitcher (CV)", "vision/delivery.py", "Pitcher's delivery time (s): lead-foot lift-off to ball release, "
                      "from the broadcast clip. About 34 ms short of hand labels on average."),
    "cv_confidence": ("pitcher (CV)", "vision/delivery.py", "high = every signal inside the envelope verified on the 50 gold "
                      "clips; low = outside it (reasons in cv_why); not measured = no PASS."),
    "cv_why": ("pitcher (CV)", "vision/delivery.py", "Signals outside the verified envelope."),
    "cv_feed": ("pitcher (CV)", "vision/delivery.py", "Broadcast the delivery was timed on (HOME first, AWAY when HOME fails QA)."),
    "cv_home_qa": ("pitcher (CV)", "vision/delivery.py", "QA result on the home broadcast."),
    "cv_fps": ("pitcher (CV)", "vision/delivery.py", "Frames per second of the clip."),
    "cv_lift_frame": ("pitcher (CV)", "vision/delivery.py", "Frame of lead-foot lift-off."),
    "cv_release_frame": ("pitcher (CV)", "vision/delivery.py", "Frame of ball release."),
    "cv_lift_s": ("pitcher (CV)", "derived", "Lift-off, seconds into the clip (open video_url and skip here)."),
    "cv_release_s": ("pitcher (CV)", "derived", "Release, seconds into the clip."),
    "cv_lead_foot": ("pitcher (CV)", "vision/delivery.py", "Lead (glove-side) foot: L or R."),
    "cv_release_to_wrist": ("pitcher (CV)", "vision/delivery.py", "Ball-to-wrist distance at release (x pitcher height)."),
    "cv_lift_peak_in": ("pitcher (CV)", "vision/delivery.py", "Peak rise of the lead foot (inches)."),
    "cv_ball_pts": ("pitcher (CV)", "vision/delivery.py", "Ball detections in the 9 frames after release."),
    "feed_is_lhp": ("MLB feed", "StatsAPI playByPlay", "1 if the pitcher is left-handed."),
    "feed_bat_side_r": ("MLB feed", "StatsAPI playByPlay", "1 if the batter hits right-handed."),
    "feed_pitch_code": ("MLB feed", "StatsAPI playByPlay", "MLB pitch-type code of the pitch."),
    "feed_balls": ("MLB feed", "StatsAPI playByPlay", "Balls AFTER the pitch (the feed's count). Not a pre-pitch input: use balls."),
    "feed_strikes": ("MLB feed", "StatsAPI playByPlay", "Strikes AFTER the pitch. Not a pre-pitch input: use strikes."),
    "feed_outs": ("MLB feed", "StatsAPI playByPlay", "Outs on the pitch event (the caught-stealing out is recorded on a later event)."),
    "feed_inning": ("MLB feed", "StatsAPI playByPlay", "Inning."),
    "feed_score_diff": ("MLB feed", "StatsAPI playByPlay", "Batting team's lead at the END of the plate appearance: can include "
                        "runs scored after the attempt. Never a model input; use bat_score - fld_score."),
    "attempt_idx": ("identity", "derived", "Row number in data/raw/Raw_Attempts.csv. Sorting on it restores the order the "
                    "models were fit in (it fixes the cross-validation folds)."),
}


def _flags(m: pd.DataFrame) -> pd.Series:
    num = lambda c: pd.to_numeric(m[c], errors="coerce")
    checks = {"no_statcast_row": m["at_bat_number"].isna(),
              "no_feed_context": m["feed_pitch_code"].isna() & m["feed_inning"].isna(),
              "pitcher_mismatch": m["pitcher"].notna() & (num("pitcher") != num("pitcher_id")),
              "catcher_mismatch": m["fielder_2"].notna() & m["catcher_id"].notna() & (num("fielder_2") != num("catcher_id")),
              "runner_not_on_base": m["at_bat_number"].notna() & np.where(m["base"] == "3B", num("on_2b") != num("runner_id"),
                                                                          num("on_1b") != num("runner_id")),
              "date_mismatch": m["game_date"].notna() & (m["game_date"].astype(str) != m["date"].astype(str)),
              "pitch_type_mismatch": m["pitch_type"].notna() & m["feed_pitch_code"].notna() & (m["pitch_type"] != m["feed_pitch_code"])}
    out = pd.Series("", index=m.index)
    for name, hit in checks.items():
        out = out.where(~hit, out + np.where(out == "", "", ";") + name)
    return out.replace("", "ok")


def build_meta() -> None:
    """data/meta/meta_<season>.csv + data_dictionary.csv. One row per Savant drawer row (every SB/CS and other running
    event), every Statcast column of that pitch, the MLB feed context, season pop time / sprint speed, the CV delivery
    time, links to the video, and qa_flags. Left joins only: no attempt is ever dropped or duplicated (asserted)."""
    att = pd.read_csv(ATTEMPTS)
    att["attempt_idx"] = np.arange(len(att))
    sc = pd.read_csv(PITCHES, low_memory=False)
    empty = [c for c in DEPRECATED if c in sc.columns and sc[c].isna().all()]
    sc = sc.drop(columns=empty)
    ctx = pd.read_csv(CONTEXT).rename(columns=lambda c: c if c == "play_id" else f"feed_{c}")
    sp = pd.read_csv(SPRINT).rename(columns={"sprint_speed_all": "runner_sprint_speed"})
    pop = pd.read_csv(POPTIME)[["catcher_id", "season"] + list(POP_COLS)].rename(columns=POP_COLS)
    people = pd.read_csv(PEOPLE).set_index("player_id")["name"] if PEOPLE.exists() else pd.Series(dtype=str)
    season_names = pd.read_csv(SEASONS)[["runner_id", "season", "player_name"]].rename(columns={"player_name": "season_name"})
    cvs = [pd.read_csv(f) for f in sorted(DELIVERY.glob("delivery_20[0-9][0-9].csv"))]
    cv = pd.concat(cvs).drop_duplicates("play_id") if cvs else pd.DataFrame(columns=["play_id"])
    cv = cv[["play_id"] + [c for c in CV_COLS if c in cv.columns]].rename(columns=CV_COLS)

    m = (att.merge(sc, on="play_id", how="left").merge(ctx, on="play_id", how="left")
            .merge(sp, on=["runner_id", "season"], how="left").merge(pop, on=["catcher_id", "season"], how="left")
            .merge(season_names, on=["runner_id", "season"], how="left").merge(cv, on="play_id", how="left"))
    assert len(m) == len(att) and m.attempt_idx.is_unique, "a join multiplied rows"
    m["runner_name"] = m.runner_id.map(people).fillna(m.pop("season_name"))
    m["batter_name"] = m.batter.map(people).fillna(m["player_name"])      # Statcast's player_name is the batter
    m["video_url"] = VIDEO_URL.format(pid="") + m.play_id
    m["video_url_away"] = m.video_url + "&videoType=AWAY"
    m["gamefeed_url"] = np.where(m.game_pk.notna(), GAMEFEED_URL.format(pk="") + m.game_pk.astype("Int64").astype(str), "")
    if "cv_lift_frame" in m:
        m["cv_lift_s"], m["cv_release_s"] = (m.cv_lift_frame / m.cv_fps).round(2), (m.cv_release_frame / m.cv_fps).round(2)
    m["qa_flags"] = _flags(m)
    rest = [c for c in m.columns if c not in FRONT and not c.startswith(("cv_", "feed_")) and c != "attempt_idx"]
    order = [c for c in FRONT if c in m] + rest + sorted(c for c in m if c.startswith("cv_") and c not in FRONT) + \
            [c for c in m if c.startswith("feed_")] + ["attempt_idx"]
    m = m[order].sort_values(["season", "date", "game_pk", "at_bat_number", "pitch_number", "attempt_idx"], na_position="last")
    META.mkdir(parents=True, exist_ok=True)
    for s, g in m.groupby("season"):
        g.to_csv(META / f"meta_{s}.csv", index=False)
        print(f"[write] meta_{s}.csv  {len(g):,} rows x {g.shape[1]} cols | Statcast row {g.at_bat_number.notna().mean():.1%} | "
              f"feed context {g.feed_inning.notna().mean():.1%} | CV timed {int((g['cv_qa'] == 'PASS').sum()) if 'cv_qa' in g else 0:,} | "
              f"clean rows {(g.qa_flags == 'ok').mean():.1%}")
    dictionary(m, empty)


def dictionary(m: pd.DataFrame, dropped: list) -> None:
    docs = pd.read_csv(DOCS).set_index("field")["description"].to_dict() if DOCS.exists() else {}
    alias = {"vz0": "vx0-2", "hit_distance_sc": "hit_distance", "release_spin_rate": "release_spin"}   # the docs page's own ids
    undocumented = {"post_fld_score": "the fielding team's score after the play", "bat_speed": "Statcast bat-tracking bat speed",
                    "swing_length": "Statcast bat-tracking swing length", "miss_distance": "a bat-tracking miss distance",
                    "estimated_slg_using_speedangle": "expected slugging from exit velocity and launch angle",
                    "delta_pitcher_run_exp": "the change in run expectancy credited to the pitcher"}
    sc_cols = set(pd.read_csv(PITCHES, nrows=0).columns) - {"play_id"}
    rows = []
    for c in m.columns:
        if c in DERIVED:
            block, src, desc = DERIVED[c]
        elif c in sc_cols:
            block, src = "Statcast pitch", "Savant Statcast search CSV (per game), joined on play_id via the MLB feed's (at_bat_number, pitch_number)"
            desc = docs.get(c) or docs.get(alias.get(c, ""), "") or (
                "No entry on Savant's CSV docs page" + (f"; the name suggests {undocumented[c]} (unverified)." if c in undocumented else "."))
        else:
            block, src, desc = "other", "", ""
        rows.append({"column": c, "block": block, "source": src, "description": desc,
                     "non_null_pct": round(100 * m[c].notna().mean(), 1)})
    rows += [{"column": c, "block": "dropped", "source": "Statcast", "description": "Empty for every attempt pitch (deprecated by Savant).",
              "non_null_pct": 0.0} for c in dropped]
    pd.DataFrame(rows).to_csv(META / "data_dictionary.csv", index=False)
    print(f"[write] data_dictionary.csv  ({len(rows)} columns; dropped as always empty: {dropped or 'none'})")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("leads", help="one runner-season's per-attempt leads")
    pl.add_argument("runner_id", type=int); pl.add_argument("year", type=int)
    pd_ = sub.add_parser("discover", help="rank base-stealers for a season range")
    pd_.add_argument("--start", type=int, required=True); pd_.add_argument("--end", type=int, required=True)
    pd_.add_argument("--min-attempts", type=int, default=1); pd_.add_argument("--max-sprint-pctile", type=float, default=None)
    pd_.add_argument("--top", type=int, default=None); pd_.add_argument("--sort", choices=["attempts", "slow"], default="attempts")
    pd_.add_argument("--expand", action="store_true", help="then pull leads for every kept runner")
    for name, hlp, start in [("poptime", "catcher pop time + arm", 2023), ("sprint", "full sprint-speed leaderboard", 2023),
                             ("league", "league SB/CS/games per season", 2015)]:
        p = sub.add_parser(name, help=hlp); p.add_argument("--start", type=int, default=start); p.add_argument("--end", type=int, default=2026)
    sub.add_parser("context", help="per-pitch hand / count / situation for every attempt")
    sub.add_parser("opportunities", help="every pitch with a runner on 1B and 2B empty")
    sc = sub.add_parser("statcast", help="every Statcast column for every attempt pitch")
    sc.add_argument("--workers", type=int, default=3); sc.add_argument("--stop-at", default=None, help="HH:MM local")
    sub.add_parser("people", help="names for every runner and batter")
    sub.add_parser("docs", help="Savant's Statcast field definitions")
    sub.add_parser("assets", help="headshots + team map")
    b = sub.add_parser("build", help="Raw_Attempts.csv + Raw_Season.csv from the caches")
    b.add_argument("--refresh", default=None, help="comma list of seasons to overlay live StatsAPI SB/CS on (network)")
    sub.add_parser("meta", help="the per-season meta tables + data dictionary")
    a = ap.parse_args()

    if a.cmd == "leads":
        write_leads(a.runner_id, a.year, LEADS_DIR / f"{a.runner_id}_{a.year}.csv")
    elif a.cmd == "discover":
        rows = discover(a.start, a.end, a.min_attempts, a.max_sprint_pctile, a.top, a.sort)
        out = DISC_DIR / f"runners_{a.start}_{a.end}.csv"; write_rows(out, rows, DISC_COLS)
        print(f"[write] {out.name}  ({len(rows)} runners)")
        if a.expand:
            for r in rows:
                for y in range(a.start, a.end + 1):
                    write_leads(r["runner_id"], y, LEADS_DIR / f"{r['runner_id']}_{y}.csv")
    elif a.cmd in ("poptime", "sprint", "league"):
        {"poptime": fetch_poptime, "sprint": fetch_sprint_leaderboard, "league": fetch_league_rates}[a.cmd](a.start, a.end)
    elif a.cmd == "statcast":
        fetch_statcast(a.workers, a.stop_at)
    elif a.cmd == "build":
        build_season([int(y) for y in a.refresh.split(",")] if a.refresh else None); build_attempts()
    else:
        {"context": fetch_context, "opportunities": fetch_opportunities, "people": fetch_people, "docs": fetch_docs,
         "assets": fetch_assets, "meta": build_meta}[a.cmd]()


if __name__ == "__main__":
    main()
