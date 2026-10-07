#!/usr/bin/env python3
"""
delivery.py — pitcher delivery time (lead-foot lift-off -> ball release) from the Baseball Savant broadcast clip of
every stolen-base attempt. Output: 1-Data-Ingestion/data/delivery/delivery_<season>.csv, one row per SB/CS attempt
(QA verdict, the two frame numbers, delivery_s, confidence), which ingest.py joins into the meta tables.

Detectors: BaseballCV's broadcast models (github.com/dylandru/BaseballCV; weights in vision/weights/, run as Core ML
on the Apple Neural Engine when the .mlpackage exports exist) — pitcher/hitter/catcher, glove/homeplate/baseball/
rubber, baseball — plus RTMPose-m (Halpe-26 feet keypoints). Per clip, the minimum work for the two frame numbers:
  1 anchor   one set-position frame: centre-field picture (pitcher in front of catcher and plate) + pitcher box
  2 release  ball detector on the native 640 px pitcher-catcher slice, every 4th frame over the release prior, then
             full rate around the first hit; a thrown ball traces a near-straight image line (constant-velocity
             chaining) while a ball in the hand follows the arm's arc; the earliest point on the line is the release
  3 lift     pose on [release - 1.75 s, release - 0.35 s]; lift-off = the glove-side foot's lowest point rising
             > lift_in above a still SET position (both feet still >= 0.3 s), measured in inches (rubber = 24 in,
             else home plate = 17 in)
  4 QA       PASS only if no camera cut between the set and release, every sampled frame is the CF picture, the
             camera is steady, the release is AT THE HAND, a still set precedes the lift, and 0.45 s <= time <= 2 s
The camera gate comes first: a clip that cuts to the runner mid-delivery is never timed. HOME broadcast first, AWAY
only when HOME fails QA. Detections are cached per clip+feed (cache/fast/), and decide() is a pure function of that
cache, so scoring 1, 5 or 11,000 clips gives identical rows. Accuracy and its history: vision/REPORT.md.

Usage (the cv venv: 1-Data-Ingestion/vision/.venv/bin/python):
  delivery.py season 2025            time every SB/CS attempt of a season (resumable: re-run to continue)
  delivery.py prefetch 2025 [4]      download the season's clips ahead of `season` (so timing can run offline)
  delivery.py gold                   score the 50 hand-labelled clips (accuracy check)
  delivery.py refresh 2025           re-decide a season from the detection cache (no video)
  delivery.py pilot 2025             one attempt per pitcher (seeded) — a quick coverage check
"""
import html, json, re, sys, threading, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import cv2

VISION = Path(__file__).resolve().parent
DATA = VISION.parent / "data"
OUT, RAW = DATA / "delivery", DATA / "raw"
CACHE = VISION / "cache"
CLIPS, FDET, PBP = CACHE / "clips", CACHE / "fast", CACHE / "pbp"
GOLD = VISION / "gold" / "labels_manual.csv"
for d in (CLIPS, FDET, PBP, OUT):
    d.mkdir(parents=True, exist_ok=True)

WEIGHTS = {"phc": "pitcher_hitter_catcher_detector_v3.pt", "glove": "glove_tracking_v4_YOLOv11.pt",
           "ball": "ball_tracking_v4-YOLOv11.pt"}
NAMES = {   # the Core ML export loses class names; these are the .pt weights' own lists (checked 2026-10-01)
    "phc": {0: "hitter", 1: "pitcher", 2: "catcher"},
    "glove": {0: "glove", 1: "homeplate", 2: "baseball", 3: "rubber", 4: "na"},
    "ball": {0: "glove", 1: "homeplate", 2: "baseball", 3: "rubber"},
}
RTMPOSE_M = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
             "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip")
FEET_LOW = {"L": (20, 22, 24), "R": (21, 23, 25)}   # Halpe-26 (big toe, small toe, heel): lowest = on the ground
RUBBER_IN, PLATE_IN = 24.0, 17.0                    # MLB pitcher's plate 24 in long; home plate 17 in wide
CFG = dict(
    conf_person=0.25, conf_obj=0.15, conf_ball=0.15,
    conf_floor=0.05,             # ball detections are CACHED down to this; conf_ball applies at decide time
    conf_rubber=0.03,            # rubber is accepted only under the pitcher's feet (see scene())
    flight_gap=5,                # max missing frames inside the flight run (batter / catcher occlusion)
    flight_tol=0.03,             # max deviation from the constant-velocity prediction (x pitcher height),
                                 # widened by gap/2 for gaps > 2 frames (extrapolation uncertainty)
    flight_vmax=0.12,            # max speed between a chain's first two points (x pitcher height / frame)
    flight_min_pts=8,
    catch_s=0.032,               # flight run end - plateTime: ball travels plate front -> glove
    rest_n=8, rest_px=2.0,       # the ball is AT REST (caught) once it stays within rest_px for rest_n frames
    base_s=0.3,                  # a still SET position lasts >= this
    steady=0.10,                 # max drift of pitcher's / catcher's feet (x pitcher height): pan / zoom guard
    plate_to_mound=1.18,         # scale at the mound / scale at the plate (camera ~400 ft vs ~340 ft)
    lift_in=0.75, peak_in=2.0,   # lift-off threshold (leave-one-out vs the hand labels); the lift must reach peak_in
    still_in=0.3,                # a foot moving < this many inches / frame (2-D) is still (set-position test)
    delivery_s=(0.45, 2.0),      # physically plausible lift-off -> release time
)
FAST = dict(rel_band=(150, 236), stride=4, wide_band=(100, 280), pose_s=(1.75, 0.35), hand_max=0.6)


def _device():
    """NVIDIA GPU -> 'cuda', Apple GPU -> 'mps', else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = _device()


# ── fetch ────────────────────────────────────────────────────────────────────
def get(url, timeout, tries=4, **kw):
    """requests.get with retries (2, 4, 8 s backoff): one dropped connection must not end a long run."""
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout, **kw); r.raise_for_status(); return r
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))


def fetch(play_id, feed="HOME"):
    """Savant clip + game_pk for one pitch (the page BaseballCV's BaseballSavVideoScraper parses)."""
    mp4, meta = CLIPS / f"{play_id}_{feed}.mp4", CLIPS / f"{play_id}_{feed}.json"
    if meta.exists():
        return (mp4 if mp4.exists() else None), json.loads(meta.read_text())
    url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}" + ("&videoType=AWAY" if feed == "AWAY" else "")
    page = get(url, 30, headers={"User-Agent": "Mozilla/5.0"}).text
    src = re.search(r'<source\s+src="(https://sporty-clips\.mlb\.com/[^"]+\.mp4)"', page)
    pk = re.search(r"gamefeed\?gamePk=(\d+)", page)
    m = {"game_pk": int(pk.group(1)) if pk else None,
         "mp4_url": html.unescape(src.group(1)) if src else None}       # the src is entity-escaped (&#x3D; = '=')
    for _ in range(2 if src else 0):
        _get_clip(m["mp4_url"], mp4)
        cap = cv2.VideoCapture(str(mp4)); ok = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 30; cap.release()
        if ok:
            break
        mp4.unlink(); time.sleep(2)
    meta.write_text(json.dumps(m))
    return (mp4 if mp4.exists() else None), m


PREFIX_FRAMES = 330   # process() reads frames <= ~300 (ball band to 280 + 12, cuts() to 280/fps + 0.3 s): 30 spare


def _boxes(b, i, end):
    """(type, payload start, end) of the MP4 boxes in b[i:end]."""
    while i + 8 <= end:
        size, typ, hdr = int.from_bytes(b[i:i + 4], "big"), bytes(b[i + 4:i + 8]), 8
        if size == 1:
            size, hdr = int.from_bytes(b[i + 8:i + 16], "big"), 16
        elif size == 0:
            size = end - i
        if size < hdr:
            return
        yield typ, i + hdr, min(i + size, end)
        i += size


def _video_prefix(moov, frames):
    """Bytes of the file that hold its first `frames` video samples (decode order), from the moov's sample tables
    (stsz, stsc, stco/co64); None when they cannot be read or the clip is not longer than that."""
    u = lambda o, k=4: int.from_bytes(moov[o:o + k], "big")
    def child(s, e, *path):
        for typ in path:
            s, e = next(((a, b) for t, a, b in _boxes(moov, s, e) if t == typ), (None, None))
            if s is None:
                return None, None
        return s, e
    for t, s, e in _boxes(moov, 8, len(moov)):
        h = child(s, e, b"mdia", b"hdlr")[0]
        if t != b"trak" or h is None or moov[h + 8:h + 12] != b"vide":
            continue
        box = {t2: a for t2, a, _ in _boxes(moov, *child(s, e, b"mdia", b"minf", b"stbl"))} if child(s, e, b"mdia", b"minf", b"stbl")[0] else {}
        if not {b"stsz", b"stsc"} <= box.keys() or not ({b"stco", b"co64"} & box.keys()):
            return None
        z = box[b"stsz"]; fixed, count = u(z + 4), u(z + 8)
        sizes = [fixed] * count if fixed else [u(z + 12 + 4 * j) for j in range(count)]
        k = 8 if b"co64" in box else 4; c = box.get(b"co64", box.get(b"stco"))
        offs = [u(c + 8 + k * j, k) for j in range(u(c + 4))]
        q = box[b"stsc"]; runs = [(u(q + 8 + 12 * j), u(q + 12 + 12 * j)) for j in range(u(q + 4))]   # (first chunk, samples)
        if frames >= count:
            return None
        end, n = 0, 0
        for ci, off in enumerate(offs, 1):
            for _ in range([r[1] for r in runs if r[0] <= ci][-1]):
                if n == frames:
                    return end
                off += sizes[n]; end, n = max(end, off), n + 1
        return None
    return None


def _get_clip(url, mp4):
    """Download what process() reads, not the whole clip: the moov (sample tables, at the end of these files) and the
    bytes holding the first PREFIX_FRAMES video samples, each written at its own offset in a file of full size (the gap
    reads as zeros and is never decoded, so every frame process() decodes is the original). Any surprise (no range
    support, moov not at the end, unreadable tables) downloads the whole file instead."""
    r = get(url, 90, headers={"Range": "bytes=-524288"})
    if r.status_code != 206:
        mp4.write_bytes(r.content); return
    total, tail = int(r.headers["Content-Range"].rsplit("/", 1)[1]), r.content
    t0, i = total - len(tail), tail.rfind(b"moov")
    while i >= 4 and not (tail[i + 8:i + 12] == b"mvhd" and i - 4 + int.from_bytes(tail[i - 4:i], "big") <= len(tail)):
        i = tail.rfind(b"moov", 0, i)
    x = _video_prefix(tail[i - 4:i - 4 + int.from_bytes(tail[i - 4:i], "big")], PREFIX_FRAMES) if i >= 4 else None
    head = get(url, 90, headers={"Range": f"bytes=0-{x - 1}"}).content if x and x < t0 else b""
    if len(head) != (x or -1) or not x or x >= t0:
        mp4.write_bytes(get(url, 90).content); return
    with open(mp4, "wb") as f:
        f.write(head); f.seek(t0); f.write(tail)


def plate_time(game_pk, play_id):
    """Statcast release -> plate flight time (s), from the MLB play-by-play feed (cached per game)."""
    f = PBP / f"{game_pk}.json"
    if not f.exists():
        try:
            d = get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
                    "?fields=allPlays,playEvents,playId,pitchData,plateTime", 40).json()
        except requests.RequestException:
            return None                  # unreachable: carry on without it (not cached, so retried next time)
        tmp = f.with_name(f"{f.stem}.{threading.get_ident()}.tmp")    # atomic: download threads share the cache
        tmp.write_text(json.dumps({e["playId"]: (e.get("pitchData") or {}).get("plateTime")
                                   for p in d.get("allPlays", []) for e in p.get("playEvents", []) if e.get("playId")}))
        tmp.replace(f)
    return json.loads(f.read_text()).get(play_id)


# ── models + frame access ────────────────────────────────────────────────────
_M = {}
def model(name):
    """BaseballCV detectors as Core ML packages when exported (else the .pt), RTMPose on onnxruntime."""
    if name not in _M:
        if name == "pose":
            from rtmlib import RTMPose
            _M[name] = RTMPose(onnx_model=RTMPOSE_M, model_input_size=(192, 256), backend="onnxruntime",
                               device="mps" if DEVICE == "mps" else DEVICE)
        else:
            from ultralytics import YOLO
            pt = VISION / "weights" / WEIGHTS[name]
            ml = pt.with_suffix(".mlpackage")
            _M[name] = (YOLO(str(ml), task="detect"), True) if ml.exists() else (YOLO(str(pt)), False)
    return _M[name]


def detect(name, frame, imgsz, conf, box=None):
    """[(class, conf, x1, y1, x2, y2), ...] from one detector, on the whole frame or on a slice of it."""
    m, coreml = model(name)
    x1, y1 = (int(box[0]), int(box[1])) if box else (0, 0)
    img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])] if box else frame
    r = m.predict(img, imgsz=imgsz, conf=conf, verbose=False, **({} if coreml else {"device": DEVICE}))[0]
    return [(NAMES[name][int(c)], round(float(s), 3), *[round(float(v) + o, 1) for v, o in zip(b, (x1, y1, x1, y1))])
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def corridor(s, W, H, pad=0.35, side=640):
    """Slice holding the whole pitch in one CF scene: pitcher + catcher boxes padded by pad x pitcher height,
    grown to >= side px each way (so the detector sees it at native resolution or better)."""
    ph = s["pitcher"][3] - s["pitcher"][1]
    box = []
    for k, lim in ((0, W), (1, H)):
        lo = min(s["pitcher"][k], s["catcher"][k]) - pad * ph
        hi = max(s["pitcher"][k + 2], s["catcher"][k + 2]) + pad * ph
        if hi - lo < side:
            lo, hi = (lo + hi - side) / 2, (lo + hi + side) / 2
        lo, hi = (0, hi - lo) if lo < 0 else (lo, hi)            # shift inside the frame, then clip
        lo, hi = (lo - (hi - lim), lim) if hi > lim else (lo, hi)
        box.append((max(0, int(lo)), min(lim, int(hi))))
    return [box[0][0], box[1][0], box[0][1], box[1][1]]


def read(path, idx):
    """Yield (i, frame) for the requested indices, streaming (sequential decode: H.264 seeking is not frame-exact;
    nothing is held in memory — the machine has 8 GB shared with the GPU)."""
    want, i = set(int(k) for k in idx), 0
    cap = cv2.VideoCapture(str(path))
    while want and i <= max(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            yield i, f
        i += 1
    cap.release()


# ── the centre-field picture ─────────────────────────────────────────────────
def cuts(path, end_s):
    from scenedetect import detect as sd_detect, AdaptiveDetector
    return [s[0].frame_num for s in sd_detect(str(path), AdaptiveDetector(), end_time=float(end_s))][1:]


def scene(frame):
    """Best box per CF object in one frame: pitcher/catcher/hitter + homeplate, plus the rubber only where it should be
    — under the pitcher's feet (his feet hide most of it, so its confidence is often 0.03-0.1; the location constraint
    is what makes it safe). Core ML models are fixed at the exported size: 800."""
    best, rubbers = {}, []
    for c, s, *b in detect("phc", frame, 800, CFG["conf_person"]) + detect("glove", frame, 800, CFG["conf_rubber"]):
        if c == "rubber":
            rubbers.append((s, b))
        elif (c in ("pitcher", "catcher", "hitter") or (c == "homeplate" and s >= CFG["conf_obj"])) \
                and s > best.get(c, (0,))[0]:
            best[c] = (s, b)
    out = {c: b for c, (s, b) in best.items()}
    if "pitcher" in out:
        px1, py1, px2, py2 = out["pitcher"]; h = py2 - py1
        at_feet = [(s, b) for s, b in rubbers if px1 - 0.3 * h <= (b[0] + b[2]) / 2 <= px2 + 0.3 * h
                   and abs((b[1] + b[3]) / 2 - py2) <= 0.15 * h]
        if at_feet:
            out["rubber"] = max(at_feet)[1]
    return out


def is_cf(s):
    """Pitcher's centre-field view: pitcher in the foreground (bigger, feet lower in frame) with the catcher beyond
    him; home plate, when visible, lies beyond the pitcher's feet with the catcher at it."""
    if not all(k in s for k in ("pitcher", "catcher")):
        return False
    px1, py1, px2, py2 = s["pitcher"]; cx1, cy1, cx2, cy2 = s["catcher"]
    ok = py2 - py1 > 1.2 * (cy2 - cy1) and py2 > cy2  # pitcher nearer the camera than the catcher
    if "homeplate" in s:
        hx, hy = (s["homeplate"][0] + s["homeplate"][2]) / 2, (s["homeplate"][1] + s["homeplate"][3]) / 2
        ok = ok and hy < py2 and cx1 - (cx2 - cx1) <= hx <= cx2 + (cx2 - cx1)
    return ok


# ── release: the pitch's flight run ──────────────────────────────────────────
def balls_of(det, sources=("balls", "balls_hr")):
    """{frame: [(x, y, conf), ...]} of 'baseball' detections."""
    B = {}
    for src in sources:
        for i, d in det.get(src, {}).items():
            B.setdefault(int(i), []).extend(((x1 + x2) / 2, (y1 + y2) / 2, s) for c, s, x1, y1, x2, y2 in d
                                            if c == "baseball" and s >= CFG["conf_ball"])
    return B


def flight(det, min_pts=None, gapmax=None):
    """The pitch in flight. Detections are chained by constant-velocity prediction (error <= flight_tol x pitcher
    height; the 2nd point only needs speed <= flight_vmax), each chain is cut at the catch (the ball stops in, or is
    pulled back by, the glove), the longest qualifying chain is taken (>= flight_min_pts, moves >= 15% of pitcher
    height, lasts <= plateTime + 0.25 s), and it is extended BACKWARD to the earliest detection on its back-
    extrapolated line — that frame is the release. Returns (release, catch) frames, or None.
    (min_pts / gapmax are relaxed only to LOCATE the flight on the coarse pass.)"""
    ph, fps, pt = det["ph"], det["fps"], det.get("plate_time")
    B, B_all = balls_of(det, ("balls",)), balls_of(det)
    tol, vmax = CFG["flight_tol"] * ph, CFG["flight_vmax"] * ph
    gapmax, min_pts = gapmax or CFG["flight_gap"], min_pts or CFG["flight_min_pts"]

    def err(c, t, x, y):          # 1-point chain: speed; longer: deviation from constant-velocity prediction
        gap = t - c[-1][0]
        if len(c) == 1:
            return np.hypot(x - c[-1][1], y - c[-1][2]) / gap
        (t1, x1, y1), (t2, x2, y2) = c[-2], c[-1]
        return np.hypot(x - (x2 + (x2 - x1) / (t2 - t1) * gap), y - (y2 + (y2 - y1) / (t2 - t1) * gap))

    n_max = int(((pt or 0.45) + CFG["catch_s"] + 0.10) * fps)   # release -> catch can't take longer than this

    def to_catch(c):              # end at the catch: the image path is a projected arc that slows and turns at its
        c = [p for p in c if p[0] - c[0][0] <= n_max]          # apex, so no speed/turn test; a ball AT REST (< rest_px
        P = np.array([(x, y) for _, x, y in c])                 # for rest_n frames: in the glove) ends it
        for k in range(len(c) - CFG["rest_n"]):
            if np.hypot(*(P[k + 1:k + 1 + CFG["rest_n"]] - P[k]).T).max() < CFG["rest_px"]:
                return c[:k + 1]
        return c

    widen = lambda gap: max(1.0, gap / 2)          # extrapolation over longer gaps is less certain
    chains = []
    for t in sorted(B):
        for x, y, s in sorted(B[t], key=lambda b: -b[2]):
            ok = [(err(c, t, x, y), c) for c in chains if 0 < t - c[-1][0] <= gapmax]
            ok = [(e, c) for e, c in ok if e <= (tol * widen(t - c[-1][0]) if len(c) > 1 else vmax)]
            if ok:
                min(ok, key=lambda ec: (t - ec[1][-1][0], ec[0]))[1].append((t, x, y))
            else:
                chains.append([(t, x, y)])
    chains = [to_catch(c) for c in chains if len(c) >= 2]
    good = [c for c in chains if len(c) >= min_pts
            and max(np.hypot(x - c[0][1], y - c[0][2]) for _, x, y in c) >= 0.15 * ph   # reach, not net displacement
            and (pt is None or (c[-1][0] - c[0][0]) / fps <= pt + 0.25)]
    if not good:
        return None
    c = max(good, key=len)
    while True:                                    # backward extension along the flight line
        (t1, x1, y1), (t2, x2, y2) = c[0], c[1]
        vx, vy = (x2 - x1) / (t2 - t1), (y2 - y1) / (t2 - t1)
        back = [(np.hypot(x - (x1 - vx * k), y - (y1 - vy * k)), t1 - k, x, y)
                for k in range(1, gapmax + 1) for x, y, s in B_all.get(t1 - k, [])]
        back = [b for b in back if b[0] <= tol * widen(t1 - b[1])]
        if not back:
            break
        _, t, x, y = min(back, key=lambda b: (-b[1], b[0]))   # nearest frame first
        c.insert(0, (t, x, y))
    return c[0][0], to_catch(c)[-1][0]


# ── lift-off: the lead foot leaves the ground ────────────────────────────────
def lift_off(pose, ppi, fps, lead_side=None):
    """First frame the lead foot leaves the ground (its LOWEST point — heel or toe — rises), measured relative to the
    planted foot in inches (cancels camera tilt / zoom), 3-frame median (guards a one-frame left/right keypoint swap).
    Anchored on a SET POSITION: >= base_s with both feet still in 2-D (MLB's mandatory stop; 2-D so a slide step's
    level slide is not mistaken for a stop). Lift-off = first frame after the set > lift_in above it that reaches
    peak_in. lead_side = 'L' / 'R' (glove side) when known. Returns (frame, peak_rise_in, lead, set_found)."""
    t = np.array(sorted(pose)); nb = max(3, int(round(CFG["base_s"] * fps)))
    med = lambda v: pd.Series(v).rolling(3, center=True, min_periods=1).median().values
    low = {s: med([max(pose[k][j, 1] for j in ix) for k in t]) / ppi for s, ix in FEET_LOW.items()}
    ctr = {s: np.c_[med([pose[k][list(ix), 0].mean() for k in t]), med([pose[k][list(ix), 1].mean() for k in t])] / ppi
           for s, ix in FEET_LOW.items()}                                 # foot centre (x, y), inches
    speed = {s: np.hypot(*np.gradient(ctr[s], axis=0).T) for s in ctr}
    still = np.append((speed["L"] < CFG["still_in"]) & (speed["R"] < CFG["still_in"]), False)
    sets, s0 = [], None                                       # runs of >= base_s with both feet still
    for i, ok in enumerate(still):
        if ok and s0 is None:
            s0 = i
        elif not ok and s0 is not None:
            if i - s0 >= nb:
                sets.append((s0, i))
            s0 = None
    best = None
    for lead, piv in (("L", "R"), ("R", "L")):
        if lead_side and lead != lead_side:
            continue
        d = low[piv] - low[lead]                              # lead foot's height above the pivot foot
        for r, (s0, s1) in enumerate(sets):                   # the first set that is followed by a lift
            up = d[s1:sets[r + 1][0] if r + 1 < len(sets) else len(t)] - np.median(d[s0:s1])
            if (up >= CFG["peak_in"]).any():
                below = np.where(up[:int(np.argmax(up >= CFG["peak_in"]))] <= CFG["lift_in"])[0]
                k = s1 + (below[-1] + 1 if len(below) else 0)
                if best is None or t[k] < best[0]:
                    best = (int(t[k]), round(float(up.max()), 2), lead)
                break
    return (*best, True) if best else (None, None, None, False)


# ── one clip: detect (expensive, cached) -> decide (pure) ────────────────────
def process(play_id, feed="HOME"):
    cache = FDET / f"{play_id}_{feed}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    t0 = time.time()
    mp4, meta = fetch(play_id, feed)
    det = {"play_id": play_id, "feed": feed, **meta}

    def done(status):
        det.update(status=status, seconds=round(time.time() - t0, 1)); cache.write_text(json.dumps(det)); return det

    if mp4 is None:
        return done("no_clip")
    cap = cv2.VideoCapture(str(mp4)); fps = cap.get(cv2.CAP_PROP_FPS)
    W, H, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(7)); cap.release()
    det.update(fps=fps, n=n, plate_time=plate_time(meta["game_pk"], play_id) if meta.get("game_pk") else None)
    det["cuts"] = cuts(mp4, min(n / fps, FAST["wide_band"][1] / fps + 0.3))
    anchor = None                                             # 1 anchor: first CF one of three candidate frames
    for f0 in (140, 110, 170):
        s = scene(next(read(mp4, [f0]))[1])
        if is_cf(s):
            anchor = (f0, s); break
    if anchor is None:
        return done("not_cf_view")
    det["anchor"] = [anchor[0], anchor[1]]
    hb = det["corridor"] = corridor(anchor[1], W, H)          # native 640 px slice
    det["ph"] = anchor[1]["pitcher"][3] - anchor[1]["pitcher"][1]
    det["balls"] = {}                                         # 2 release: coarse pass, then full rate around the hit
    for lo, hi in (FAST["rel_band"], FAST["wide_band"]):
        for i, f in read(mp4, range(lo, min(hi, n), FAST["stride"])):
            det["balls"][str(i)] = detect("ball", f, 640, CFG["conf_floor"], hb)
        c0 = flight(det, min_pts=3, gapmax=2 * FAST["stride"])
        if c0:
            break
    if not c0:
        return done("no_release")
    for i, f in read(mp4, [i for i in range(c0[0] - 8, c0[0] + 12) if str(i) not in det["balls"]]):
        det["balls"][str(i)] = detect("ball", f, 640, CFG["conf_floor"], hb)
    r = flight(det, min_pts=5)
    if not r:
        return done("no_release")
    rel = det["rel_process"] = r[0]
    a, b = rel - int(round(FAST["pose_s"][0] * fps)), rel - int(round(FAST["pose_s"][1] * fps))   # 3 lift window
    qf = sorted({max(0, a), (a + b) // 2, b, min(n - 1, rel + 6)})          # 4 QA frames (also give the pitcher box)
    det["samples"] = {str(i): scene(f) for i, f in read(mp4, qf)}
    P = sorted((int(i), s["pitcher"]) for i, s in det["samples"].items() if "pitcher" in s) + [(anchor[0], anchor[1]["pitcher"])]
    P = sorted(dict(P).items()); ts, bx = np.array([p[0] for p in P]), np.array([p[1] for p in P])
    det["pose"] = {}
    for i, f in read(mp4, list(range(max(0, a), b)) + [rel - 1]):
        x1, y1, x2, y2 = [np.interp(i, ts, bx[:, k]) for k in range(4)]; pad = 0.15 * (y2 - y1)
        kp, sc = model("pose")(f, bboxes=[[x1 - pad, y1 - pad, x2 + pad, y2 + pad]])
        det["pose"][str(i)] = np.round(np.c_[kp[0], sc[0]], 2).tolist()
    return done("detected")


def decide(det, throws=None):
    """Pure: QA + the two frame numbers + delivery time, from the cache."""
    row = {k: det.get(k) for k in ("play_id", "feed", "fps", "seconds")}
    if det.get("status") != "detected":
        return {**row, "qa": det["status"]}
    fps, rel = det["fps"], det["rel_process"]
    S = {int(i): s for i, s in det["samples"].items()}
    kp = det["pose"].get(str(rel - 1)); near = S[min(S, key=lambda q: abs(q - rel))]
    x0, y0 = max(balls_of(det)[rel], key=lambda b: b[2])[:2]
    hand = (min(np.hypot(x0 - kp[j][0], y0 - kp[j][1]) for j in (9, 10)) / (near["pitcher"][3] - near["pitcher"][1])
            if kp and "pitcher" in near else None)
    ph = np.median([s["pitcher"][3] - s["pitcher"][1] for s in S.values() if "pitcher" in s] or [np.nan])
    feet = lambda k: [s[k][3] for s in S.values() if k in s]
    pose = {int(i): np.array(k)[:, :2] for i, k in det["pose"].items() if int(i) < rel - 1}
    rub = [s["rubber"][2] - s["rubber"][0] for s in list(S.values()) + [det["anchor"][1]] if "rubber" in s]
    plate = [s["homeplate"][2] - s["homeplate"][0] for s in list(S.values()) + [det["anchor"][1]] if "homeplate" in s]
    ppi = np.median(rub) / RUBBER_IN if len(rub) >= 2 else (np.median(plate) / PLATE_IN * CFG["plate_to_mound"] if plate else np.nan)
    lift, peak, lead, has_set = lift_off(pose, ppi, fps, {"R": "L", "L": "R"}.get(throws)) if pose and ppi == ppi else (None, None, None, False)
    B = balls_of(det)
    row.update(release_frame=rel, lift_frame=lift, lead_foot=lead, release_to_wrist=None if hand is None else round(float(hand), 2),
               lift_peak_in=peak, ball_pts=sum(1 for i in range(rel, rel + 9) if B.get(i)))   # confidence signals
    start = (lift if lift is not None else min(pose or [rel])) - int(round(CFG["base_s"] * fps))
    if any(start < c <= rel + 6 for c in det["cuts"]):
        qa = "cut_in_window"
    elif not all(is_cf(s) for s in S.values()):
        qa = "not_cf_view"
    elif np.ptp(feet("pitcher")) > CFG["steady"] * ph or np.ptp(feet("catcher")) > CFG["steady"] * ph:
        qa = "camera_moved"
    elif hand is None or hand > FAST["hand_max"]:
        qa = "release_not_at_hand"
    elif not has_set:
        qa = "no_set_before_lift"
    elif not CFG["delivery_s"][0] <= (rel - lift) / fps <= CFG["delivery_s"][1]:
        qa = "implausible_delivery"
    else:
        qa = "PASS"
    row["qa"] = qa
    if qa == "PASS":
        row["delivery_s"] = round((rel - lift) / fps, 3)
    return row


_HAND = {}
def throws(play_id):
    """Pitcher's hand for a play_id: Raw_Attempt_Context.csv (is_lhp), else the gold labels."""
    if not _HAND:
        c = pd.read_csv(RAW / "Raw_Attempt_Context.csv", usecols=["play_id", "is_lhp"])
        _HAND.update(zip(c.play_id, np.where(c.is_lhp == 1, "L", "R")))
        if GOLD.exists():
            g = pd.read_csv(GOLD).dropna(subset=["play_id", "p_throws"])
            _HAND.update({p: h for p, h in zip(g.play_id, g.p_throws) if p not in _HAND})
    return _HAND.get(play_id)


def measure(play_id):
    """HOME feed first; AWAY only when HOME fails QA."""
    rows = []
    for feed in ("HOME", "AWAY"):
        rows.append(decide(process(play_id, feed), throws(play_id)))
        if rows[-1]["qa"] == "PASS":
            break
    return {**rows[-1], "home_qa": rows[0]["qa"], "seconds": sum(r.get("seconds") or 0 for r in rows)}


# ── confidence: inside or outside the envelope verified on the gold clips ────
# Ranges of the signals on the 37 gold clips that PASS (delivery MAE 0.048 s there). Signals did not predict error
# inside it (|Spearman| <= 0.27), so this is "inside verified territory" vs "outside", not a calibrated probability.
VERIFIED = dict(release_to_wrist=(None, 0.32), ball_pts=(5, None), lift_peak_in=(4.28, None), delivery_s=(0.70, 1.31))


def confidence(r):
    """('high', '') when every signal lies inside the verified envelope, ('low', reasons) otherwise."""
    if r.get("qa") != "PASS":
        return "not measured", r.get("qa")
    out = [f"{k} {r.get(k)} outside {lo if lo is not None else ''}-{hi if hi is not None else ''}"
           for k, (lo, hi) in VERIFIED.items()
           if r.get(k) is None or pd.isna(r.get(k)) or (lo is not None and r[k] < lo) or (hi is not None and r[k] > hi)]
    return ("low", "; ".join(out)) if out else ("high", "")


def write(rows, path):
    d = pd.DataFrame(rows)
    d[["confidence", "why"]] = [confidence(r) for r in d.to_dict("records")]
    d.to_csv(path, index=False)
    return d


# ── runs ─────────────────────────────────────────────────────────────────────
def pool(season, every=True, seed=0):
    """A season's SB/CS attempts in date order (all of them, or one per pitcher for the pilot)."""
    A = pd.read_csv(RAW / "Raw_Attempts.csv")
    p = A[(A.season == season) & A.result.isin(["SB", "CS"])]
    return (p if every else p.groupby("pitcher_id", group_keys=False).sample(1, random_state=seed)).sort_values("date")


def _download(pid, feed):
    """fetch() + the plate-time feed, for the download threads. Returns the error's name when fetch() fails (after
    get()'s 4 tries), else None."""
    try:
        mp4, meta = fetch(pid, feed)
    except Exception as e:
        return type(e).__name__
    try:
        if meta.get("game_pk"):
            plate_time(meta["game_pk"], pid)
    except Exception:
        pass                                              # process() asks for it again


def run(season, every=True, seed=0, workers=4, ahead=200):
    """Time a season's attempts in two passes: the HOME broadcast of every attempt, then the AWAY broadcast of those whose
    HOME clip failed QA (~43%). Same rows as timing attempt by attempt (decide() is a pure function of each clip's cache),
    but only the clips that are needed are downloaded, by `workers` threads that keep the next `ahead` clips coming while
    the current one is timed; when that queue is full they fetch clips needed later (AWAY clips of attempts whose HOME
    clip already failed, then the next season's HOME clips), so the downloads finish early and the timing can end
    offline. Resumable: the per-clip cache is the checkpoint. Videos are deleted once timed (gold clips kept). Writes
    data/delivery/delivery_<season>.csv (or pilot_<season>.csv) at the end."""
    from concurrent.futures import ThreadPoolExecutor
    pick = pool(season, every, seed)
    out = OUT / f"{'delivery' if every else 'pilot'}_{season}.csv"
    keep, t0, name = set(pd.read_csv(GOLD).play_id), time.time(), dict(zip(pick.play_id, pick.pitcher_name.astype(str)))
    home = lambda p: (FDET / f"{p}_HOME.json").exists() and decide(json.loads((FDET / f"{p}_HOME.json").read_text()),
                                                                    throws(p))["qa"]
    print(f"{out.name}: {len(pick)} attempts" + ("" if every else " (one per pitcher)"), flush=True)
    later = [(p, "HOME") for p in pool(season + 1).play_id] if every else []      # empty for the latest season
    away, jobs, errors = [], {}, {}                         # errors: (play_id, feed) -> what failed this run
    have = lambda q, f: (FDET / f"{q}_{f}.json").exists() or (CLIPS / f"{q}_{f}.json").exists()
    with ThreadPoolExecutor(workers) as ex:
        def spare():                                       # the pool would idle: fetch what a later pass will need
            busy = sum(not j.done() for j in jobs.values())
            while busy < 2 * workers and (away or later):
                q, f = (away or later).pop(0)
                if (q, f) not in jobs and not have(q, f):
                    jobs[(q, f)] = ex.submit(_download, q, f); busy += 1
        for feed in ("HOME", "AWAY"):
            ids = list(pick.play_id) if feed == "HOME" else [p for p in pick.play_id if home(p) not in (False, "PASS")]
            away, t1 = [], time.time()                     # pass 2 queues its own AWAY clips below
            print(f"{feed} pass: {len(ids)} clips", flush=True)
            for k, pid in enumerate(ids, 1):
                for q in ids[k - 1:k - 1 + ahead]:                       # keep the next `ahead` downloads queued
                    if (q, feed) not in jobs and not (FDET / f"{q}_{feed}.json").exists():
                        jobs[(q, feed)] = ex.submit(_download, q, feed)
                spare()
                err = jobs.pop((pid, feed)).result() if (pid, feed) in jobs else None   # the clip is on disk
                try:
                    if err:                               # its download already failed 4 times: not again here
                        raise RuntimeError(err)
                    r = decide(process(pid, feed), throws(pid))
                except Exception as e:                    # never let one clip end the run; not cached -> retried on resume
                    errors[(pid, feed)] = err or type(e).__name__
                    r = {"qa": f"error: {errors[(pid, feed)]}"}
                if pid not in keep:                       # timed clips are not needed again, nor AWAY once HOME passes
                    for f in [feed] + (["AWAY"] if r["qa"] == "PASS" else []):
                        (CLIPS / f"{pid}_{f}.mp4").unlink(missing_ok=True)
                if feed == "HOME" and r["qa"] != "PASS":
                    away.append((pid, "AWAY"))
                el = time.time() - t1
                print(f"[{feed} {k}/{len(ids)}] {name[pid]:24s} {r['qa']:22s} delivery {r.get('delivery_s')} | "
                      f"{el / k:.1f} s/clip, ETA {el / k * (len(ids) - k) / 60:.0f} min", flush=True)
    rows = []
    for a in pick.itertuples():                           # HOME, then AWAY on a fail, from the cache (no network)
        failed = [f for f in ("HOME", "AWAY") if (a.play_id, f) in errors and not (FDET / f"{a.play_id}_{f}.json").exists()]
        if failed and (failed[0] == "HOME" or home(a.play_id) != "PASS"):
            r = {"play_id": a.play_id, "qa": f"error: {errors[(a.play_id, failed[0])]}"}
        else:
            try:
                r = measure(a.play_id)
            except Exception as e:
                r = {"play_id": a.play_id, "qa": f"error: {type(e).__name__}"}
        rows.append({**r, "pitcher_id": a.pitcher_id, "pitcher_name": a.pitcher_name, "result": a.result, "base": a.base})
    d = write(rows, out)
    p = d[d.qa == "PASS"]
    print(f"\nwall time {(time.time() - t0) / 60:.1f} min | PASS {len(p)}/{len(d)} ({len(p) / max(1, len(d)):.0%}) | "
          f"feed of PASS: {p.feed.value_counts().to_dict()} | QA funnel: {d.qa.value_counts().to_dict()}")
    print(f"delivery_s on PASS: mean {p.delivery_s.mean():.3f}, SD {p.delivery_s.std():.3f}; confidence "
          f"{d.confidence.value_counts().to_dict()}")
    return d


def refresh(season):
    """Rebuild delivery_<season>.csv from the detection cache (re-decide only; no video needed)."""
    write([{**measure(a.play_id), "pitcher_id": a.pitcher_id, "pitcher_name": a.pitcher_name, "result": a.result,
            "base": a.base} for a in pool(season).itertuples()], OUT / f"delivery_{season}.csv")


def prefetch(season, workers=4, ahead=10, window=1000):
    """Download a season's clips (and plate-time feeds) ahead of `run`, so the timing never waits on the network and can
    finish offline. Keeps HOME and AWAY clips for the next `window` attempts past the timing loop's position (AWAY is
    needed only when HOME fails QA, ~43% of attempts, but having it ready keeps the run off the network), so the clip
    cache stays bounded at about window x 2 x 6 MB; starts `ahead` attempts past the loop so it never writes the clip the
    loop is reading. Exits once the rest of the season is cached; a clip that fails 3 times is left to the run."""
    from concurrent.futures import ThreadPoolExecutor

    def get_one(pid, feed):
        try:
            mp4, meta = fetch(pid, feed)
            if meta.get("game_pk"):
                plate_time(meta["game_pk"], pid)
            return "ok" if mp4 else "no clip"
        except Exception as e:                     # not cached, so it is retried (here, then by the run)
            return f"error {type(e).__name__}"

    ids, fails, t0, n, mb = pool(season).play_id.tolist(), {}, time.time(), 0, 0.0
    cached = lambda p, f: (FDET / f"{p}_{f}.json").exists() or (CLIPS / f"{p}_{f}.json").exists()
    with ThreadPoolExecutor(workers) as ex:
        while True:
            pos = sum((FDET / f"{p}_HOME.json").exists() for p in ids)          # the run goes in this order
            todo = [(p, f) for p in ids[pos + ahead:pos + ahead + window] for f in ("HOME", "AWAY")
                    if not cached(p, f) and fails.get((p, f), 0) < 3]
            if not todo:
                if pos + ahead + window >= len(ids):
                    break
                time.sleep(30); continue
            for (pid, feed), r in zip(todo, ex.map(lambda a: get_one(*a), todo)):
                n += 1; c = CLIPS / f"{pid}_{feed}.mp4"
                mb += c.stat().st_size / 1e6 if r == "ok" and c.exists() else 0
                if r.startswith("error"):
                    fails[(pid, feed)] = fails.get((pid, feed), 0) + 1
                if n % 100 == 0:
                    print(f"[{season} clip {n}] run at #{pos}/{len(ids)}; {mb:,.0f} MB in {(time.time() - t0) / 60:.0f} min; "
                          f"{(time.time() - t0) / n:.2f} s/clip", flush=True)
    print(f"prefetch {season} done: {n} clips, {mb:,.0f} MB, {(time.time() - t0) / 60:.0f} min", flush=True)


def gold():
    """Score the 50 hand-labelled clips: release / lift-off frame error and delivery-time error (HOME feed frames)."""
    G = pd.read_csv(GOLD)
    rows, t0 = [], time.time()
    for _, g in G.iterrows():
        r = measure(g.play_id)
        rows.append({**r, "gold_lift": pd.to_numeric(g.manual_first_move_frame, errors="coerce"),
                     "gold_release": g.manual_release_frame, "pitcher": g.pitcher_name})
        print(f"[{len(rows)}/{len(G)}] {g.play_id[:8]} {r['qa']:22s} rel {r.get('release_frame')} lift {r.get('lift_frame')} "
              f"delivery {r.get('delivery_s')} ({r.get('seconds')} s)", flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "delivery_gold.csv", index=False)
    ok = d[d.qa == "PASS"]; home = ok[ok.feed == "HOME"]
    e_rel, e_lift = home.release_frame - home.gold_release, home.lift_frame - home.gold_lift   # same mp4 only
    e_del = ((ok.release_frame - ok.lift_frame) - (ok.gold_release - ok.gold_lift)) / ok.fps   # durations: any feed
    print(f"\nwall time {time.time() - t0:.0f} s for {len(d)} attempts | PASS {len(ok)}/{len(d)} "
          f"(HOME {len(home)}, AWAY {len(ok) - len(home)}) | QA: {d.qa.value_counts().to_dict()}")
    for nm, e, u in (("release", e_rel, "fr"), ("lift-off", e_lift, "fr"), ("delivery", e_del, "s")):
        e = e.dropna()
        print(f"  {nm:9s} n={len(e)} MAE {e.abs().mean():.3f} {u}  bias {e.mean():+.3f}  LoA [{e.mean()-1.96*e.std():+.3f}, {e.mean()+1.96*e.std():+.3f}]")
    sd = ok.groupby("pitcher").delivery_s.agg(["count", "std"]).query("count >= 2")
    print(f"  within-pitcher SD max {sd['std'].max():.3f} s (n={len(sd)}) | seconds per clip: median {d.seconds.median():.1f}")
    return d


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd, yr = (a[0] if a else "gold"), (int(a[1]) if len(a) > 1 else 2023)
    if cmd == "season":
        run(yr)
    elif cmd == "pilot":
        run(yr, every=False)
    elif cmd == "prefetch":
        prefetch(yr, int(a[2]) if len(a) > 2 else 4)
    elif cmd == "refresh":
        refresh(yr)
    elif cmd == "gold":
        gold()
    else:
        sys.exit(__doc__)
