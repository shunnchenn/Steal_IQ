# Draft ledger: Pitcher Delivery Time from Broadcast Video — Vision Pipeline Report

Tracked file: `1-Data-Ingestion/vision/REPORT.md` (was `cv/REPORT.md` until the 5 Oct 2026 reorganization)
Current snapshot: `v004`
Code tracked alongside: `1-Data-Ingestion/vision/delivery.py` (was `cv/delivery_time.py` + `cv/delivery_fast.py`)

## BLUF

v001 reports a delivery-time pipeline (BaseballCV + RTMPose, camera-QA gate first) that passes all three
May-pilot gates on the 50 hand-labelled clips: release MAE 1.83 frames, within-pitcher SD ≤ 0.060 s,
coverage 76%. v002 adds a head-to-head test of prior art (Wiedemann et al. 2020; Bair): both lose clearly, and
only their release-frame prior is kept, as a speed-up. v003 records the production design (`delivery.py`, the
consolidated fast pipeline: gold re-detected from an empty cache, identical row for row, delivery MAE 0.048 s) and
the first full seasons: 2023 2,167/3,065 and 2024 2,236/3,114 attempts timed. v004 completes 2025 (2,060/2,997) and
2026 (2,291/3,266) with a two-pass run that downloads only the clips and bytes it reads (identical rows, proven on 2024
and 28 gold clip-feeds). Next: the sub-0.5 s audit and the 2023-only delivery studies on all four seasons.

## Experiment board

### Tried

- BaseballCV detectors (phc v3, glove_tracking v4, ball_tracking v4) on Savant clips — v001 — Achieved
  usable detections, as measured by all 6 CF objects found on the n=1 clip and phc finding the pitcher in
  121/121 frames, by running the downloaded weights with ultralytics on MPS.
- Camera-QA gate first (cuts + CF picture + steady camera) — v001 — Achieved a safety step that refuses
  real camera cuts, as measured by 3/3 cut clips rejected (2 also flagged "No first move" by the labeller)
  and 0 gross errors among 38 PASS, by gating every measurement on `decide()` QA.
- Trajectory-based release (constant-velocity chain + backward extension) — v001 — Achieved release MAE
  1.83 frames (bias −1.83, LoA [−3.30, −0.37]), as measured by `delivery_time.py gold` vs hand labels, by
  replacing speed caps with prediction-error chaining.
- Physics checks on the release (at the hand; tracked to the plate vs Statcast plateTime) — v001 — Achieved
  flight residual mean +0.006 s, SD 0.021 s on PASS, as measured by `flight_resid_s`, by adding the
  `release_not_at_hand` / `flight_too_short` gates.
- Set-anchored, whole-foot, glove-side lift-off; lift_in chosen by leave-one-out — v001 — Achieved lift-off
  MAE 1.70 frames (bias −0.03; LOO MAE 1.7), as measured by `delivery_time.py tune`, by measuring the lead
  foot's lowest keypoint after a 2-D-still set, with handedness from `Raw_Attempt_Context.csv`.
- SAHI-style corridor slice for the ball — v001 — Achieved recovery of wide-shot clips (e.g. b39330a9: a
  wrong event 76 frames late → release −1 frame vs label), as measured by the targeted re-detection, by
  running the ball model on a ≥ 640 px pitcher–catcher slice.
- Prior-art search for public delivery-time / slide-step CV (2026-10-01, no report version) — Achieved
  a negative result: no public repo measures delivery time with validated accuracy. Measured by GitHub
  searches (8 queries: none relevant) and a read of the closest work, Wiedemann et al. 2020 (arXiv
  2003.03856), code `NinaWie/pitch_type`: last push 2018, no license; pitcher first movement validated
  only indirectly, because Statcast first-move labels were "very unreliable". Savant's Pitcher Running
  Game leaderboard publishes lead distance gained (first move → release) but no time column.
- Wiedemann + Bair combination, head-to-head on 30 labelled PASS clips — v002 — Achieved a clear negative
  result, as measured by `cv/experiments/prior_art.py`: leg-motion lift-off MAE 5.73 vs pose 1.70 frames;
  wrist-height release MAE 4.80 / 4.93 vs ball flight 1.83. Done by re-implementing both papers' event
  detectors from their descriptions on cached clips and boxes.
- Release-frame prior (Bair 175–183; Wiedemann "≈ frame 93") — v002 — Achieved a usable search prior, as
  measured by 97% of 66 detected releases in frames 165–215 (Bair's window alone: 71–76%), by tabulating
  `rel_process` from the det cache.
- Smaller ball detector, stock COCO YOLO11s / YOLO11n (2026-10-02, no report version) — Achieved a fast
  but failing swap, as measured by `cv/experiments/small_ball_model.py` on 5 labelled PASS clips
  (`cv/out/small_ball_model.csv`). Ball stage 26.2 → 6.5 s (s) / 4.4 s (n) per clip. QA PASS 5/5 → 1/5 (s)
  / 0/5 (n), mostly flight_too_short: release found (|release − label| 1.2 / 1.0 frames vs 1.8 current)
  but the ball is lost far from the camera before the plate.
- Minimal "two frame numbers" design, `cv/delivery_fast.py` (2026-10-02, no report version) — Achieved
  8.4× less compute at near-equal accuracy, as measured on the 50 labelled clips (HOME feed only): median
  6.4 s per clip (vs ~54 s); PASS 32/50 (vs 30 HOME); release MAE 1.91 (vs 1.83), lift-off 1.94 (vs 1.70),
  delivery 0.046 s (vs 0.041). Done by running one anchor frame, the ball detector only in an upper-body crop
  over the release prior (every 4th frame, then full rate around the hit), pose only on [rel−1.75 s,
  rel−0.35 s], Core ML detectors and pose, and verifying release by the at-the-hand check instead of
  tracking the flight to the plate. Same-pitcher SD max 0.107 s vs 0.083 s in the hand labels on the same
  clips (the gate sits near real pitcher variation); one real error: Keller b39330a9 release +5 frames.
- Fast design, fixed (2026-10-02) — Achieved all 3 May gates at 6.6× less wall time, as measured by
  `cv/delivery_fast.py` on the 50 labelled clips from an empty cache: 565 s total (vs 3,726 s), median 8.4 s
  per attempt incl. downloads and AWAY retries; PASS 38/50 (76%; HOME 33, AWAY 5); release MAE 1.73 frames;
  same-pitcher SD max 0.095 s (13 pitchers); delivery MAE 0.048 s (0.042 without 26b50bd5). Done by
  (1) running the ball search on the native 640 px corridor slice instead of the upper-body crop, which
  was upscaled ×1.27 and lost the first ball frames (Keller b39330a9: release 186 → 180, label 181), and
  (2) adding the AWAY-feed retry. Outlier 26b50bd5 (AWAY, −0.251 s): frames show Castillo still at the
  labelled first move (f98–99) and the foot leaving at f114 = auto, so it is a label disagreement.
- Ball search on a smaller upper-body crop — 2026-10-02 — upscaling a 505×275 crop to 640 lost
  borderline early ball detections, and gave no speed gain (one 640 px inference per frame either way).
- 2023 pilot, one SB/CS attempt per pitcher (621 pitchers, seed 0), fast design (2026-10-02) — Achieved
  delivery times for 442/621 pitchers (71%) in 99 min (9.6 s/attempt incl. downloads), as measured by
  `cv/delivery_fast.py pilot 2023` → `cv/out/pilot_2023.csv` (picks: `pilot_2023_ids.csv`). Funnel:
  no_release 52, cut_in_window 36, release_not_at_hand 28, no_set_before_lift 23, no_clip 21, not_cf_view 19;
  PASS on HOME 353 / AWAY 89; 2B 71%, 3B 74%. Delivery mean 0.992 s, SD 0.126, range 0.452–1.408.
  Validity: vs Statcast gain_to_release_ft r = 0.41 (p = 3e-19, n = 438), slope 13.1 ft/s (11.0 ft/s
  controlling for runner sprint speed, n = 247). SB 1.002 vs CS 0.961 s (Welch p = 0.007, d = 0.32).
  Unverified: 2 deliveries < 0.5 s (Cordero 0.452, Houck 0.485), below every hand label (min 0.617).
- Confidence column for PASS rows (2026-10-02) — Achieved a validation triage, as measured by the 2023
  pilot splitting into 384 high / 58 low / 179 not measured (`pilot_2023_links.xlsx`, green / amber / grey).
  Done by adding 3 signals to `delivery_fast.decide` (lift_peak_in, ball_pts; release_to_wrist already
  there) and labelling a row high only when all 4 lie inside the range seen on the 37 gold PASS clips.
  Finding: inside that envelope the signals do NOT predict error (|Spearman| ≤ 0.27, n = 38), so it means
  "verified territory", not a calibrated probability. Low reasons: release_to_wrist 27, lift_peak_in 20,
  delivery_s 6, ball_pts 5; Cordero and Houck are low (foot rise 3.92 / 2.05 in).
- Staged reproducibility n=1 → n=5 → batch — v001 — Achieved identical rows, as measured by a column-by-
  column comparison of `delivery_n1/n5/batch7.csv`, by caching detections and keeping `decide()` pure.
- 8 GB memory hardening (stream frames, `torch.mps.empty_cache`, crop instead of a 1280 pass) — v001 —
  Achieved ~54 s per clip-feed with swap stable, as measured by the `timing` blocks, by not holding frames.

- Full-season runs with the fast design (2026-10-03 to 10-05, v003) — Achieved delivery times for 2,167 of 3,065
  (2023) and 2,236 of 3,114 (2024) SB/CS attempts, as measured by `data/delivery/delivery_2023.csv` and
  `delivery_2024.csv` (QA PASS 71% / 72%), by running `season <yr>` resumably; 2024's clips were prefetched on
  Wi-Fi first (`prefetch 2024`: HOME then AWAY, ~0.7 s/clip, 4 threads) so the timing finished offline.
- Consolidation into `vision/delivery.py` (2026-10-05, v003) — Achieved one production file with unchanged output,
  as measured by identical `decide()` rows on all 6,297 cached clip-feeds and an identical gold re-detection from an
  empty cache (50 clips, processing time aside), by keeping the fast path and the v1 functions it calls and archiving
  the rest of `delivery_time.py` (slow path, `tune`, `sheet`) at `4-Archive/code/cv/`.

### Ruled out

- Pose-based release (peak throwing-hand speed) — May pilot — latches onto the catcher's throw after a
  cut; release MAE 132.75 frames.
- Absolute ball-speed cap to separate flight from hand-carry — v001 — camera-dependent; rejected a real
  16 px/frame flight (left-hander clip 3e34f93d).
- Greedy longest-chain association — v001 — tie-break artifact chose an in-hand point (n=1, f178).
- Chaining the 1280/crop detections forward — v001 — in-hand points broke the flight (gold crash,
  836d520f); crop detections are used only for backward extension.
- Speed/turn-based catch detection, and net-displacement "moved" filter — v001 — the pitch's image path
  is an arc (apex pause ~4 frames, then reversal); falsely cut Houck/Castillo runs at half the flight.
- Single-frame shot classification; rubber or home plate required in every frame — v001 — false rejects
  (feet hide the rubber at conf 0.03–0.07; the batter hides the plate in wide shots).
- Home-plate box width as a zoom/PiP cue — v001 — the box swallows the batter's shoe (26 → 59 px).
- 2.0 s uncut-window rule only — v001 — false-rejected a cut 0.87 s before lift-off (Ureña).
- Lift relative to a cut without a still set — v001 — accepted a mid-stride window (Feltner, +52 frames).
- Heel-rise lift-off (mean ankle+heel, 0.5 in) — v001 — earlier than "foot leaves the ground" by 4–8
  frames on slow heel peels.
- Vertical-only stillness for the set — v001 — a slide step's level slide read as a second set → pivot
  foot chosen (Keller, +22 frames).
- open-command data as release ground truth — v001 — clips are release-centred (release_s within
  ±0.0084 s over 3,946 rows), so absolute release frames are unrecoverable.
- Installing the `baseballcv` package — v001 — import auto-installs a git fork; pins conflict with the
  user's Trajekt venv; weights-only instead.
- Leg-motion (frame-difference) lift-off as method or cross-check — v002 — MAE 5.73, worst 33 frames;
  disagreement with pose does not flag pose errors (pose MAE 1.27 when they disagree vs 2.88 when they agree).
- Wrist-height / arm-above-shoulder release as method or no-ball fallback — v002 — MAE 4.8–4.9, worst 13
  frames; fails the ≤ 2-frame gate, so it can't stand in when the ball is missed.
- Bair's fixed 175–183 release window as a hard window — v002 — misses 24–29% of releases.
- Stock COCO small model as a drop-in ball detector — 2026-10-02 — loses the small far-flight ball
  (1/5 and 0/5 PASS); usable only for the near-hand segment.
- Batch > 1 / fp16 on MPS — v001 — no gain; batch 16 swaps (3.8 s/frame) on 8 GB.

### Not tried

0. Kaggle GPU 20-clip test (IN PROGRESS 2026-10-01): prompt `cv/KAGGLE_PROMPT.md`, bundle
   the Kaggle upload zip in `cv/cache/kaggle/` (since deleted). Code delta (no report change): `delivery_time.py` device was
   hard-coded `mps` (YOLO) / `cpu` (RTMPose) → auto `cuda` > `mps` > `cpu`, RTMPose on CUDA when available;
   evidence: n=5 rows byte-identical on the Mac after the edit. Also measured: Core ML ball detector 78 vs
   448 ms/frame (5.7×), 134/140 identical decisions at conf 0.15 — not yet wired in.
0a. Audit the 2 sub-0.5 s deliveries (consider a 0.55 s floor). Full runs: all of 2023–2026 done (v004).
1. Speed-up package (measured 2026-10-01 on the 8 GB M2, not yet wired in). Core ML for phc 85→42 ms,
   glove 139→64 ms, ball 448→78 ms; RTMPose on the onnxruntime CoreML provider 18→5.3 ms (model only).
   Decode is 0.9 ms/frame (not a bottleneck). The crop pass is NOT redundant (removing it changes the
   release on 5/66 clip-feeds). Plan, largest first:
   (a) Core ML for all 3 detectors; (b) narrow the coarse ball band to ≈ frames 160–245 (release prior);
   (c) track, don't re-detect, in the fill pass — small crop around the constant-velocity prediction;
   (d) pose on the CoreML provider + stride-2 pose with local refill; (e) glove model on 3 of the 10 QA
   samples (it only serves plate/rubber); (f) 2 worker processes (pose on CPU/Neural Engine alongside
   detection). Target ≈ 10 s per clip-feed vs 53.8. Each step must re-pass the 50-clip gold gates
   (Core ML changes 6/140 borderline ball calls).
   (g) small-model options after the 2026-10-02 test: hybrid (YOLO11s for the coarse pass and near-hand
   frames, the big model only for the far half of the flight), or distil YOLO11n/s on the big model's
   pseudo-labels from the cached clips (needs a training run; then re-run the 5-clip test, then gold).
2. 500-clip random pilot (`delivery_time.py pilot 500`) → real funnel and coverage.
2. Release convention decision (literal first-on-trajectory vs +2 frames to match hand labels).
3. Full run over 11,140 SB/CS attempts (`all`), or a pitcher-stratified subset.
4. Join `delivery_s` to `Raw_Attempts` and test in `model/engines.py`: VIF/BKW vs `gain_to_release_ft`,
   missingness bias, grouped-CV AUC gain.
5. Coverage: fine-tune `ball_tracking_v4` on missed wide-shot / curveball frames.
6. Physics-assisted release for occluded starts, reported as a separate method.
7. Second labeller and ~150 random gold clips → inter-rater ceiling.

## Version log

### v001 — 2026-10-01

**XYZ:** Achieved a complete engineer-facing report, as measured by `cv/REPORT.md` containing the gate
table, agreement table (MAE/bias/LoA/bootstrap CI), reliability (ICC 0.94), QA funnel, engineering metrics
and a 5-step fitting procedure, every figure re-derived from `cv/out/eval_gold.csv` and the det cache. Done
by writing the draft from the final fresh gold run (lift_in 0.75) and correcting two drafted claims
against the data before the snapshot: max delivery error 0.084 s (not 0.101), and max lift-off error 6
frames (not 4).

**Follows:** nothing (initial). Content follows the chat report of the same date. The numbers supersede
it where the threshold changed (lift_in 0.5 → 0.75).

**Delta vs previous:** initial capture.

**Code delta (history of `cv/delivery_time.py` up to v001):**
- files: `cv/delivery_time.py` (new), `cv/gold/labels_manual.csv` (restored from git `8b0eb4a^`),
  `.gitignore` (+`cv/weights/`, `cv/cache/`)
- previous behavior: none in the repo. The deleted May pilot (`Computer Vision/CV Detection Pipeline/
  extract_delivery.py`) used YOLOv8-pose release and an HSV-histogram cut guard.
- new behavior, in build order (each step forced by a failing test):
  1. fetch + QA + release + lift (n=1). Tie-break fix (nearest-in-time association).
  2. `html.unescape` on the Savant src (33-byte CDN error). CF test without required rubber; 3-probe shots.
  3. Trajectory chaining + backward extension (left-hander late release). Plate-time QA.
  4. Lenient home plate + steady-camera test (false not_cf_view).
  5. Stream frames, MPS cache release, native crop instead of a 1280 pass (8 GB swap).
  6. Catch-anchored coarse→fine fill. Detection floor 0.05 cached, threshold at decide time; gap 5 with
     widening tolerance.
  7. Chains on the 640 pass only; crop for backward extension. `rel_process` stale-cache guard.
  8. Lift-relative cut rule; then a set-anchored, whole-foot, 2-D-still, glove-side lift-off;
     implausible-delivery guard.
  9. Rest-based catch + Statcast truncation; release-at-hand + one-sided flight QA; reach filter.
  10. SAHI corridor slice for the ball. `tune` (leave-one-out lift_in → 0.75). `funnel`, `pilot`, `all`.
- unchanged: v15 model, report and site; `Data/*` inputs; no commits.
- evidence: `cv/.venv/bin/python cv/delivery_time.py gold` → PASS 38/50, release MAE 1.833, lift MAE
  1.700, within-pitcher SD max 0.060 s; `... tune` → LOO MAE 1.7.

**Board moves:** initialized: 8 Tried, 14 Ruled out, 7 Not tried (above).

**Not done:** all items under Not tried; nothing committed.

### v002 — 2026-10-01

**XYZ:** Achieved a tested answer to "combine Wiedemann et al. and Bair", as measured by a 5-method table on
the 30 labelled PASS clips: current pose lift-off 1.70 vs leg-motion 5.73 frames MAE; current ball release
1.83 vs wrist-height 4.80 and arm-above-shoulder 4.93. Done by adding `cv/experiments/prior_art.py`
(re-implementations, no copied code; both repos unlicensed) and a prior-art bullet and table in §8, plus a
speed note in §9.

**Follows:** v001 §8 ("Decisions and what was ruled out") and §9 step 1 (pilot). v001 already ruled out
pose-based release as the PRIMARY method (the May failure); v002 tests it as a cross-check or fallback,
which v001 had not done.

**Delta vs previous:**
- §8: + prior-art bullet with the head-to-head table, the cross-check finding, and the release-frame prior
  (97% in frames 165–215; Bair's window 71–76%).
- §9 step 1: + narrow the coarse ball band before the pilot (est. ~40% off that stage) and combine with
  Core ML / a GPU.

**Code delta:**
- files: `cv/experiments/prior_art.py` (new, experiment only); `cv/delivery_time.py` unchanged.
- previous behavior: no prior-art comparison existed.
- new behavior: replays the cached gold clips → `cv/out/prior_art.csv` plus the comparison table.
- unchanged: the pipeline and its outputs (a method that loses is not wired in).
- evidence: `cv/.venv/bin/python cv/experiments/prior_art.py` → the table above, n = 30.

**Board moves:**
- Tried: Wiedemann + Bair head-to-head; release-frame prior.
- Ruled out: leg-motion lift-off; wrist-height release (method and fallback); Bair's hard 175–183 window.
- Not tried added: narrow coarse ball band (now item 1).

**Not done:** Kaggle test; band narrowing (not implemented); pilot; full run; model integration.

### v003 — 2026-10-05

**XYZ:** Achieved a report that matches the production pipeline and the repository, as measured by `REPORT.md`
gaining a Status section whose every number is re-derived from `data/delivery/delivery_gold.csv` (re-detected from an
empty cache by `delivery.py gold`) and `delivery_2023.csv` / `delivery_2024.csv`, by moving the report into
`1-Data-Ingestion/vision/` with the consolidated code and recording the first full seasons.

**Follows:** v002 (the v1 report and its prior-art test) plus the ledger's un-versioned fast-design entries of
2 October; v003 is the first version written after the fast design became the production pipeline.

**Delta vs previous:**
- + "Status (v003)" section after the BLUF: production file and layout, the fresh-cache gold metrics, the
  consolidation evidence, 2023 / 2024 / 2026 coverage, run commands.
- + a note that §1–9 are the v1 report and refer to the old layout; last line points to this ledger in place.
- No v1 number changed.

**Code delta:**
- files: `cv/delivery_time.py` + `cv/delivery_fast.py` + `cv/prefetch.py` → `1-Data-Ingestion/vision/delivery.py`;
  outputs `cv/out/all_<season>.csv` → `1-Data-Ingestion/data/delivery/delivery_<season>.csv` (same rows + `confidence`,
  `why`); the colour-coded `_links.xlsx` export dropped (the meta tables carry the video link per row).
- previous behavior: `delivery_fast.py` imported `delivery_time.py` and monkeypatched its `detect` inside `scene()`.
- new behavior: one module; `scene()` calls the Core ML `detect` directly at 800 px (what the monkeypatch did).
- unchanged: every threshold (`CFG`, `FAST`, `VERIFIED`), the QA rules, the outputs.
- evidence: `decide()` identical on 6,297 cached clip-feeds; gold re-detection from an empty cache identical row for
  row; `confidence()` identical on every 2023 and 2024 row (vs the old links export).

**Board moves:**
- Tried: full-season runs 2023 and 2024; consolidation into `vision/delivery.py`.
- Not tried: 0a narrowed to the 2 sub-0.5 s audit plus 2025 and the rest of 2026.

**Not done:** 2025 and 2026 runs; the sub-0.5 s audit; Kaggle test; band narrowing.

### v004 — 2026-10-06

**BLUF:** 2025 and 2026 are timed (2,060/2,997 and 2,291/3,266), and the season run now downloads only what it reads.

**XYZ:** Achieved all four seasons timed, as measured by `delivery_2025.csv` (69% PASS, 0 errors after one retry) and
`delivery_2026.csv` (70% PASS, 0 errors), by restructuring `run()` into a HOME pass and an AWAY pass with downloads inside
the run, fetching only the moov plus the first 330 frames of each clip, recording failed downloads once, and moving
the cache off the iCloud-synced Desktop.

**Follows:** v003 (production `delivery.py`, 2023 and 2024 timed; 2025 and 2026 left).

**Delta vs previous:**
- Status section: v004 title; 2025 and 2026 rows replace "2026: 75 processed"; two evidence rows (two-pass rows
  identical on 2024; partial downloads identical on 28 gold clip-feeds); a "How a season runs" paragraph; the run
  commands drop `prefetch`; the cache location.
- §1–9 unchanged.

**Code delta (`delivery.py`):**
- previous behavior: `run()` timed attempt by attempt (HOME, then AWAY on a fail), fed by a separate `prefetch`
  process that downloaded both feeds of every attempt; `fetch()` downloaded whole clips; a failed download was retried
  by the run and again at the end; `plate_time()` wrote its cache in place.
- new behavior: `run()` makes two passes (HOME for all, then AWAY where HOME failed QA) with 4 download threads inside
  it (200 clips ahead; idle threads fetch later-needed clips: known AWAY clips, then the next season's HOME); rows are
  assembled from the cache at the end without the network. `_get_clip()` downloads the moov and the bytes of the
  first `PREFIX_FRAMES` = 330 video samples (from stsz/stsc/stco) into a full-size file, falling back to the whole
  clip on any surprise. A download that fails 4 times is recorded once as `error: <type>`. `plate_time()` writes
  atomically. `prefetch` is kept for offline timing.
- unchanged: `process()`, `decide()`, every threshold, the QA rules, the output columns.
- evidence: 2024 rebuilt from the cache by the new `run()` — 3,114 rows identical to `delivery_2024.csv`, 0 downloads;
  28 gold clip-feeds — raw detections and rows identical between the partial and the full file, and identical on two
  runs of the full file; 3 server-error clips recorded once in 17 s (were about 36 s each).
- also: the cache moved to `~/stealiq-vision-cache` (`vision/cache` links to it); the project renamed StealIQ (paths
  in this ledger's snapshots updated, no content change).

**Board moves:**
- Tried: full-season runs 2025 and 2026; two-pass run; partial clip downloads; cache off iCloud.
- Not tried: 0a now only the sub-0.5 s audit.

**Not done:** the sub-0.5 s audit; the 2023-only delivery studies (report §6.1–6.3) on all four seasons; Kaggle test.

