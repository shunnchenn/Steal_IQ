# 1 · Data Ingestion

Every byte of data in the project enters here and leaves as **one table per season**:
`data/meta/meta_<season>.csv`, one row per tracked running event (every SB and CS attempt, plus the drawer's other running
events), with every Statcast field of that pitch, the MLB feed's context, the season's catcher pop time and runner sprint
speed, the pitcher's delivery time measured from video, links to the video, and a `qa_flags` column. The models in
`2-Data-Analysis/` read only these tables plus the season-level raw tables, so what you QA here is exactly what they see.

```
1-Data-Ingestion/
├── ingest.py            the one scraper and builder: scrape → data/raw → data/meta   (python3 ingest.py -h)
├── vision/
│   ├── delivery.py      pitcher delivery time from Savant broadcast video → data/delivery/
│   ├── REPORT.md        the CV pipeline: definitions, accuracy on the gold clips, history (+ ledger, versions)
│   ├── gold/            labels_manual.csv: the 50 hand-labelled clips
│   └── weights/ cache/ .venv/      detector weights, clip + detection cache, the CV environment (gitignored);
│                                   cache/ is a link to ~/stealiq-vision-cache, off the iCloud-synced Desktop
└── data/
    ├── raw/             committed source tables (immutable inputs; rebuilt only by ingest.py)
    ├── meta/            meta_2023.csv … meta_2026.csv + data_dictionary.csv   ← open these for QA
    ├── delivery/        delivery_<season>.csv (CV timing runs) + delivery_gold.csv (accuracy check)
    └── cache/           request caches: Savant drawer, MLB feed, Statcast per game (gitignored, regenerable)
```

## The meta tables

Open `data/meta/meta_<season>.csv` in any spreadsheet. The first columns are what a QA pass needs, in this order:
identity and links (`season`, `date`, `game_pk`, `play_id`, `video_url`, `video_url_away`, `gamefeed_url`, `qa_flags`),
the game state before the pitch (`inning`, `inning_topbot`, `outs_when_up`, `balls`, `strikes`, `on_1b/2b/3b`, score),
the runner and the attempt (`runner_name`, `base`, `result`, `lead_at_firstmove_ft`, `gain_to_release_ft`,
`runner_sprint_speed`), the pitcher (`pitcher_name`, `p_throws`, `cv_delivery_s`, `cv_qa`, `cv_confidence`), the catcher
(`catcher_name`, `catcher_pop_2b`, `catcher_arm_mph`), the batter and the pitch (`batter_name`, `pitch_type`,
`release_speed`, `description`, `events`); then every other Statcast field, the remaining CV detail (`cv_*`, including
`cv_lift_s` / `cv_release_s`: where in the clip the delivery starts and ends) and the MLB feed's own fields (`feed_*`).
**`data/meta/data_dictionary.csv` defines every column** (Statcast fields with Savant's own definitions from
baseballsavant.mlb.com/csv-docs) and lists the columns dropped.

| Join | Source | Key | Rule |
|---|---|---|---|
| base table | Savant base-stealing drawer, per runner-season (`raw/Raw_Attempts.csv`) | `play_id` + `runner_id` | every row kept |
| pitch | Savant Statcast search CSV, per game (`raw/Raw_Statcast_Pitches.csv.gz`) | `play_id` → (`game_pk`, `at_bat_number`, `pitch_number`) via the MLB feed | left join |
| context | MLB StatsAPI play-by-play (`raw/Raw_Attempt_Context.csv`) | `play_id` | left join; columns prefixed `feed_` |
| season traits | Savant sprint-speed and pop-time leaderboards | (`runner_id`, `season`), (`catcher_id`, `season`) | left join |
| delivery time | `data/delivery/delivery_<season>.csv` (vision/delivery.py) | `play_id` | left join; columns prefixed `cv_` |
| names | MLB StatsAPI people (`raw/people.csv`) | `runner_id`, `batter` | left join |

Joins are asserted not to add or drop a row. **Dropped:** Statcast columns Savant has retired that are empty for every
attempt pitch (listed in the dictionary). Nothing else is filtered here; the models filter to SB/CS themselves.

**`result`:** SB stolen base, CS caught stealing. PK, BK and FB are running events that did not happen on a pitch
(checked on 12 sampled plays per code against the MLB feed: PK on a pickoff attempt, BK on a balk, FB on a disengagement,
i.e. a pickoff attempt or a step-off), so they have no Statcast row.

**`qa_flags`:** `ok`, or the checks this row fails: `no_statcast_row`, `no_feed_context`, `pitcher_mismatch` (Statcast's
pitcher differs from the drawer's), `catcher_mismatch`, `runner_not_on_base` (Statcast does not show the runner on the
base he left), `date_mismatch`, `pitch_type_mismatch` (mostly pitchouts: Statcast labels them `PO`).

**Two fields never to use as model inputs:** `feed_balls` / `feed_strikes` are the count *after* the pitch and
`feed_score_diff` is the score at the *end* of the plate appearance. Statcast's `balls`, `strikes`, `bat_score` and
`fld_score` are the pre-pitch versions.

## Running it

```bash
python3 1-Data-Ingestion/ingest.py meta          # rebuild the meta tables from data/raw (offline, ~1 min)
```

**Adding a season** (or refreshing the current one) — every step is cached and resumable, and nothing in the folder
layout changes:

```bash
python3 1-Data-Ingestion/ingest.py discover --start 2027 --end 2027 --expand   # who steals + their per-attempt drawer
python3 1-Data-Ingestion/ingest.py sprint --end 2027                          # season tables
python3 1-Data-Ingestion/ingest.py poptime --end 2027
python3 1-Data-Ingestion/ingest.py league --end 2027
python3 1-Data-Ingestion/ingest.py build --refresh 2027                       # Raw_Attempts.csv, Raw_Season.csv
python3 1-Data-Ingestion/ingest.py context                                    # MLB-feed context per attempt
python3 1-Data-Ingestion/ingest.py statcast --stop-at 07:45                   # every Statcast field per attempt pitch
python3 1-Data-Ingestion/ingest.py people
python3 1-Data-Ingestion/ingest.py opportunities                              # decision-model denominator
python3 1-Data-Ingestion/ingest.py meta
```

**Delivery times from video** (the CV environment lives in `vision/.venv`; about 6 MB of video per clip):

```bash
1-Data-Ingestion/vision/.venv/bin/python 1-Data-Ingestion/vision/delivery.py season 2025     # time a season (resumable)
1-Data-Ingestion/vision/.venv/bin/python 1-Data-Ingestion/vision/delivery.py gold            # accuracy on the 50 gold clips
python3 1-Data-Ingestion/ingest.py meta                                                     # fold the new times in
```

`delivery.py season` times the home broadcast of every attempt, then the away broadcast where the home one failed QA
(about 43%), downloading only those clips while it works, and writes `data/delivery/delivery_<season>.csv` at the end. To
time offline later, `delivery.py prefetch 2025` downloads both broadcasts ahead. The analysis uses a season once at
least 95% of its SB/CS attempts are processed. The pipeline, its accuracy (about 48 ms mean absolute error against the
hand labels) and its history are in `vision/REPORT.md`.

## Sources

- **Baseball Savant**: the base-stealing running-game drawer (per attempt: leads, result, run value), Statcast search
  (every pitch field), sprint speed, catcher pop time, broadcast video (`sporty-videos?playId=`), the CSV field docs.
- **MLB StatsAPI**: season SB/CS (year-correct; Savant's leaderboard export ignores its year parameter), play-by-play
  (context, base state, play_id → pitch key), names, team totals.
- **BaseballCV** (github.com/dylandru/BaseballCV) detector weights and **RTMPose** (rtmlib) for the video pipeline.
