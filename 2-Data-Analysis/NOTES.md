# Notes for later: a speed-vs-success view for the live web app

Not built yet. This is the design for a dashboard panel that answers "do faster runners steal more successfully?"
with the sample size shown on screen, so a significant-but-small effect (the Powers et al. speed term) cannot be
mistaken for a large one. The static version is `figures/1-Paper-GLMM/Fig_paper_speed.png` (steals of second, made by
`stealiq.py paperfigs`) and the paper's §4.5; the pooled version is `figures/5-Public-Baselines/Fig_cmp_speed_by_season.png`.

## What the evidence says (the panel must make this visible, not just state it)

From `results/5-Public-Baselines/DF_cmp_speed_by_season.csv`, made by `stealiq.py speed` on the shipped calculator's 12,165 rows:

- Speed is **significant** pooled over 2023–26 (+0.12 log-odds per ft/s, p < 0.001), and in 3 of 4 single seasons.
- Speed is **small**. From a 10th- to a 90th-percentile runner, the share safe rises 4–6 points; ground gained moves it 29–32.
  Speed alone ranks attempts at AUROC 0.53–0.55, against 0.71–0.73 for ground gained.
- Across runners, the correlation between a runner's speed and their success rate is r = +0.08 to +0.14 each season
  (runners with 10+ attempts), and every season's CI includes 0. Pooled over 450 runner-seasons it is +0.11 (CI +0.02 to +0.20).
- Why both statements hold: with about 3,000 attempts a season, a 5-point difference is detectable. With 30 attempts,
  a single runner's rate is uncertain by ±14 points, and that noise is larger than speed's whole effect.

## Controls

- Speed range (ft/s slider) and grouping. Equal-count quantile groups are the default because every group then has
  the same precision; 0.5 ft/s bins are an option.
- Season(s), base (2B / 3B), pitcher hand, outs, and a minimum number of attempts per runner.
- Unit of analysis: attempts pooled, or runners (each runner's rate, shrunk; see below).

## What each group shows

- SB and CS counts overlaid, in the same bins and two colours, with the number of runners above the bar (as in the figure).
- Share safe with a **Wilson 95% interval**. Groups under 20 attempts are drawn hollow.
- Attempts per runner. A few high-volume runners can carry a bin.

## Accounting for volume

- **Shrink each runner's rate toward the league** (beta-binomial empirical Bayes):
  `(SB + a) / (attempts + a + b)`, with a and b fitted to the league's runner-season rates by method of moments.
  This stops 9-for-10 from outranking 45-for-55.
- **Minimum-detectable-difference readout** (normal approximation, two-sided α = 0.05, 80% power, league rate p = 0.807).
  For the current selection, show "observed gap X pts; detectable at this n: Y pts".

  | attempts per group | smallest detectable gap between two groups | 95% half-width of one group's rate |
  |---:|---:|---:|
  | 30 | 28.5 pts | 14.1 pts |
  | 100 | 15.6 pts | 7.7 pts |
  | 500 | 7.0 pts | 3.5 pts |
  | 1,000 | 4.9 pts | 2.4 pts |
  | 3,000 | 2.9 pts | 1.4 pts |

  Inverse: a 5-point gap needs about 980 attempts per group; a 2-point gap about 6,100.
- **Correlation power** (Fisher z). Detecting r = 0.10 between runner speed and success needs about 780 runners.
  One season has about 110 runners with 10+ attempts, so it can only detect r ≥ 0.26; 450 runner-seasons can detect r ≥ 0.13.
  This is why each season's runner-level r has a CI that includes 0 while the pooled one barely excludes it.
- Formulas: `MDD = (z_α/2 + z_β) · sqrt(2p(1−p)/n)`, `n = 2p(1−p)(z_α/2 + z_β)² / d²`, and runners for r = `((z_α/2 + z_β)/atanh r)² + 3`.

## Caveats to print next to it

- These are attempts actually made. The fastest speed quintile attempts about 7 times as often as the slowest
  (4.8% of opportunities against 0.7%; `results/3-Engines-v12-v13/DF_v13_SelectionEffect.csv`, report §8), so success by speed is
  conditional on choosing to run. Show the attempt rate per opportunity (`Raw_Opportunities.csv.gz`)
  beside the success rate, to separate who runs from who succeeds.
- Faster runners take less ground before release. With ground gained held fixed, speed is worth about three times more
  (report §4.3). A toggle can switch between the raw view and the model's effect at fixed ground.
- League success fell every season, from 82.8% in 2023 to 78.9% in 2026. Compare speed groups within a season, or adjust for season.
- Sprint speed is the season's Statcast sprint speed (top two-thirds of competitive runs), not the speed on that attempt.
- Attempts are SB + CS only. PK, BK and FB rows are running events with no pitch, so they are excluded.

## Data and implementation

- Source columns in `1-Data-Ingestion/data/meta/meta_<season>.csv`: `season`, `runner_id`, `runner_name`,
  `runner_sprint_speed`, `base`, `result`, plus `video_url` for clicking through to a play.
- The panel can start from `results/5-Public-Baselines/DF_cmp_speed_bins.csv`: per season × 0.5 ft/s bin, it holds attempts,
  SB, CS, runners and the Wilson CI.
- The site runs without a server (GitHub Pages). Ship per-runner-season counts as a small JSON next to `v15_players.json`,
  and compute pooling, Wilson intervals, shrinkage and MDD in the browser. The counts are a few hundred rows a season.
