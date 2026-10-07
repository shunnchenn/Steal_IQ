# Stolen-base models: the literature, and how StealIQ compares

The academic standard is Powers et al.'s mixed model, and it reproduces on this project's data: same weights, same
2.5% deviance explained. Nearly everything that separates the paper's success model (AUROC 0.791) from it (0.603) is
one input it did not have, ground gained by release. The industry models (Statcast, TruMedia) use inputs we do not yet:
pitch velocity, pitch location and pitchouts, and the runner's position when the pitch reaches the plate.

Reviewed 7 October 2026, starting from the references of Powers et al. (arXiv 2601.15608 v2) and theirs. Every source
was read at its own text, code or PDF. Numbers marked *ours* come from `python3 2-Data-Analysis/literature.py` and
`results/1-Paper-GLMM/DF_paper_forward_all.csv`.

## 1. The field

Models of a steal's success before or at the pitch (academic):

| Model | Data | Method | Inputs | Out-of-sample accuracy |
|---|---|---|---|---|
| Loughin & Bargen (2008), *J. Sports Sci.* 26(1) | 1978–90 first innings, ~9,000 attempts | mixed-effects logistic | pitcher and catcher effects (no runner), game state | none |
| Stanley (2023), U. South Carolina senior thesis | before 2023 | logistic, random forest | sprint speed, catcher arm, game state; no player identities | not read (full text not retrievable) |
| Powers, Ramani, Hahn & Schaefer (2026), arXiv v2 | 2022–23, steals of second | GLMM, runner/pitcher/catcher effects | lead at first move, sprint speed, catcher arm, year | 2.5% deviance explained (20-fold CV) |
| Melville et al., *The Pitcher's Dilemma* (MIT Sloan; 42 Analytics) | Statcast 2016–25 | logistic | lead, sprint speed, runner's average jump speed, pop time, post-2023; a pitch version adds velocity and location × batter side | none |
| Whitney-Epstein, Sissman & Dodson (2025), *Wharton Sports Analytics Journal* | 2024, pitches taken | logistic | lead, pitcher "Threat" (Savant net bases prevented per 100 IP), pop time, sprint speed | none |
| Price (2026), Harvard senior thesis | 2021–25 | hierarchical logistic | count, outs, inning, score, disengagements, sprint speed, pitch flight time | abstract only |

Models that grade a steal with what happens during it (industry):

| Model | Measured at | Inputs | Accuracy published |
|---|---|---|---|
| Statcast Caught Stealing Above Average (Petriello, 2023) | the pitch crossing the plate | runner's distance from the base, runner speed, pitch location, both hands, pitchouts and delayed steals | no |
| Statcast running game (pitchers) and basestealing run value (runners) | each opportunity | "most notably the speed of the runner"; non-attempts valued on a sliding scale | no |
| TruMedia expected stolen bases | separate models for second and third, 2016–23 | primary and secondary lead, sprint speed, first step, steal time, pitch release time, pitch velocity, pitchout | no |
| Baseball Prospectus SRAA / TRAA (Pavlidis & Judge, 2016) | season | mixed model with runner, pitcher and catcher effects | no (checked at summary level only) |

Around them: Downey & McGarrity (2015, 2019) and Turocy (2014) treat the pickoff and the steal as a game; Lindsey (1963)
and Tango, Lichtman & Dolphin (2007) supply the run-expectancy break-even. Clemens (FanGraphs, 9 December 2024) is the
precedent for this paper's per-attempt call: with Statcast's caught-stealing probabilities, 81% of 3,410 tracked 2024
attempts cleared break-even and the other fifth cost 53 runs. Statcast's probability uses the runner's position at the
plate, which a runner only has after he goes; the paper's decision model is the before-the-pitch version.

## 2. Their weights on our data (*ours*)

Powers et al.'s model, refit with their filters and centring on 10,234 steals of second, 2023–26:

| Steal of second | Powers v2 (2022–23) | Same model, our data |
|---|---|---|
| Lead at first move, per ft | 0.14 (SE 0.03) | 0.148 (0.020) |
| Sprint speed, per ft/s | 0.18 (0.04) | 0.115 (0.028) |
| Catcher arm, per mph | −0.06 (0.01) | −0.068 (0.012) |
| SD runner / pitcher / catcher | 0.29 / 0.30 / 0.23 | 0.28 / 0.26 / 0.26 |
| Fit 2023–25, scored on 2026 | — | AUROC 0.603, deviance explained 2.5% |

Every model in log-odds per standard deviation of each input on our steals of second (SDs: lead 1.33 ft, speed
1.05 ft/s, arm 2.96 mph, pop time 0.051 s, ground gained 3.26 ft):

| | Lead | Speed | Catcher | Ground gained |
|---|---|---|---|---|
| Powers v2 | 0.19 | 0.19 | arm −0.18 | — |
| Powers model, our data | 0.20 | 0.12 | arm −0.20 | — |
| Pitcher's Dilemma | 0.31 | 0.22 | pop 0.08 | (jump speed 0.20) |
| Whitney-Epstein et al. | 0.10 | 0.08 | pop 0.27 | — |
| StealIQ decision GLMM | 0.20 | 0.13 | pop 0.29 | — |
| Powers model + ground gained | 0.27 | 0.38 | arm −0.28 | 1.29 |
| StealIQ success GLMM | 0.33 | 0.46 | pop 0.46 | 1.67 |

- The pre-pitch models agree, and the decision GLMM sits on Powers.
- Ground gained is about 3.5 times the next input. Adding it triples speed's weight (0.12 to 0.38): slower runners gain
  more ground, so a model without it hides speed.
- The pitcher effect is mostly ground. Loughin & Bargen found pitchers vary more than catchers (95% ranges 0.50–0.84
  against 0.59–0.79, about 0.42 and 0.25 on the log-odds scale); Powers find 0.36 (v1) and 0.30 (v2). With ground
  gained in the model the pitcher spread falls 71%, from 0.26 to 0.075, and runner and catcher carry what is left
  (0.34 each).

## 3. How the models rank on the same 2026 attempts (*ours*)

Every model fit on 2023–25 and scored on the same 3,145 attempts of 2026:

| Model | AUROC | Deviance explained |
|---|---|---|
| Sprint speed only | 0.551 | 0.4% |
| Powers model, plain logistic (paper, Table 2) | 0.602 | 2.1% |
| Powers model with player effects | 0.603 | 2.5% |
| Pitcher's Dilemma model | 0.612 | 3.0% |
| StealIQ decision GLMM | 0.629 | 3.8% |
| Powers model + ground gained | 0.765 | 13.6% |
| v16 tuned | 0.782 | 17.3% |
| StealIQ success GLMM | 0.791 | 17.9% |

Ground gained alone is worth +0.16 AUROC; curvature, pitch type, pop time to the base, game state and separate fits per
base add the last +0.026. Player effects add nothing a season ahead (0.603 against 0.602), so the paper's Table 2
comparison was fair.

## 4. What is missing, in order

1. Pitch velocity, location and pitchouts (Statcast, The Pitcher's Dilemma, TruMedia). The meta tables already hold
   `release_speed`, `plate_x`, `plate_z` and `zone`. The Dilemma's weights: −0.031 per mph; location zones −0.32 to
   +0.56 depending on batter side.
2. The runner's position when the pitch reaches the plate, and his first step (Statcast, TruMedia). Not in our data;
   later in the play than release, so it grades attempts rather than making the call.
3. Disengagements before the pitch (Powers, Price, the Dilemma's game): +0.84 and +1.54 log-odds on going after one and
   two in Powers. Savant's running-game leaderboard filters on them. Belongs in the decision model.
4. A pre-pitch measure of the pitcher's hold (Threat, pitch flight time, release time). A pitcher's earlier delivery
   times add +0.015 AUROC (95% CI +0.007 to +0.022) to a pre-pitch model (`results/4-Studies/DF_add_prior_delivery.csv`);
   the paper's decision model does not use them.
5. A season term (Powers +0.26 for 2023 over 2022; the Dilemma +0.45 after 2023), for calibration: the calculator read
   79.8% for 2026 against 78.5% observed.
6. Break-even by count (Tango's 288-state run expectancy, used by the Dilemma), not base and outs only.
7. The lead itself as a decision, with pickoff risk (Powers, Whitney-Epstein, the Dilemma). Needs leads on pickoff
   throws, which MLB does not publish.

## 5. Corrections for this repository

- `research.py`'s "Powers et al. v1, as published" row leaves out v1's own year term (+0.311 for 2023). As coded it
  predicts 73.3% success on our attempts against 80.7% observed; with the term, 78.9%. AUROC is unaffected and the paper
  does not use the row.
- That row encodes v1; the paper cites v2, whose lead weight rose from 0.052 to 0.14 and whose inputs are centred
  (lead − 10, speed − 27, arm − 80).
- The paper cites "42 Analytics" for *The Pitcher's Dilemma*; the authors are Melville, Greathouse, Mott, Archibald and
  Grimsman (MIT Sloan Sports Analytics Conference).
- The paper's related work should add Loughin & Bargen (2008), Statcast's Caught Stealing Above Average, TruMedia and
  Clemens (2024).

## Sources

- Powers et al.: [v2](https://arxiv.org/html/2601.15608v2), [v1](https://arxiv.org/html/2601.15608v1), [code](https://github.com/jfhahn2/pickoff-game-theory)
- [Loughin & Bargen (2008)](https://lida.sport-iat.de/twm/Record/4013568?lng=en); [Stanley (2023)](https://scholarcommons.sc.edu/senior_theses/602)
- [The Pitcher's Dilemma](https://www.sloansportsconference.com/research-papers/the-pitchers-dilemma-a-game-theoretical-model-of-pickoffs-and-stolen-bases); [Whitney-Epstein et al. (2025)](https://wsb.wharton.upenn.edu/wp-content/uploads/2025/12/WHITNEY-EPSTEIN_Lead_Distance_Optimization.pdf); [Price (2026)](https://dash.harvard.edu/entities/publication/69bbabbc-7b57-4fd4-b179-39b2cd164fa7)
- Statcast: [Caught Stealing Above Average](https://www.mlb.com/news/caught-stealing-above-average-stat-breakdown), [catcher throwing](https://baseballsavant.mlb.com/leaderboard/catcher-throwing), [pitcher running game](https://baseballsavant.mlb.com/leaderboard/pitcher-running-game)
- [TruMedia expected stolen bases](https://baseball.help.trumedianetworks.com/baseball/expected-stolen-bases-model); [Baseball Prospectus SRAA / TRAA](https://www.baseballprospectus.com/news/article/28193/prospectus-feature-catching-up); [Clemens (2024)](https://blogs.fangraphs.com/breaking-even-is-for-suckers/)
