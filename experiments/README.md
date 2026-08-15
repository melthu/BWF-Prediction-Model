# experiments/

Exploratory work on making the prediction better. Nothing here runs in CI or is
imported by `src/`; it exists to decide what *should* move into `src/`.

## Why a separate harness

`promote.py` benchmarks candidates on a single temporal holdout — the latest
season, roughly a thousand matches — and the three candidates historically sit
within 0.002 AUC of each other. A holdout that thin cannot tell a real
improvement from resampling noise, so before this directory there was no honest
way to answer "does this feature help?".

`harness.py` scores everything identically:

* **rolling temporal evaluation** — train on all years strictly before Y,
  evaluate on Y, for Y in 2022–2026, and report the mean of the five;
* **train-side mirroring only** — each training match appears twice with the
  players swapped, exactly as `data_loader.py` does, so slot order carries no
  signal;
* **order-invariant scoring** — the validation slice is predicted as
  `(P(orig) + 1 − P(swapped)) / 2`, exactly as `predict_match` serves it.
  Scoring one orientation would flatter any model that learned a slot bias the
  real serving path removes.

The mirror spec is derived from column names (`player_a_*` ↔ `player_b_*`), so
a new per-player feature needs no registration. Pair-level features are the
only ones that transform under a swap and must be added to `PAIR_LEVEL`.

## The trap this directory was built to avoid

The first ablation added all 36 candidate columns to the shipped LightGBM and
reported that **every** feature group made the model worse. That result was an
artifact: the shipped config was Optuna-tuned for 30 features, including
`feature_fraction=0.42`, and handing it 66 features changes what that number
means. A feature set and its hyperparameters are one object and have to be
compared as one.

`run_search.py` is the fix — each feature set gets its own random search on an
identical budget, and selection is split from reporting:

| | years |
|---|---|
| select the config | 2022–2024 |
| report its score | 2025–2026 |

`run_capacity.py` is kept for the same reason: it shows the shipped *default*
params (1000 trees × 63 leaves) at 0.7157 AUC against 0.7326 for the tuned
ones, i.e. most of the "capacity" story was really a params story.

## Reference points

Measured on the rolling harness, mean over 2022–2026:

| | AUC | logloss |
|---|---|---|
| logistic regression on `elo_diff` alone | 0.7032 | 0.6261 |
| logistic regression, all 34 features | 0.7162 | 0.6184 |
| CatBoost, tuned | 0.7180 | 0.6302 |
| XGBoost, tuned | 0.7288 | 0.6108 |
| **LightGBM, tuned — the shipped baseline** | **0.7326** | **0.6060** |

Elo alone gets most of the way. That is the reason for `run_elo.py`.

## What shipped, and what did not

One change made it into `src/`, and it was not a feature:

| | all 5 years | logloss |
|---|---|---|
| shipped baseline, hand-set Elo | 0.7309 | 0.6074 |
| **fitted Elo (now in `src/pipeline/elo.py`)** | **0.7362** | **0.6041** |

Six candidate feature groups were built, screened, and rejected. The noise
floor below is what makes that a judgement rather than an opinion.

### The noise floor — read this before any table

`run_noise.py` holds the feature set and the search budget fixed and varies
only the seeds. Eight runs:

| | mean | std | spread |
|---|---|---|---|
| all-5-year AUC | 0.7293 | 0.0011 | **0.0034** |
| report-window AUC | 0.7191 | 0.0018 | 0.0057 |

So a feature-set difference under ~0.003 AUC is not evidence of anything.
Against that: the fitted Elo's +0.0053 is real; every candidate group's
±0.002 is not.

### Re-tuning the hyperparameters did not help either

`best_params.json` was tuned by Optuna on the old 34-feature set, so an obvious
worry was that it no longer fit the 35-feature one. A fresh 60-trial search per
model, then compared on the rolling harness against the original params:

| model | original AUC | re-tuned AUC |
|---|---|---|
| lgbm | **0.7376** | 0.7344 |
| xgb | 0.7356 | 0.7347 |
| catboost | 0.7231 | 0.7341 |

The re-tune made the best model (lgbm) *worse* by 0.003 — outside the noise
floor — and left xgb within it. A 60-trial search was simply unluckier than the
original run; the fitted-Elo feature did not move the optimum enough to matter.
The original params were kept.

One thing this surfaced is worth flagging: on the rolling harness lgbm (0.7376)
beats xgb (0.7356), but `promote.py` selects on the single latest season, where
xgb's 0.7315 edges lgbm's 0.7289, so it ships xgb. That is the same thin-holdout
weakness this directory exists to route around, now visible in the production
selector itself. Changing how `promote.py` picks is a separate decision and was
left alone here.

### Model-level changes did not help

Two changes that owe nothing to features, each given the same per-set search:

| | all-5yr AUC | logloss |
|---|---|---|
| fitted-Elo LightGBM | 0.7362 | 0.6031 |
| + soft-vote blend (lgb+xgb+cat) | 0.7354 | 0.6052 |
| + isotonic calibration | 0.7355 | 0.6034 |

The blend is slightly worse on both; calibration is flat. The calibration
result is the interesting one, because the Monte Carlo consumes probabilities
directly and a miscalibrated model would show up as a logloss the isotonic step
could recover — and there was nothing to recover. The fitted rating carries
`elo_expected`, a real probability, straight into the trees, so the model's
output is already close to calibrated without a correction layer.

### Tabular foundation models are a real gain, and still not worth shipping

TabPFN and TabICL are pretrained transformers that classify a new table *in
context*: the training rows go in as a prompt and the test rows come out of one
forward pass, with no gradient step on this dataset at all. They win most public
tabular benchmarks under ~10k rows, which is about the size of this corpus, so
they are a fair thing to ask about. `run_tabfm.py` scores them on the rolling
harness; `run_tabfm_combine.py` scores the blends off cached predictions.

**Re-measure the baseline before reading these.** The corpus has grown and the
round-ordering fix has landed since the tables above, so LightGBM now scores
0.7258 on the same harness rather than 0.7376. Everything below is measured
against *that* run, in the same process, on the same folds.

Standalone, over 4,357 validation matches — every one a paired bootstrap over
matches against shipped LightGBM:

| | rolling AUC | logloss | Δ vs shipped | 95% CI | |
|---|---|---|---|---|---|
| lgbm (shipped) | 0.7258 | 0.6114 | — | — | — |
| TabPFN, 4×10k subsample ens. | 0.7290 | 0.6084 | +0.0030 | [−0.0021, +0.0080] | no difference |
| TabPFN, no player IDs | 0.7274 | 0.6105 | +0.0016 | [−0.0043, +0.0071] | no difference |
| TabICL, 4×4k subsample ens. | 0.7286 | 0.6087 | +0.0033 | [−0.0020, +0.0086] | no difference |
| TabICL, no player IDs | 0.7281 | 0.6086 | +0.0026 | [−0.0029, +0.0081] | no difference |

Not one of them separates from the baseline on its own. **Blended 50/50 with it,
all four do:**

| blend with lgbm | pooled AUC | Δ | 95% CI | worst fold |
|---|---|---|---|---|
| + TabICL | **0.7309** | **+0.0047** | [+0.0019, +0.0073] | +0.0018 |
| + TabICL, no player IDs | 0.7306 | +0.0043 | [+0.0015, +0.0070] | +0.0025 |
| + TabPFN | 0.7303 | +0.0040 | [+0.0015, +0.0065] | +0.0033 |
| + TabPFN, no player IDs | 0.7301 | +0.0038 | [+0.0008, +0.0067] | +0.0002 |

This is the first thing tried here that beats the shipped model on the harness's
own terms. It is worth being precise about why it is believable and about what
it is not.

It is believable because it replicates. Four independently configured runs —
two architectures × player IDs in and out — land within 0.0009 of each other,
and every one is positive in **every fold**, which is not what the ±0.002 of the
rejected feature groups looked like. The soft-vote of lgbm+xgb+catboost failed
above because three tree ensembles fit the same way make the same mistakes; an
in-context transformer is a genuinely different estimator, and the blend is
collecting exactly that decorrelation. Note the two halves of the evidence pull
in opposite directions on the same models: TabPFN alone is +0.0030 with a
*wider* interval than its own blend, because a blend is strongly correlated with
the baseline inside it and the paired test measures only the difference.

What it is not is large. +0.0047 AUC is above the 0.003 seed-noise floor but
within spitting distance of it, and the blend weight was not tuned — 0.5 and 0.3
were both tried, both positive, and 0.5 reported.

**The cost is what settles it.** Measured on this harness, order-invariant match
predictions per second:

| | preds/s | matchup shards (26,335 pairs) | one 32-draw Monte Carlo |
|---|---|---|---|
| lgbm | — | seconds | ~3 s |
| TabICL | 58.9 | 0.2 h | **2.9 h** |
| TabPFN | 1.7 | 8.4 h | **99 h** |

`run_monte_carlo` is 10,000 sims batched two orientations deep per round, so a
single 32-player draw is 620,000 forward rows. The daily export runs that for
every tournament whose rows changed. Three seconds becomes three hours, and a
live event re-exports as each result lands. There is no version of this that
fits a scheduled job.

The obvious dodge — serve the blend on the cards and matchup shards, leave the
simulation on the trees — is worse than it looks. `known_probs` exists precisely
because a card and a leaderboard describing the same match shipped 0.60 against
0.64; deliberately building the two surfaces on different models reintroduces
that class of bug by construction.

So: recorded as a measured positive that is not deployable at this repo's
serving budget, not as a dead end. What would change the answer is distillation
— train the trees on the blend's probabilities rather than serving the blend —
which is a different experiment and was not run.

Two practical notes for anyone re-running it. `tabpfn>=6` gates its weights
behind a Prior Labs licence token (`TABPFN_TOKEN`); the 2.x line downloads the
same v2 checkpoints unauthenticated, and that is what these numbers use. And
**torch must be imported before anything that pulls in scikit-learn** — both
ship a libomp, and on macOS the second one loses, with `OMP: Error #179` and a
segfault on the first forward pass, stdout still buffered so the log is empty.

TabICL's context is the one place these runs are not like-for-like. It has no
10k pretraining limit and the honest run is on all 18,754 mirrored rows, but
that reached 6.25 GB resident and drove the test machine into swap, so it is
capped at 4k per member against TabPFN's 10k — measured at 2.66 GB at 4k, 4.55 GB
at 6k, 7.4 GB at 10k. It matches TabPFN anyway, on 40% of the context and 3% of
the time, so the cap costs it nothing that shows up in the score.

### The groups that did not survive

Each got its own hyperparameter search on an identical budget:

| set | select | report | all 5yr | verdict |
|---|---|---|---|---|
| baseline | 0.7367 | 0.7186 | 0.7295 | — |
| +QUALITY | 0.7382 | 0.7204 | 0.7311 | +0.0016, inside noise |
| +RANK | 0.7325 | 0.7247 | 0.7294 | signs disagree — noise |
| +FATIGUE | 0.7362 | 0.7171 | 0.7285 | negative |
| +CONTEXT | 0.7361 | 0.7169 | 0.7284 | negative |
| +SCORING | 0.7344 | 0.7150 | 0.7266 | negative |
| +ELO derivs | 0.7315 | 0.7154 | 0.7251 | negative |

`+RANK` is the instructive one. Best report score of any set, second-worst
select score — that inconsistency is the shape of noise, not of signal. And
stacked on the fitted rating it actively hurts (0.7347 → 0.7301). A ranking is
a summary of results; a rating that has actually been fit to those results
already contains it.

## Findings

### Elo's own constants were never fit — tuning them is worth +0.027 AUC

`run_elo.py` scores the raw Elo expectancy with no model on top, so the number
is a statement about the rating rather than about the trees. It searches
parameters the shipped rating does not have at all:

* **margin of victory** — a 21-5 win moves the rating further than 22-20. The
  shipped rating parses scorelines for other features and then throws them away
  here.
* **provisional K** — a larger K for a player's first ~23 matches, so a
  newcomer converges off 1500 in a few events instead of over a season.
* **inactivity decay** — regression toward the mean during a layoff.
* **tier→K curve** — currently a hand-written lookup.

Selected on 2019–2023, reported on 2024–2026 which the fit never saw:

| | AUC | logloss |
|---|---|---|
| shipped Elo | 0.6868 | 0.6342 |
| tuned Elo | **0.7135** | **0.6197** |

The tuned parameters are worth reading for what they say about the sport:
`tier_alpha ≈ 0.06` means the tier of an event barely matters to how much a
result should move a rating — the shipped `K_BY_TIER` spread of 20→50 is doing
almost nothing. Nearly all of the gain comes from margin of victory and from
provisional K.

`tuned_elo.py` emits the tuned rating as columns so the tree model can be asked
whether a better input makes it a better model.

### The ranking feature, and what it actually is

Real BWF ranking history is not fetchable here: `bwfbadminton.com` and
`corporate.bwfbadminton.com` both return 403 behind Cloudflare, the community
mirror [raywan/bwf-data](https://github.com/raywan/bwf-data) covers only 2015
w1–2016 w7, and the Kaggle archive needs an authenticated download. **No
ranking column in this repo is downloaded data.**

What `candidate_features.py` builds instead is a proxy computed from the
results already in the corpus: each player is awarded the points their
finishing round earns at that tournament's tier, on BWF's own table, and the
best 10 events inside a rolling 52 weeks are summed — BWF's own accounting
rule. From that, a live ordinal rank.

`validate_rank_proxy.py` checks it against every scrap of real ranking data
that *is* reachable, in two different eras:

| | 58 weekly snapshots, 2015–16 | Wikipedia top 20, 2026-07-21 |
|---|---|---|
| Spearman ρ vs published ranking | **0.854** (0.812–0.897) | **0.806** |
| real top 10 also in proxy top 10 | 6.9 / 10 | 7 / 10 |
| median absolute rank error | 6.6 places | **2.0 places** |
| players compared | ~61 per week | 20 / 20 |

The two disagree about error size because they measure different things: the
2015–16 figure runs ~60 deep, where the proxy's tail is noisy, while the
Wikipedia snapshot is the top 20 only — and there the proxy is within a couple
of places. The head of the list is what a match model actually cares about.

So it is the same quantity, imperfectly measured, and most accurate exactly
where it matters. Good enough to use as a feature; not good enough to display
to a user as "world ranking".

### What Wikipedia can and cannot give

[BWF World Ranking](https://en.wikipedia.org/wiki/BWF_World_Ranking) carries a
**single current snapshot** — top 20 per discipline, with points and career
peak — plus year-end number ones and a number-one timeline. There is no weekly
or per-player historical series, so it cannot supply a ranking column for
10,196 matches spanning 2010–2026. It is used here as a validation point, not
as a feature source.

**To use real rankings as a feature**, download the year-by-year XLS from
[BWF Corporate → Historical Rankings](https://corporate.bwfbadminton.com/players/historical-rankings/)
(the page needs a browser; the download is one file per year) and drop them in
`data/raw/rankings/`. Only the loader in `candidate_features.py` would change.

## Files

| file | what it does |
|---|---|
| `harness.py` | rolling evaluation, mirroring, order-invariant scoring |
| `models.py` | model factories, shared param defaults |
| `candidate_features.py` | all candidate features in one chronological pass |
| `tuned_elo.py` | the tuned rating as feature columns |
| `run_baseline.py` | the shipped feature set, three model families |
| `run_capacity.py` | capacity/regularisation sweep, player-ID ablation |
| `run_features.py` | the naive (unfair) group ablation — kept as the warning |
| `run_search.py` | per-feature-set hyperparameter search, split select/report |
| `run_elo.py` | tunes Elo's own constants |
| `run_tabfm.py` | TabPFN and TabICL on the rolling harness |
| `run_tabfm_combine.py` | scores cached predictions and their blends |
| `run_combined.py` | does the tuned rating help the trees, and does it compound with the ranking proxy |
| `validate_rank_proxy.py` | proxy vs the real published ranking |

Results land in `experiments/results/*.json`.
