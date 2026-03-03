# BWF Men's Singles Match Prediction — Project Reference

All scripts run from the **project root** (`bwfML/`). Relative paths like `data/raw/raw_matches.csv` resolve correctly from there.

---

## Directory Structure

```
bwfML/
├── run_pipeline.py              # Master CLI: --scrape --features --train --tune --all
├── app.py                       # Streamlit dashboard (Monte Carlo + SHAP)
├── Makefile
├── requirements.txt
├── src/
│   ├── build_config.py          # Step 1 — BWF calendar scraper (2010-2026)
│   ├── scraper_wiki_single.py   # Step 2a — single-tournament Wikipedia scraper
│   ├── scraper_orchestrator.py  # Step 2b — orchestrates all tournaments
│   ├── feature_engineering.py  # Step 3 — temporal feature engineering (30 features)
│   ├── data_loader.py           # Step 4 — dataset mirroring for positional symmetry
│   ├── dataset.py               # PyTorch dataset, extract_numpy, preprocessing
│   ├── model.py                 # BWFDeepFM (PyTorch)
│   ├── train.py                 # DeepFM training loop (standalone, not in pipeline)
│   ├── train_lgbm.py            # LightGBM trainer
│   ├── train_catboost.py        # CatBoost trainer
│   ├── train_xgb.py             # XGBoost trainer
│   ├── train_ensemble.py        # Ensemble selection + DeepFMWrapper
│   ├── temporal_cv.py           # Rolling 3-fold temporal cross-validation
│   ├── tune_hyperparams.py      # Optuna search for XGBoost + LightGBM
│   └── simulate_german_open.py  # Monte Carlo tournament simulation
├── data/
│   ├── config/tournaments_config.csv   # 241 tournaments 2010-2026 (tracked)
│   ├── raw/raw_matches.csv             # 9,255 match results (tracked)
│   ├── interim/                        # engineered features (git-ignored)
│   └── processed/                      # ML-ready mirrored dataset (git-ignored)
└── models/
    ├── best_params.json     # Optuna best hyperparameters (tracked)
    └── *.pkl / *.pt         # trained artifacts (git-ignored, run make train)
```

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `build_config.py` | `data/config/tournaments_config.csv` — 241 tournaments |
| 2 | `scraper_orchestrator.py` | `data/raw/raw_matches.csv` — 9,255 matches with scores, seeds, walkovers |
| 3 | `feature_engineering.py` | `data/interim/engineered_matches.csv` — 30 temporal features, walkovers filtered |
| 4 | `data_loader.py` | `data/processed/final_training_data.csv` — 18,118 rows (mirrored) |

---

## Features

**4 categorical:** tier, round, player\_a ID, player\_b ID

**30 continuous (`CONT_COLS` in `dataset.py`):**
- *Original (10):* same\_nationality, h2h\_win\_rate, home flags ×2, 14d matches ×2, days\_since ×2, 180d win\_rate ×2
- *Elo / EMA (10):* player\_a/b Elo, elo\_diff, player\_a/b EMA form, h2h\_last\_winner, win\_streak ×2, matches\_7d ×2
- *Score-derived (4):* avg\_point\_diff ×2, avg\_games\_per\_match ×2 (rolling 10 matches)
- *Phase 5 (6):* rubber\_game\_rate ×2, avg\_victory\_margin ×2, seed ×2

**Train split:** 2010–2025 (17,784 rows) | **Val split:** 2026+ (334 rows)

---

## Current Model Results

| Model    | Val AUC |
|----------|---------|
| LightGBM | 0.7372  |
| CatBoost | 0.7821  |
| XGBoost  | **0.7872** ← best |
| Ensemble | 0.7768  |

Best model saved to `models/best_model.pkl` as `{"type": "single", "model": xgb, "name": "xgb"}`.

---

## Key Rules & Gotchas

**No data leakage:** All temporal features use strict `hist = df[df['start_date'] < current_date]`.

**Preprocessing is stateful:** Vocab (player/tier/round IDs) and `StandardScaler` are fit on train only, then applied to val. `fill_missing_cont_cols()` in `dataset.py` fills absent `CONT_COLS` with 0.0 for backward compat.

**`extract_numpy(dataset)`** lives in `src/dataset.py` — import from there; do not re-define locally.

**`start_date` is kept** in `final_training_data.csv` (needed for chronological split). `data_loader.py` does NOT drop it.

**CatBoost gotcha:** `np.hstack([cat_int, cont_float])` produces `float64`. CatBoost's `cat_features=` parameter doesn't work on float arrays — leave it off.

**Order-invariant prediction:** `simulate_german_open.py` averages P(A beats B) as `[P(A|slot_a) + (1 - P(B|slot_a))] / 2` to eliminate positional bias.

**Backward-compat X trimming:** Older saved models have `n_features_in_ < 34`. `simulate_german_open.py` and `app.py` trim `X[:, :n]` via `model.n_features_in_` before calling `predict_proba`.

**In-bracket Elo/EMA:** `simulate_german_open.py` deep-copies `player_stats` at the start of each Monte Carlo iteration so Elo/EMA updates don't bleed between simulations.

**Scraper — winner detection:** Two Wikipedia formats exist:
- Modern (2018+): `<b><span class="flagicon">…</span><a>Name</a></b>` → `flagicon.parent.name == "b"`
- Classic (2010–2017): `<span class="flagicon">…</span><b><a>Name</a></b>` → `player_link.find_parent("b")`
Both checks are applied with `or` in `extract_player_cells`.

**Elo K-factors by tier:** `{100: 20, 300: 24, 500: 28, 750: 32, 1000: 40, 1500: 50}`. Default Elo = 1500. EMA α = 0.3, default = 0.5.
