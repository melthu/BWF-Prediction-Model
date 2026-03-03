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

# Spec: `app.py` (Phase 5: The "Live Terminal" UI Revamp)

## 1. Context & Objective
Transform the Streamlit app into a live, interactive Point-in-Time trading terminal. It must feature a chronological tournament selector with a mini-calendar, live simulation progress rendering, a most-likely bracket visualizer, and a deep-dive Matchup Analyzer.

## 2. UI Layout & Temporal Navigation
1. **Sidebar - The Timeline:**
   * Display the "Current Date" clearly.
   * **Mini-Calendar Selector:** Implement `st.date_input` to render a mini-calendar.
   * **Dynamic Dropdown:** When a user picks a date on the calendar, filter the tournament dropdown to show tournaments happening in that specific month/year.
   * Format the tournament options to show: Date, Host Country Flag, Tier, and Tournament Name (e.g., "📅 2026-03-03 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Super 1000 All England Open").
   * Visually group them as [PAST] or [UPCOMING] relative to today.
   * Add a slider for `N_SIMULATIONS`.

## 3. The Live Engine (Tab 1)
1. **Point-in-Time Training:** Show live steps with `st.status()` when training the model strictly on data before the selected date.
2. **Live Ticker Effect:** Update the leaderboard placeholder every 500 simulations to show probabilities converging.
3. **Most Likely Bracket:** Render a visual tree of the "Expected Bracket".

## 4. Matchup Analyzer (Tab 2)
1. Dropdowns to pick Player A and Player B.
2. **Radar Chart:** Compare stats (Elo, Form, Streak, Fatigue) using Plotly.
3. **SHAP Waterfall:** Explain the XGBoost prediction for this exact matchup.

## 5. Flag Dictionaries
1. Implement `PLAYER_NATIONS` and `TOURNAMENT_HOSTS` for UI emojis.

# Spec: `app.py` (UI Upgrade: National Flags)

## 1. Context & Objective
Upgrade the Streamlit dashboard (`app.py`) to display national flag emojis next to player names to improve the visual UX.

## 2. Implementation Logic
1. **The Flag Dictionary:** Create a helper dictionary `PLAYER_FLAGS` at the top of `app.py` mapping the top ~40 BWF Men's Singles players to their country's emoji flag. 
   * *Examples:* 'Viktor Axelsen': '🇩🇰', 'Shi Yuqi': '🇨🇳', 'Anders Antonsen': '🇩🇰', 'Jonatan Christie': '🇮🇩', 'Anthony Sinisuka Ginting': '🇮🇩', 'Kodai Naraoka': '🇯🇵', 'Kunlavut Vitidsarn': '🇹🇭', 'Li Shifeng': '🇨🇳', 'Chou Tien-chen': '🇹🇼', 'Lee Zii Jia': '🇲🇾', 'Loh Kean Yew': '🇸🇬', 'Prannoy H. S.': '🇮🇳', 'Lakshya Sen': '🇮🇳', 'Toma Junior Popov': '🇫🇷', 'Christo Popov': '🇫🇷', 'Kento Momota': '🇯🇵'.
2. **The Helper Function:** Create a function `def format_name(player_name):` that looks up the player in the dictionary. If found, return `f"{flag} {player_name}"`. If not found, return `f"🏸 {player_name}"` as a fallback.
3. **UI Integration:** * Update the **Leaderboard DataFrame**: Apply this formatting function to the 'Player' column before rendering it.
   * Update the **First Round Bracket**: Format the player names in the bracket view.
   * Update the **SHAP Explainer Tab**: Ensure the selection dropdowns and the "Tale of the Tape" headers display the flags alongside the names.

# Spec: `app.py` (Phase 5: Point-in-Time Dashboard & UX)

## 1. Context & Objective
Upgrade the Streamlit dashboard to function as a Point-in-Time Historical Backtester with an interactive UI, real-time loading states, and a "Reality Check" actual-results overlay.

## 2. Implementation Logic
1. **Dynamic Point-in-Time Engine:**
   * Allow the user to select *any* tournament from the dataset.
   * When a tournament is selected, filter the master dataset to only include rows where `start_date < tournament_start_date`.
   * Rapidly train a fresh model (or use strictly pre-calculated Point-in-Time features) on this subset to guarantee zero future data leakage.
2. **Interactive Loading States (`st.status`):**
   * Wrap the execution block in `with st.status("Running Point-in-Time Engine...", expanded=True) as status:`.
   * Add `st.write()` steps inside to show progress: "Slicing historical data...", "Training Point-in-Time model...", "Running 10,000 Monte Carlo simulations...".
   * Change the status to complete when finished.
3. **The "Reality Check" Overlay:**
   * Query the dataset to find the actual winner of the selected tournament.
   * Add a column to the Monte Carlo probability leaderboard called `Actual Result`. 
   * Place a gold medal emoji (🥇) or "Winner" text next to the player who actually won, so users can visually backtest the model's accuracy.
4. **Tale of the Tape (Matchup Tab):**
   * In the SHAP Explainer tab, render a side-by-side comparison of the two selected players' Point-in-Time stats (Current Elo, Fatigue, Win Streak) *before* rendering the SHAP waterfall plot.

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
