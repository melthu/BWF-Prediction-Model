# Project: BWF Men's Singles Match Prediction

## Directory Structure
```
bwfML/
├── run_pipeline.py               # Master runner — executes all 4 ETL steps end-to-end
├── src/                          # All Python source files
│   ├── __init__.py
│   ├── build_config.py           # Step 1 — scrapes BWF calendar pages (2021–2026)
│   ├── scraper_wiki_single.py    # Step 2a — single-tournament Wikipedia scraper
│   ├── scraper_orchestrator.py   # Step 2b — loops config and calls scraper_wiki_single
│   ├── feature_engineering.py   # Step 3 — computes rolling temporal features
│   └── data_loader.py            # Step 4 — mirrors dataset for ML symmetry
├── data/
│   ├── config/
│   │   └── tournaments_config.csv   # 143 tournaments: url, name, tier, start_date, host_country
│   ├── raw/
│   │   └── raw_matches.csv          # 4,305 match results with player names, nationalities, round
│   ├── interim/
│   │   └── engineered_matches.csv   # 4,305 rows + 11 engineered features (H2H, form, fatigue)
│   └── processed/
│       └── final_training_data.csv  # 8,610 rows — original + mirrored, metadata dropped, ML-ready
├── models/                       # Trained model artefacts (future)
└── CLAUDE.md
```

# Spec: `src/model.py` (DeepFM Architecture)

## 1. Context & Objective
Build a Deep Factorization Machine (DeepFM) in PyTorch to predict `player_a_won`. The model will process categorical indices via embeddings and continuous features via a standard feed-forward network.

## 2. Model Architecture
Create a class `BWFDeepFM(nn.Module)`:
* **Inputs**: `vocab_sizes` (dict of the 4 categorical sizes), `embed_dim` (default 16), `num_cont_features` (default 10), `hidden_dims` (default `[64, 32]`).
* **Embeddings**: Create `nn.Embedding` tables for `tier`, `round`, `player_a`, and `player_b`. All must have the same `embed_dim`.
* **FM Component**: Compute the 2nd-order pairwise interactions between the 4 categorical embeddings using the Factorization Machine formula.
* **Deep Component**: 
  1. Flatten and concatenate the 4 categorical embeddings.
  2. Concatenate that result with the `cont_features`.
  3. Pass the combined tensor through an MLP defined by `hidden_dims`, using ReLU activations and `nn.Dropout(p=0.2)`.
* **Output**: A final linear layer combining the FM output and the Deep output. Do NOT apply a Sigmoid at the end. Return the raw logits (we will use `BCEWithLogitsLoss` for numerical stability).

---

# Spec: `src/train.py` (The Training Loop)

## 1. Context & Objective
Train the `BWFDeepFM` model using the chronologically split datasets. Implement Early Stopping to prevent overfitting. Evaluate performance using Accuracy and ROC-AUC.

## 2. Implementation Logic
1. Call `get_train_val_datasets('data/processed/final_training_data.csv')` from `src.dataset`.
2. Initialize `DataLoader` for both train and val splits (batch size = 64).
3. Initialize `BWFDeepFM`, `nn.BCEWithLogitsLoss()`, and `torch.optim.Adam` (lr=0.001, weight_decay=1e-5).
4. **The Loop**: Train for up to 20 epochs. 
5. **Evaluation**: At the end of each epoch, calculate Validation Loss, Validation Accuracy, and Validation ROC-AUC (using `sklearn.metrics.roc_auc_score` after applying a sigmoid to the logits).
6. **Early Stopping**: Track the Validation Loss. If it does not improve for 3 consecutive epochs, halt training and print "Early stopping triggered".
7. Save the model weights with the best Validation Loss to `models/best_deepfm.pt`.



# Spec: `src/data_loader.py` (Dataset Mirroring Update)
Update the existing script to **KEEP** the `start_date` column in the final output. Do not drop it. We need it for chronological splitting in PyTorch. 

# Spec: `src/dataset.py` (Stateful PyTorch Preprocessing)

## 1. Context & Objective
Read the single `final_training_data.csv`, split it chronologically (Train = 2021-2025, Val = 2026+), and apply strict "Fit on Train, Transform on Val" preprocessing to avoid data leakage. Handle unknown future players gracefully.

## 2. Implementation Logic
Create a helper function `get_train_val_datasets(csv_path)` that does the following:
1. Load the CSV. Convert `start_date` to datetime.
2. **Chronological Split:** `train_df` is all rows where year <= 2025. `val_df` is all rows where year >= 2026.
3. **The Shared Player Dictionary (with UNK):** * Extract all unique player names from `player_a` and `player_b` *only in train_df*.
   * Assign ID `0` to `<UNK>` (Unknown Player). Assign IDs `1` to `N` for the training players.
4. **Stateful Scaling:**
   * Initialize a `StandardScaler` for the 10 continuous features.
   * `fit()` the scaler ONLY on `train_df`.
5. **Transforming the Data:**
   * Apply the scaler to both `train_df` and `val_df`.
   * Map `player_a` and `player_b` to their IDs. If a player in `val_df` is not in the dictionary (a 2026 rookie), assign them ID `0` (`<UNK>`).
   * Map `tier` and `round` to integer indices.
6. Return a tuple: `(train_dataset, val_dataset, vocab_sizes)` where the datasets are instances of a `BWFDataset` class (which simply returns the `cat_features`, `cont_features`, and `label` tensors for a given index, ignoring the `start_date`).

## 3. Execution Request
Rewrite `src/dataset.py` with this logic. Under `if __name__ == "__main__":`, call the function, and print the lengths of the Train and Val datasets, and the vocabulary size of the players (including the `<UNK>` token).


All scripts are run from the **project root** (`bwfML/`) so relative paths like
`data/raw/raw_matches.csv` resolve correctly.

## Pipeline Overview

The full pipeline is triggered by `python3 run_pipeline.py` and runs these four steps in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `build_config.py` | Scrapes the Wikipedia BWF World Tour calendar pages for 2021–2026. Finds every Super 100+ tournament by locating `<li><b>Level:</b></li>` cells, extracts the year-specific Draw URL, tier, start date, and host country. Saves 143 tournaments to `data/config/tournaments_config.csv`. |
| 2a | `scraper_wiki_single.py` | Core Wikipedia scraper for a single tournament. Isolates the Men's Singles section, classifies each table (bracket / group\_match / skip), uses a rowspan/colspan-aware column tracker to map players to rounds, extracts nationality from flagicon links, and determines the winner via bold tag detection. Returns a DataFrame of match pairs with a binary `player_a_won` target. |
| 2b | `scraper_orchestrator.py` | Macro-scraper that reads `tournaments_config.csv` and calls `scraper_wiki_single` for each tournament. Injects `start_date` and `host_country` into each result, sleeps 2 s between requests, and concatenates everything into `data/raw/raw_matches.csv`. |
| 3 | `feature_engineering.py` | Transforms raw matches into an ML-ready feature set. For every row, builds a strict historical slice (`start_date < current_date`) to prevent data leakage, then computes: H2H win rate, home advantage flags, matches played in last 14 days, days since last match, and 180-day rolling win rate — for both players. Saves to `data/interim/engineered_matches.csv`. |
| 4 | `data_loader.py` | Drops text metadata columns, then mirrors every row by swapping all Player A / Player B feature columns and inverting the target (`player_a_won`) and H2H rate. This doubles the dataset to 8,610 rows and enforces positional symmetry so the model cannot learn to favour whichever slot a player happens to be assigned to. Saves to `data/processed/final_training_data.csv`. |

---


# Spec: `build_config.py` (BWF Calendar Scraper)

## 1. Context & Objective
Automate the creation of `data/config/tournaments_config.csv` by scraping the official Wikipedia
BWF World Tour calendar pages for 2021–2026.

## 2. Scraping Logic
1. **Target URLs:** Loop `https://en.wikipedia.org/wiki/2021_BWF_World_Tour` → `2026_BWF_World_Tour`.
2. **Table Parsing:** Each tournament block is a `<td>` that contains `<li><b>Level:</b> …</li>`.
3. **Extraction per row:**
   * **Tournament Name & URL:** Bold `<a>` inside the cell (skip flagicon links). Append year to get e.g. "Malaysia Open 2025". Draw URL = `<a>` with text "Draw" pointing to `/wiki/`. Redlinks are skipped.
   * **Tier:** From `<li><b>Level:</b> Super 1000</li>`. Map Super 100/300/500/750/1000 → int; World Tour Finals → 1500. Filter out anything below Super 100.
   * **Start Date:** First `<td>` of the same `<tr>` as the tournament cell. Format: "7–12 January" → YYYY-MM-DD.
   * **Host Country:** From `<li><b>Host:</b> Kuala Lumpur, Malaysia</li>` — take the last comma-separated token.

## 3. Output Schema (`data/config/tournaments_config.csv`)
`url`, `tournament_name`, `tier`, `start_date`, `host_country`


---


# Spec: `scraper_wiki_single.py` (BWF Wikipedia MS Scraper V3)

## 1. Context & Objective
Extract strictly **Men's Singles** match outcomes from a single BWF tournament Wikipedia page.

## 2. Extraction Logic
**Step 1: Isolate the Men's Singles Section**
* Find the `<div class="mw-heading mw-heading2">` wrapping an h2/h3 matching "Men's singles".
* Walk next siblings, collecting `<table>` elements, until the next discipline's mw-heading2 div.

**Step 2: Classify Each Table**
* `classify_table()` returns `'bracket'`, `'group_match'`, or `'skip'`.
  * `'group_match'`: headers contain both "Player 1" and "Player 2" → label all matches "Group stage".
  * `'skip'`: headers contain "Seeds", "Rank", "NOCs", "W", "L", "Pld", "Pts", or "Nation".
  * `'bracket'`: everything else — use column-position round mapping.

**Step 3: Round Extraction via Column Tracking**
* `build_round_ranges(table)`: parse first-row headers accounting for colspan → `(start_col, end_col, round_name)`.
* `extract_player_cells(table)`: rowspan/colspan-aware column tracking via `col_occupancy` dict.
* `col_to_round(col_idx, ranges)`: maps a player's column index to its round name.

**Step 4: Match Pairing, Nationality & Binary Outcome**
* Players extracted via `<span class="flagicon">` tags.
* Nationality: `flagicon.find("a").get("title")`, cleaned of "national badminton team" artifacts.
* Player name: first `<a>` in the cell NOT inside a flagicon AND NOT containing "national badminton team".
* Winner: `flagicon.parent.name == "b"` → `player_a_won = 1`, else `0`.
* Players are paired sequentially (DOM order = match order).

## 3. Output Schema
`tournament`, `tier`, `round`, `player_a`, `player_a_nat`, `player_b`, `player_b_nat`, `player_a_won`


---


# Spec: `scraper_orchestrator.py` (BWF Macro-Scraper)

## 1. Context & Objective
Loop over every row in `data/config/tournaments_config.csv`, call `scrape_wiki_single`, inject
`start_date` and `host_country`, and compile `data/raw/raw_matches.csv`.

## 2. I/O Paths
* Input:  `data/config/tournaments_config.csv`
* Output: `data/raw/raw_matches.csv`

## 3. Logic
1. Read config. For each tournament: call scraper, insert `start_date` + `host_country`, append.
2. `time.sleep(2)` between requests.
3. Concatenate all frames and save.


---


# Spec: `feature_engineering.py` (Temporal Feature Generation)

## 1. Context & Objective
Transform `data/raw/raw_matches.csv` → `data/interim/engineered_matches.csv`.

## 2. The Golden Rule: Zero Data Leakage
`hist = df[df['start_date'] < current_date]` — strict less-than. Sort by `start_date` first.

## 3. Engineered Features
1. `tier` (kept as-is)
2. `same_nationality`
3. `h2h_win_rate_a_vs_b` (default 0.5)
4. `player_a_is_home`
5. `player_a_matches_last_14_days`
6. `player_a_days_since_last_match` (default 100)
7. `player_a_recent_win_rate` (180-day window, default 0.5)
8. `player_b_is_home`
9. `player_b_matches_last_14_days`
10. `player_b_days_since_last_match` (default 100)
11. `player_b_recent_win_rate` (180-day window, default 0.5)

## 4. I/O Paths
* Input:  `data/raw/raw_matches.csv`
* Output: `data/interim/engineered_matches.csv`


---


# Spec: `data_loader.py` (Dataset Mirroring & Final Prep)

## 1. Context & Objective
Prepare `data/interim/engineered_matches.csv` for ML by enforcing predictive symmetry via mirroring.

## 2. Columns to Drop
`tournament`, `start_date`, `host_country`, `player_a_nat`, `player_b_nat`

## 3. The Mirroring Logic
1. Load and drop metadata columns.
2. Create `mirrored_df`: swap all `player_a_*` ↔ `player_b_*` columns (including player name).
3. Invert: `player_a_won = 1 - player_a_won`, `h2h_win_rate_a_vs_b = 1.0 - h2h_win_rate_a_vs_b`.
4. Concatenate original + mirrored → row count exactly doubles.

## 4. I/O Paths
* Input:  `data/interim/engineered_matches.csv`
* Output: `data/processed/final_training_data.csv`


# Spec: `run_pipeline.py` (Master ETL Runner)

## 1. Context & Objective
Create a single master script in the root directory that executes the entire end-to-end BWF data engineering pipeline sequentially. 

## 2. Execution Flow
The script should use `subprocess.run` to execute the following files in order:
1. `src/build_config.py` (Generates the 5-year tournament calendar)
2. `src/scraper_orchestrator.py` (Scrapes Wikipedia and outputs `data/raw/raw_matches.csv`)
3. `src/feature_engineering.py` (Calculates temporal features and outputs `data/interim/engineered_matches.csv`)
4. `src/data_loader.py` (Mirrors the dataset and outputs `data/processed/final_training_data.csv`)

## 3. Logging & Error Handling
* Add clean console print statements (e.g., `[1/4] Building 5-year tournament config...`) before each step.
* Track the execution time of each step and print it when the step completes.
* If any script returns a non-zero exit code (fails), the master script should immediately halt, print an error message, and not attempt to run the subsequent steps.
* At the very end, print a final success message with the total pipeline execution time.