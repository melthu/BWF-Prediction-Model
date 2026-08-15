"""
Rolling temporal evaluation harness for feature and model experiments.

Why this exists: `promote.py` picks a winner on a single holdout (the latest
season). With ~1,000 matches a season and three candidates within 0.002 AUC of
each other, that holdout cannot tell a real improvement from noise. Every
experiment in this directory is scored the same way instead: train on
everything strictly before year Y, evaluate on year Y, for several Y, and
report the mean.

Two things this harness does that a plain sklearn split would not:

  * it mirrors the TRAIN slice only (each match twice, players swapped), the
    same as `data_loader.py`, so slot order carries no signal;
  * it predicts the VAL slice order-invariantly - p = (P(orig) + 1 -
    P(swapped)) / 2 - which is exactly what `predict_match` does in
    production. Scoring a single orientation would flatter any model that
    happened to learn a slot bias.

The mirror spec is derived from the column names (`player_a_*` <-> `player_b_*`)
rather than hard-coded, so a new per-player feature is handled automatically.
Pair-level features must be registered in PAIR_LEVEL - they are the only ones
that transform under a swap.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

# Every fit here goes through numpy, so sklearn's "X does not have valid
# feature names" fires once per predict and buries the actual results.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTERIM_PATH = "data/interim/engineered_matches.csv"

# Pair-level columns: how each transforms when the two players swap slots.
# "neg" -> x becomes -x (a difference), "comp" -> x becomes 1 - x (a rate or a
# probability-like flag). Everything else is per-player and only moves slots.
PAIR_LEVEL = {
    "elo_diff":            "neg",
    "h2h_win_rate_a_vs_b": "comp",
    "h2h_last_winner":     "comp",
    # candidate pair-level features (experiments/candidate_features.py)
    "elo_expected_a":      "comp",
    "rank_points_diff":    "neg",
    "rank_diff":           "neg",
    "elo_diff_t":          "neg",
    "elo_expected_t":      "comp",
}

# Columns that are identifiers or bookkeeping, never model inputs.
NON_FEATURE = {
    "tournament", "start_date", "host_country", "player_a_nat", "player_b_nat",
    "player_a_won", "is_pending", "is_walkover", "player_a", "player_b",
    "tier", "round",
}

CAT_COLS = ["tier", "round", "player_a", "player_b"]

EVAL_YEARS = [2022, 2023, 2024, 2025, 2026]


# ----------------------------------------------------------------- data

def load_frame(path: str = INTERIM_PATH, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Engineered matches, completed only, chronologically sorted.

    `extra` is an optional frame of candidate features indexed identically to
    the engineered CSV (same row order, before any filtering), joined on
    position so an experiment can add columns without rewriting the pipeline.
    """
    df = pd.read_csv(path)
    if extra is not None:
        if len(extra) != len(df):
            raise ValueError(f"extra has {len(extra)} rows, engineered CSV has {len(df)}")
        df = pd.concat([df, extra.reset_index(drop=True)], axis=1)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower()
    df = df[(df["is_pending"] == 0) & (df["is_walkover"] == 0)].reset_index(drop=True)
    return df


def swap_pairs(cols) -> list[tuple[str, str]]:
    """Every (player_a_X, player_b_X) pair present in `cols`."""
    out = []
    for c in cols:
        if c.startswith("player_a_"):
            b = "player_b_" + c[len("player_a_"):]
            if b in cols:
                out.append((c, b))
    return out


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """The A<->B counterpart of every row (see data_loader.load_and_mirror)."""
    m = df.copy()
    for a, b in swap_pairs(df.columns):
        m[a], m[b] = df[b].values, df[a].values
    m["player_a"], m["player_b"] = df["player_b"].values, df["player_a"].values
    if "player_a_nat" in df.columns:
        m["player_a_nat"], m["player_b_nat"] = df["player_b_nat"].values, df["player_a_nat"].values
    for col, how in PAIR_LEVEL.items():
        if col in df.columns:
            m[col] = -df[col].values if how == "neg" else 1.0 - df[col].values
    m["player_a_won"] = 1 - df["player_a_won"].values
    return m


# ----------------------------------------------------------- encoding

def feature_cols(df: pd.DataFrame, include=None, exclude=()) -> list[str]:
    """Continuous model inputs: everything numeric that isn't bookkeeping."""
    cols = [c for c in df.columns if c not in NON_FEATURE]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if include is not None:
        cols = [c for c in cols if c in include]
    return [c for c in cols if c not in exclude]


class Encoder:
    """Vocabularies + scaler fit on a train slice only (leakage-free)."""

    def __init__(self, train_df: pd.DataFrame, cont_cols: list[str], use_player_ids: bool = True):
        self.cont_cols = cont_cols
        self.use_player_ids = use_player_ids
        players = sorted(set(train_df["player_a"]) | set(train_df["player_b"]))
        self.player_to_id = {p: i + 1 for i, p in enumerate(players)}
        self.tier_to_id = {t: i for i, t in enumerate(sorted(train_df["tier"].unique()))}
        self.round_to_id = {r: i for i, r in enumerate(sorted(train_df["round"].unique()))}
        self.scaler = StandardScaler().fit(train_df[cont_cols].values)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cats = [
            df["tier"].map(self.tier_to_id).fillna(0).values,
            df["round"].map(self.round_to_id).fillna(0).values,
        ]
        if self.use_player_ids:
            cats += [
                df["player_a"].map(self.player_to_id).fillna(0).values,
                df["player_b"].map(self.player_to_id).fillna(0).values,
            ]
        cat = np.column_stack(cats).astype(np.float64)
        cont = self.scaler.transform(df[self.cont_cols].values)
        return np.hstack([cat, cont])


# ------------------------------------------------------------ metrics

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc":     roc_auc_score(y, p),
        "logloss": log_loss(y, p),
        "brier":   brier_score_loss(y, p),
        "acc":     float(((p > 0.5).astype(int) == y).mean()),
        "n":       int(len(y)),
    }


def evaluate(df: pd.DataFrame, make_model, cont_cols: list[str],
             use_player_ids: bool = True, years=EVAL_YEARS,
             return_preds: bool = False) -> pd.DataFrame:
    """Rolling temporal evaluation. One row per eval year, plus a MEAN row.

    make_model() must return a fresh unfitted sklearn-style classifier.
    """
    rows, preds = [], []
    for y_val in years:
        train_df = df[df["start_date"].dt.year < y_val]
        val_df   = df[df["start_date"].dt.year == y_val]
        if len(val_df) < 50 or len(train_df) < 500:
            continue

        train_full = pd.concat([train_df, mirror(train_df)], ignore_index=True)
        enc = Encoder(train_full, cont_cols, use_player_ids)

        model = make_model()
        model.fit(enc.transform(train_full), train_full["player_a_won"].values)

        # Order-invariant prediction, exactly as production serves it.
        p_fwd = model.predict_proba(enc.transform(val_df))[:, 1]
        p_rev = model.predict_proba(enc.transform(mirror(val_df)))[:, 1]
        p = (p_fwd + (1.0 - p_rev)) / 2.0

        yv = val_df["player_a_won"].values
        rows.append({"year": y_val, **metrics(yv, p)})
        if return_preds:
            preds.append(pd.DataFrame({"year": y_val, "y": yv, "p": p,
                                       "idx": val_df.index}))

    out = pd.DataFrame(rows).set_index("year")
    mean = out.mean()
    mean["n"] = out["n"].sum()
    out.loc["MEAN"] = mean
    if return_preds:
        return out, pd.concat(preds, ignore_index=True)
    return out


def summarise(name: str, table: pd.DataFrame) -> dict:
    m = table.loc["MEAN"]
    return {"experiment": name, "auc": m["auc"], "logloss": m["logloss"],
            "brier": m["brier"], "acc": m["acc"]}
