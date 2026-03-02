import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd

from src.dataset import get_train_val_datasets, CONT_COLS

DATA_PATH  = "data/processed/final_training_data.csv"
MODEL_PATH = "models/best_model.pkl"

TOURNAMENT = "German Open 2026"
TOUR_DATE  = "2026-02-24"
TIER       = 300
N_SIMS     = 10_000

ROUND_ORDER = ["first round", "second round", "quarter-finals", "semi-finals", "final"]


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def model_predict_proba(payload, X):
    """Supports both single-model and ensemble payloads."""
    if payload["type"] == "ensemble":
        return sum(
            w * m.predict_proba(X)[:, 1]
            for w, m in zip(payload["weights"], payload["models"].values())
        )
    return payload["model"].predict_proba(X)[:, 1]


def build_time_zero_state(df, tour_date=None, tier=None):
    """
    From the given tournament's first-round rows, extract each player's
    pre-tournament stats exactly as they appear on Day 1.
    Drop mirrored duplicates: keep one canonical row per (player_a, player_b) pair.
    """
    if tour_date is None:
        tour_date = TOUR_DATE
    if tier is None:
        tier = TIER
    mask = (
        (df["start_date"] == pd.Timestamp(tour_date)) &
        (df["round"] == "first round")
    )
    r32 = df[mask].copy()

    # Drop mirrored duplicates: keep the row where player_a < player_b alphabetically
    seen = set()
    keep = []
    for _, row in r32.iterrows():
        pair = tuple(sorted([row["player_a"], row["player_b"]]))
        if pair not in seen:
            seen.add(pair)
            keep.append(row)
    r32_unique = pd.DataFrame(keep).reset_index(drop=True)

    # Build player_stats dict with all 24 CONT_COLS fields
    player_stats = {}
    for _, row in r32_unique.iterrows():
        for side in ("a", "b"):
            name = row[f"player_{side}"]
            if name not in player_stats:
                player_stats[name] = {
                    "is_home":         int(row[f"player_{side}_is_home"]),
                    "matches_14d":     int(row[f"player_{side}_matches_last_14_days"]),
                    "days_since":      float(row[f"player_{side}_days_since_last_match"]),
                    "recent_win_rate": float(row[f"player_{side}_recent_win_rate"]),
                    "elo":             float(row[f"player_{side}_elo"]),
                    "ema_form":        float(row[f"player_{side}_ema_form"]),
                    "win_streak":      int(row[f"player_{side}_win_streak"]),
                    "matches_7d":      int(row[f"player_{side}_matches_last_7_days"]),
                    # New score-derived features
                    "avg_point_diff":  float(row.get(f"player_{side}_avg_point_diff", 0.0)),
                    "avg_games_pm":    float(row.get(f"player_{side}_avg_games_per_match", 2.0)),
                }

    return r32_unique, player_stats


def build_h2h_lookups(df, tour_date=None):
    """
    Pre-compute two H2H signals from all rows strictly before the given tournament date.

    Returns:
        h2h_rate_fn(pa, pb)  → float win rate of pa vs pb in [0, 1]
        h2h_last_fn(pa, pb)  → 1.0 if pa won last meeting, 0.0 if pb did, 0.5 if none
    """
    if tour_date is None:
        tour_date = TOUR_DATE
    hist = df[df["start_date"] < pd.Timestamp(tour_date)].copy()
    hist = hist.sort_values("start_date")

    rate_cache = {}
    last_cache = {}

    def h2h_rate(pa, pb):
        key = (pa, pb)
        if key in rate_cache:
            return rate_cache[key]
        rows_a = hist[(hist["player_a"] == pa) & (hist["player_b"] == pb)]
        rows_b = hist[(hist["player_a"] == pb) & (hist["player_b"] == pa)]
        wins  = rows_a["player_a_won"].sum() + (1 - rows_b["player_a_won"]).sum()
        total = len(rows_a) + len(rows_b)
        result = float(wins / total) if total > 0 else 0.5
        rate_cache[key] = result
        return result

    def h2h_last(pa, pb):
        key = (pa, pb)
        if key in last_cache:
            return last_cache[key]
        # All meetings between the two, whichever slot they occupied
        meetings = hist[
            ((hist["player_a"] == pa) & (hist["player_b"] == pb)) |
            ((hist["player_a"] == pb) & (hist["player_b"] == pa))
        ].sort_values("start_date")
        if meetings.empty:
            result = 0.5
        else:
            last_row = meetings.iloc[-1]
            if last_row["player_a"] == pa:
                result = float(last_row["player_a_won"])
            else:
                result = float(1 - last_row["player_a_won"])
        last_cache[key] = result
        return result

    return h2h_rate, h2h_last


def _predict_one_direction(
    pa, pb, round_name, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, tier=None,
):
    """Raw model call with pa in the player_a slot (20-feature vector)."""
    UNK      = 0
    tier_id  = tier_to_id.get(TIER if tier is None else tier, 0)
    round_id = round_to_id.get(round_name, 0)
    pa_id    = player_to_id.get(pa, UNK)
    pb_id    = player_to_id.get(pb, UNK)

    sa = player_stats[pa]
    sb = player_stats[pb]

    # 24 features in CONT_COLS order
    cont_raw = np.array([[
        # Original 10
        0.0,                                # same_nationality
        h2h_rate_fn(pa, pb),                # h2h_win_rate_a_vs_b
        float(sa["is_home"]),               # player_a_is_home
        float(sa["matches_14d"]),           # player_a_matches_last_14_days
        float(sa["days_since"]),            # player_a_days_since_last_match
        float(sa["recent_win_rate"]),       # player_a_recent_win_rate
        float(sb["is_home"]),               # player_b_is_home
        float(sb["matches_14d"]),           # player_b_matches_last_14_days
        float(sb["days_since"]),            # player_b_days_since_last_match
        float(sb["recent_win_rate"]),       # player_b_recent_win_rate
        # New 10 (Elo/EMA/streak/H2H)
        float(sa["elo"]),                   # player_a_elo
        float(sb["elo"]),                   # player_b_elo
        float(sa["elo"] - sb["elo"]),       # elo_diff
        float(sa["ema_form"]),              # player_a_ema_form
        float(sb["ema_form"]),              # player_b_ema_form
        h2h_last_fn(pa, pb),               # h2h_last_winner
        float(sa["win_streak"]),            # player_a_win_streak
        float(sb["win_streak"]),            # player_b_win_streak
        float(sa["matches_7d"]),            # player_a_matches_last_7_days
        float(sb["matches_7d"]),            # player_b_matches_last_7_days
        # Score-derived 4
        float(sa["avg_point_diff"]),        # player_a_avg_point_diff
        float(sb["avg_point_diff"]),        # player_b_avg_point_diff
        float(sa["avg_games_pm"]),          # player_a_avg_games_per_match
        float(sb["avg_games_pm"]),          # player_b_avg_games_per_match
    ]], dtype=np.float32)

    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_id, round_id, pa_id, pb_id]], dtype=np.int64)
    X   = np.hstack([cat, cont_scaled])
    return float(model_predict_proba(model_payload, X)[0])


def predict_match(
    pa, pb, round_name, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, tier=None,
):
    """
    Order-invariant win probability for pa beating pb.
    Averages both slot assignments so P(A beats B) == 1 - P(B beats A) exactly.
    """
    p_ab = _predict_one_direction(
        pa, pb, round_name, player_stats,
        h2h_rate_fn, h2h_last_fn,
        scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
    )
    p_ba = _predict_one_direction(
        pb, pa, round_name, player_stats,
        h2h_rate_fn, h2h_last_fn,
        scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
    )
    return (p_ab + (1.0 - p_ba)) / 2.0


def simulate_bracket(
    r32_matchups, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, rng, tier=None,
):
    """Run one full bracket simulation. Returns the champion name."""
    current_round_players = [
        (row["player_a"], row["player_b"])
        for _, row in r32_matchups.iterrows()
    ]

    for round_name in ROUND_ORDER:
        next_round = []
        for pa, pb in current_round_players:
            p_a_wins = predict_match(
                pa, pb, round_name, player_stats,
                h2h_rate_fn, h2h_last_fn,
                scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
            )
            winner = pa if rng.random() < p_a_wins else pb
            next_round.append(winner)
        current_round_players = list(zip(next_round[::2], next_round[1::2]))
        if len(current_round_players) == 0:
            return next_round[0]

    return next_round[0]


def run():
    print("Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower()

    model_payload = load_model()
    model_name    = model_payload.get("name", model_payload.get("type", "unknown"))
    print(f"Model: {model_name}")

    _, _, _, preprocessors = get_train_val_datasets(DATA_PATH)
    scaler       = preprocessors["scaler"]
    player_to_id = preprocessors["player_to_id"]
    tier_to_id   = preprocessors["tier_to_id"]
    round_to_id  = preprocessors["round_to_id"]

    r32_matchups, player_stats = build_time_zero_state(df, TOUR_DATE, TIER)
    h2h_rate_fn, h2h_last_fn  = build_h2h_lookups(df, TOUR_DATE)

    # ------------------------------------------------------------------
    # Print the bracket
    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  2026 German Open — First Round Bracket ({len(r32_matchups)} matchups)")
    print(f"{'='*62}")
    for _, row in r32_matchups.iterrows():
        p = predict_match(
            row["player_a"], row["player_b"], "first round",
            player_stats, h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id, model_payload,
        )
        print(f"  {row['player_a']:30s} vs {row['player_b']:30s}  | P(A wins)={p:.3f}")
    print(f"{'='*62}")

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    print(f"\nRunning {N_SIMS:,} simulations...")
    rng = np.random.default_rng(42)
    win_counts = {}

    for _ in range(N_SIMS):
        champion = simulate_bracket(
            r32_matchups, player_stats,
            h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id,
            model_payload, rng,
        )
        win_counts[champion] = win_counts.get(champion, 0) + 1

    # ------------------------------------------------------------------
    # Print leaderboard
    # ------------------------------------------------------------------
    leaderboard = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*54}")
    print(f"  Championship Probability Leaderboard ({N_SIMS:,} sims)")
    print(f"{'='*54}")
    print(f"  {'Player':<32} {'Win %':>7}")
    print(f"  {'-'*32}  {'-'*7}")
    for name, wins in leaderboard:
        print(f"  {name:<32} {wins/N_SIMS*100:>6.2f}%")
    print(f"{'='*54}")


if __name__ == "__main__":
    run()
