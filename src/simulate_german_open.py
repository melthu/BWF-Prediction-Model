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
MODEL_PATH = "models/best_lgbm.pkl"

TOURNAMENT = "German Open 2026"
TOUR_DATE  = "2026-02-24"
TIER       = 300
N_SIMS     = 10_000

ROUND_ORDER = ["first round", "second round", "quarter-finals", "semi-finals", "final"]


def load_lgbm():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_time_zero_state(df):
    """
    From the German Open 2026 first-round rows, extract each player's
    pre-tournament stats exactly as they appear on Day 1.
    Drop mirrored duplicates: keep one canonical row per (player_a, player_b) pair.
    """
    mask = (
        (df["start_date"] == pd.Timestamp(TOUR_DATE)) &
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

    # Build player_stats dict
    player_stats = {}
    for _, row in r32_unique.iterrows():
        for side in ("a", "b"):
            name = row[f"player_{side}"]
            if name not in player_stats:
                player_stats[name] = {
                    "is_home":           int(row[f"player_{side}_is_home"]),
                    "matches_14d":       int(row[f"player_{side}_matches_last_14_days"]),
                    "days_since":        float(row[f"player_{side}_days_since_last_match"]),
                    "recent_win_rate":   float(row[f"player_{side}_recent_win_rate"]),
                }

    return r32_unique, player_stats


def build_h2h_lookup(df):
    """
    Pre-compute H2H win rates from all rows strictly before 2026-02-24.
    Returns a function h2h(player_a, player_b) → float in [0, 1].
    """
    hist = df[df["start_date"] < pd.Timestamp(TOUR_DATE)].copy()

    cache = {}

    def h2h(pa, pb):
        key = (pa, pb)
        if key in cache:
            return cache[key]
        rows_a_home = hist[(hist["player_a"] == pa) & (hist["player_b"] == pb)]
        rows_b_home = hist[(hist["player_a"] == pb) & (hist["player_b"] == pa)]
        wins = rows_a_home["player_a_won"].sum() + (1 - rows_b_home["player_a_won"]).sum()
        total = len(rows_a_home) + len(rows_b_home)
        result = float(wins / total) if total > 0 else 0.5
        cache[key] = result
        return result

    return h2h


def _predict_one_direction(pa, pb, round_name, player_stats, h2h_fn, scaler, player_to_id, tier_to_id, round_to_id, lgbm_model):
    """Raw model call with pa in the player_a slot."""
    UNK = 0
    tier_id  = tier_to_id.get(TIER, 0)
    round_id = round_to_id.get(round_name, 0)
    pa_id    = player_to_id.get(pa, UNK)
    pb_id    = player_to_id.get(pb, UNK)

    sa = player_stats[pa]
    sb = player_stats[pb]

    cont_raw = np.array([[
        0.0,                                        # same_nationality (never same in a real match)
        h2h_fn(pa, pb),                             # h2h_win_rate_a_vs_b
        float(sa["is_home"]),                       # player_a_is_home
        float(sa["matches_14d"]),                   # player_a_matches_last_14_days
        float(sa["days_since"]),                    # player_a_days_since_last_match
        float(sa["recent_win_rate"]),               # player_a_recent_win_rate
        float(sb["is_home"]),                       # player_b_is_home
        float(sb["matches_14d"]),                   # player_b_matches_last_14_days
        float(sb["days_since"]),                    # player_b_days_since_last_match
        float(sb["recent_win_rate"]),               # player_b_recent_win_rate
    ]], dtype=np.float32)

    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_id, round_id, pa_id, pb_id]], dtype=np.int64)
    X   = np.hstack([cat, cont_scaled])
    return float(lgbm_model.predict_proba(X)[0, 1])


def predict_match(pa, pb, round_name, player_stats, h2h_fn, scaler, player_to_id, tier_to_id, round_to_id, lgbm_model):
    """
    Order-invariant win probability for pa beating pb.
    Averages both slot assignments so P(A beats B) == 1 - P(B beats A) exactly.
    """
    p_ab = _predict_one_direction(pa, pb, round_name, player_stats, h2h_fn, scaler, player_to_id, tier_to_id, round_to_id, lgbm_model)
    p_ba = _predict_one_direction(pb, pa, round_name, player_stats, h2h_fn, scaler, player_to_id, tier_to_id, round_to_id, lgbm_model)
    return (p_ab + (1.0 - p_ba)) / 2.0


def simulate_bracket(r32_matchups, player_stats, h2h_fn, scaler, player_to_id, tier_to_id, round_to_id, lgbm_model, rng):
    """Run one full bracket simulation. Returns the champion name."""
    current_round_players = []
    for _, row in r32_matchups.iterrows():
        current_round_players.append((row["player_a"], row["player_b"]))

    for round_name in ROUND_ORDER:
        next_round = []
        for pa, pb in current_round_players:
            p_a_wins = predict_match(pa, pb, round_name, player_stats, h2h_fn,
                                     scaler, player_to_id, tier_to_id, round_to_id, lgbm_model)
            winner = pa if rng.random() < p_a_wins else pb
            next_round.append(winner)
        # Pair winners for next round
        current_round_players = list(zip(next_round[::2], next_round[1::2]))
        if len(current_round_players) == 0:
            # Final was just played
            return next_round[0]

    return next_round[0]


def run():
    print("Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower()

    lgbm_model = load_lgbm()
    _, _, _, preprocessors = get_train_val_datasets(DATA_PATH)
    scaler       = preprocessors["scaler"]
    player_to_id = preprocessors["player_to_id"]
    tier_to_id   = preprocessors["tier_to_id"]
    round_to_id  = preprocessors["round_to_id"]

    r32_matchups, player_stats = build_time_zero_state(df)
    h2h_fn = build_h2h_lookup(df)

    # ------------------------------------------------------------------
    # Print the bracket
    # ------------------------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  2026 German Open — First Round Bracket ({len(r32_matchups)} matchups)")
    print(f"{'='*58}")
    for i, row in r32_matchups.iterrows():
        p = predict_match(row["player_a"], row["player_b"], "first round",
                          player_stats, h2h_fn, scaler, player_to_id,
                          tier_to_id, round_to_id, lgbm_model)
        print(f"  {row['player_a']:30s} vs {row['player_b']:30s}  | P(A wins)={p:.3f}")
    print(f"{'='*58}")

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    print(f"\nRunning {N_SIMS:,} simulations...")
    rng = np.random.default_rng(42)
    win_counts = {}

    for _ in range(N_SIMS):
        champion = simulate_bracket(
            r32_matchups, player_stats, h2h_fn,
            scaler, player_to_id, tier_to_id, round_to_id,
            lgbm_model, rng
        )
        win_counts[champion] = win_counts.get(champion, 0) + 1

    # ------------------------------------------------------------------
    # Print leaderboard
    # ------------------------------------------------------------------
    leaderboard = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*52}")
    print(f"  Championship Probability Leaderboard ({N_SIMS:,} sims)")
    print(f"{'='*52}")
    print(f"  {'Player':<32} {'Win %':>7}")
    print(f"  {'-'*32}  {'-'*7}")
    for name, wins in leaderboard:
        print(f"  {name:<32} {wins/N_SIMS*100:>6.2f}%")
    print(f"{'='*52}")


if __name__ == "__main__":
    run()
