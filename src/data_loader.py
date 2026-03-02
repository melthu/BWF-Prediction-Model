import pandas as pd

INPUT_PATH = "data/interim/engineered_matches.csv"
OUTPUT_PATH = "data/processed/final_training_data.csv"

METADATA_COLS = ["tournament", "host_country", "player_a_nat", "player_b_nat"]

SWAP_PAIRS = [
    ("player_a",                          "player_b"),
    ("player_a_is_home",                  "player_b_is_home"),
    ("player_a_matches_last_14_days",     "player_b_matches_last_14_days"),
    ("player_a_days_since_last_match",    "player_b_days_since_last_match"),
    ("player_a_recent_win_rate",          "player_b_recent_win_rate"),
    # New features
    ("player_a_elo",                      "player_b_elo"),
    ("player_a_ema_form",                 "player_b_ema_form"),
    ("player_a_win_streak",               "player_b_win_streak"),
    ("player_a_matches_last_7_days",      "player_b_matches_last_7_days"),
    # Score-derived features
    ("player_a_avg_point_diff",           "player_b_avg_point_diff"),
    ("player_a_avg_games_per_match",      "player_b_avg_games_per_match"),
]


def load_and_mirror(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df.drop(columns=METADATA_COLS)

    mirrored_df = df.copy()

    # Swap all Player A <-> Player B columns simultaneously to avoid clobbering
    for col_a, col_b in SWAP_PAIRS:
        df[col_a], df[col_b]  # just reference to confirm cols exist
        mirrored_df[col_a], mirrored_df[col_b] = df[col_b].copy(), df[col_a].copy()

    # Invert the target, H2H rate, and new directional features
    mirrored_df["player_a_won"]              = 1 - mirrored_df["player_a_won"]
    mirrored_df["h2h_win_rate_a_vs_b"]       = 1.0 - mirrored_df["h2h_win_rate_a_vs_b"]
    mirrored_df["elo_diff"]                  = -mirrored_df["elo_diff"]
    mirrored_df["h2h_last_winner"]           = 1.0 - mirrored_df["h2h_last_winner"]
    mirrored_df["player_a_avg_point_diff"]   = -mirrored_df["player_a_avg_point_diff"]

    final = pd.concat([df, mirrored_df], ignore_index=True)
    final.to_csv(output_path, index=False)

    print(f"Original shape : {df.shape}")
    print(f"Final shape    : {final.shape}")
    print(f"Saved to: {output_path}\n")

    print("First 2 rows (original):")
    print(final.head(2).to_string(index=True))
    print("\nLast 2 rows (mirrored counterpart):")
    print(final.tail(2).to_string(index=True))

    return final


if __name__ == "__main__":
    load_and_mirror()
