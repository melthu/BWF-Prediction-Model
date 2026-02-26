import pandas as pd

INPUT_PATH = "data/raw/raw_matches.csv"
OUTPUT_PATH = "data/interim/engineered_matches.csv"


def get_player_matches(hist_df: pd.DataFrame, player: str) -> pd.DataFrame:
    """All historical rows where player appeared as either player_a or player_b."""
    return hist_df[(hist_df["player_a"] == player) | (hist_df["player_b"] == player)]


def count_wins(player_df: pd.DataFrame, player: str) -> int:
    """Vectorised win count for player in a pre-filtered slice."""
    return int((
        ((player_df["player_a"] == player) & (player_df["player_a_won"] == 1)) |
        ((player_df["player_b"] == player) & (player_df["player_a_won"] == 0))
    ).sum())


def engineer_features(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["start_date"] = pd.to_datetime(df["start_date"])

    # Golden Rule: sort chronologically so the row-wise history slice is always correct
    df = df.sort_values("start_date").reset_index(drop=True)

    rows = []

    for i, row in df.iterrows():
        current_date = row["start_date"]
        pa = row["player_a"]
        pb = row["player_b"]

        # Strict historical slice — excludes any match on the same date
        hist = df[df["start_date"] < current_date]

        # ------------------------------------------------------------------
        # Feature 1: tier  (kept as-is from raw data)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Feature 2: same_nationality
        # ------------------------------------------------------------------
        same_nat = 1 if row["player_a_nat"] == row["player_b_nat"] else 0

        # ------------------------------------------------------------------
        # Feature 3: h2h_win_rate_a_vs_b
        # ------------------------------------------------------------------
        h2h = hist[
            ((hist["player_a"] == pa) & (hist["player_b"] == pb)) |
            ((hist["player_a"] == pb) & (hist["player_b"] == pa))
        ]
        h2h_rate = count_wins(h2h, pa) / len(h2h) if len(h2h) > 0 else 0.5

        # ------------------------------------------------------------------
        # Features 4 & 8: home advantage flags
        # ------------------------------------------------------------------
        a_is_home = 1 if row["player_a_nat"] == row["host_country"] else 0
        b_is_home = 1 if row["player_b_nat"] == row["host_country"] else 0

        # ------------------------------------------------------------------
        # Player A temporal features (5, 6, 7)
        # ------------------------------------------------------------------
        a_hist = get_player_matches(hist, pa)

        if len(a_hist) > 0:
            a_days = (current_date - a_hist["start_date"]).dt.days
            a_matches_14  = int((a_days <= 14).sum())
            a_days_since  = int(a_days.min())           # min = most recent
            a_180         = a_hist[a_days <= 180]
            a_win_rate    = count_wins(a_180, pa) / len(a_180) if len(a_180) > 0 else 0.5
        else:
            a_matches_14 = 0
            a_days_since = 100
            a_win_rate   = 0.5

        # ------------------------------------------------------------------
        # Player B temporal features (9, 10, 11)
        # ------------------------------------------------------------------
        b_hist = get_player_matches(hist, pb)

        if len(b_hist) > 0:
            b_days = (current_date - b_hist["start_date"]).dt.days
            b_matches_14  = int((b_days <= 14).sum())
            b_days_since  = int(b_days.min())
            b_180         = b_hist[b_days <= 180]
            b_win_rate    = count_wins(b_180, pb) / len(b_180) if len(b_180) > 0 else 0.5
        else:
            b_matches_14 = 0
            b_days_since = 100
            b_win_rate   = 0.5

        rows.append({
            # --- Identifiers (kept for traceability) ---
            "tournament":             row["tournament"],
            "tier":                   row["tier"],
            "round":                  row["round"],
            "start_date":             row["start_date"],
            "host_country":           row["host_country"],
            "player_a":               pa,
            "player_a_nat":           row["player_a_nat"],
            "player_b":               pb,
            "player_b_nat":           row["player_b_nat"],
            "player_a_won":           row["player_a_won"],
            # --- Engineered features ---
            "same_nationality":                 same_nat,
            "h2h_win_rate_a_vs_b":              round(h2h_rate, 4),
            "player_a_is_home":                 a_is_home,
            "player_a_matches_last_14_days":    a_matches_14,
            "player_a_days_since_last_match":   a_days_since,
            "player_a_recent_win_rate":         round(a_win_rate, 4),
            "player_b_is_home":                 b_is_home,
            "player_b_matches_last_14_days":    b_matches_14,
            "player_b_days_since_last_match":   b_days_since,
            "player_b_recent_win_rate":         round(b_win_rate, 4),
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(f"Feature engineering complete: {len(result)} rows written to '{output_path}'.\n")

    display_cols = [
        "player_a", "player_b", "player_a_won",
        "h2h_win_rate_a_vs_b",
        "player_a_matches_last_14_days", "player_b_matches_last_14_days",
        "player_a_days_since_last_match", "player_b_days_since_last_match",
        "player_a_recent_win_rate", "player_b_recent_win_rate",
        "same_nationality", "player_a_is_home", "player_b_is_home",
    ]
    print("TAIL (10) — engineered feature columns:")
    print(result[display_cols].tail(10).to_string(index=True))

    return result


if __name__ == "__main__":
    engineer_features()
