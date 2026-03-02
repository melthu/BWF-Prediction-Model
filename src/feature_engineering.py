import re
from collections import deque

import pandas as pd

INPUT_PATH  = "data/raw/raw_matches.csv"
OUTPUT_PATH = "data/interim/engineered_matches.csv"

EMA_ALPHA    = 0.3
K_BY_TIER    = {100: 20, 300: 24, 500: 28, 750: 32, 1000: 40, 1500: 50}
SCORE_WINDOW = 10   # rolling window for point-differential features

_GAME_RE = re.compile(r"(\d+)\s*[–\-]\s*(\d+)")


def get_player_matches(hist_df: pd.DataFrame, player: str) -> pd.DataFrame:
    """All historical rows where player appeared as either player_a or player_b."""
    return hist_df[(hist_df["player_a"] == player) | (hist_df["player_b"] == player)]


def count_wins(player_df: pd.DataFrame, player: str) -> int:
    """Vectorised win count for player in a pre-filtered slice."""
    return int((
        ((player_df["player_a"] == player) & (player_df["player_a_won"] == 1)) |
        ((player_df["player_b"] == player) & (player_df["player_a_won"] == 0))
    ).sum())


def _elo_prepass(df: pd.DataFrame):
    """
    Single O(n) chronological scan to compute pre-match Elo, EMA form, and win
    streak for every row. Returns six parallel lists aligned to df.index.

    All values represent the state BEFORE the match is played (no leakage).
    """
    elo    = {}   # player -> current Elo (default 1500)
    ema    = {}   # player -> current EMA form (default 0.5)
    streak = {}   # player -> current streak int (0)

    elo_a_list    = []
    elo_b_list    = []
    ema_a_list    = []
    ema_b_list    = []
    streak_a_list = []
    streak_b_list = []

    for _, row in df.iterrows():
        pa      = row["player_a"]
        pb      = row["player_b"]
        tier    = row["tier"]
        actual_a = int(row["player_a_won"])
        actual_b = 1 - actual_a

        # --- Record PRE-match state ---
        elo_a = elo.get(pa, 1500.0)
        elo_b = elo.get(pb, 1500.0)
        ema_a = ema.get(pa, 0.5)
        ema_b = ema.get(pb, 0.5)
        s_a   = streak.get(pa, 0)
        s_b   = streak.get(pb, 0)

        elo_a_list.append(elo_a)
        elo_b_list.append(elo_b)
        ema_a_list.append(ema_a)
        ema_b_list.append(ema_b)
        streak_a_list.append(s_a)
        streak_b_list.append(s_b)

        # --- Post-match updates ---
        K = K_BY_TIER.get(tier, 24)
        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
        elo[pa] = elo_a + K * (actual_a - expected_a)
        elo[pb] = elo_b + K * (actual_b - (1.0 - expected_a))

        ema[pa] = EMA_ALPHA * actual_a + (1 - EMA_ALPHA) * ema_a
        ema[pb] = EMA_ALPHA * actual_b + (1 - EMA_ALPHA) * ema_b

        if actual_a == 1:
            streak[pa] = max(s_a, 0) + 1
            streak[pb] = min(s_b, 0) - 1
        else:
            streak[pa] = min(s_a, 0) - 1
            streak[pb] = max(s_b, 0) + 1

    return (elo_a_list, elo_b_list, ema_a_list, ema_b_list,
            streak_a_list, streak_b_list)


def _parse_score(score_str, player_a_won: int):
    """
    Parse a score string like "21-15, 18-21, 21-16" into per-player stats.

    Scores in raw_matches.csv are stored from the winner's perspective
    (winner points listed first in each game).  We need to flip them when
    player_b won so that the returned values are always from player_a's POV.

    Returns:
        (a_pts, b_pts, n_games) — ints — or None if string is empty/unparseable.
    """
    if not score_str or not isinstance(score_str, str) or not score_str.strip():
        return None
    games = _GAME_RE.findall(score_str)
    if not games:
        return None
    # games is a list of ("winner_pts", "loser_pts") string tuples
    if player_a_won:
        a_pts = sum(int(w) for w, _ in games)
        b_pts = sum(int(l) for _, l in games)
    else:
        # score stored from B's (winner's) perspective → swap
        a_pts = sum(int(l) for _, l in games)
        b_pts = sum(int(w) for w, _ in games)
    return (a_pts, b_pts, len(games))


def _score_prepass(df: pd.DataFrame):
    """
    Single O(n) chronological scan to compute rolling point differential
    and games-per-match averages for every row. Returns four parallel lists.

    All values represent the player's state BEFORE the match (no leakage).
    Defaults: point_diff = 0.0, games_per_match = 2.0 (minimum games in badminton).
    """
    hist_a: dict[str, deque] = {}   # player → deque of (point_diff, n_games)

    pdiff_a_list: list[float] = []
    pdiff_b_list: list[float] = []
    gpm_a_list:   list[float] = []
    gpm_b_list:   list[float] = []

    def _get_stats(player: str):
        dq = hist_a.get(player)
        if not dq:
            return 0.0, 2.0
        diffs = [d for d, _ in dq]
        gms   = [g for _, g in dq]
        return sum(diffs) / len(diffs), sum(gms) / len(gms)

    for _, row in df.iterrows():
        pa = row["player_a"]
        pb = row["player_b"]

        # --- Record PRE-match state ---
        pa_pdiff, pa_gpm = _get_stats(pa)
        pb_pdiff, pb_gpm = _get_stats(pb)

        pdiff_a_list.append(pa_pdiff)
        pdiff_b_list.append(pb_pdiff)
        gpm_a_list.append(pa_gpm)
        gpm_b_list.append(pb_gpm)

        # --- Post-match update (only when a score is available) ---
        parsed = _parse_score(row.get("score", ""), int(row["player_a_won"]))
        if parsed is not None:
            a_pts, b_pts, n_games = parsed
            if pa not in hist_a:
                hist_a[pa] = deque(maxlen=SCORE_WINDOW)
            hist_a[pa].append((a_pts - b_pts, n_games))
            if pb not in hist_a:
                hist_a[pb] = deque(maxlen=SCORE_WINDOW)
            hist_a[pb].append((b_pts - a_pts, n_games))

    return pdiff_a_list, pdiff_b_list, gpm_a_list, gpm_b_list


def engineer_features(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["start_date"] = pd.to_datetime(df["start_date"])

    # Golden Rule: sort chronologically so the row-wise history slice is always correct
    df = df.sort_values("start_date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Pre-pass: compute Elo, EMA, streak in a single chronological scan
    # ------------------------------------------------------------------
    (elo_a_pre, elo_b_pre, ema_a_pre, ema_b_pre,
     streak_a_pre, streak_b_pre) = _elo_prepass(df)

    # ------------------------------------------------------------------
    # Pre-pass: rolling point differential & games-per-match
    # ------------------------------------------------------------------
    (pdiff_a_pre, pdiff_b_pre,
     gpm_a_pre,   gpm_b_pre) = _score_prepass(df)

    rows = []

    for i, row in df.iterrows():
        current_date = row["start_date"]
        pa = row["player_a"]
        pb = row["player_b"]

        # Strict historical slice — excludes any match on the same date
        hist = df[df["start_date"] < current_date]

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
        # Feature: h2h_last_winner (H2H momentum)
        # 1 = A won most recent meeting, 0 = B won, 0.5 = no history
        # ------------------------------------------------------------------
        if len(h2h) > 0:
            last_h2h = h2h.sort_values("start_date").iloc[-1]
            if last_h2h["player_a"] == pa:
                h2h_last = float(last_h2h["player_a_won"])
            else:
                h2h_last = float(1 - last_h2h["player_a_won"])
        else:
            h2h_last = 0.5

        # ------------------------------------------------------------------
        # Features 4 & 8: home advantage flags
        # ------------------------------------------------------------------
        a_is_home = 1 if row["player_a_nat"] == row["host_country"] else 0
        b_is_home = 1 if row["player_b_nat"] == row["host_country"] else 0

        # ------------------------------------------------------------------
        # Player A temporal features
        # ------------------------------------------------------------------
        a_hist = get_player_matches(hist, pa)

        if len(a_hist) > 0:
            a_days = (current_date - a_hist["start_date"]).dt.days
            a_matches_14  = int((a_days <= 14).sum())
            a_matches_7   = int((a_days <= 7).sum())
            a_days_since  = int(a_days.min())
            a_180         = a_hist[a_days <= 180]
            a_win_rate    = count_wins(a_180, pa) / len(a_180) if len(a_180) > 0 else 0.5
        else:
            a_matches_14 = 0
            a_matches_7  = 0
            a_days_since = 100
            a_win_rate   = 0.5

        # ------------------------------------------------------------------
        # Player B temporal features
        # ------------------------------------------------------------------
        b_hist = get_player_matches(hist, pb)

        if len(b_hist) > 0:
            b_days = (current_date - b_hist["start_date"]).dt.days
            b_matches_14  = int((b_days <= 14).sum())
            b_matches_7   = int((b_days <= 7).sum())
            b_days_since  = int(b_days.min())
            b_180         = b_hist[b_days <= 180]
            b_win_rate    = count_wins(b_180, pb) / len(b_180) if len(b_180) > 0 else 0.5
        else:
            b_matches_14 = 0
            b_matches_7  = 0
            b_days_since = 100
            b_win_rate   = 0.5

        rows.append({
            # --- Identifiers ---
            "tournament":           row["tournament"],
            "tier":                 row["tier"],
            "round":                row["round"],
            "start_date":           row["start_date"],
            "host_country":         row["host_country"],
            "player_a":             pa,
            "player_a_nat":         row["player_a_nat"],
            "player_b":             pb,
            "player_b_nat":         row["player_b_nat"],
            "player_a_won":         row["player_a_won"],
            # --- Original 10 engineered features ---
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
            # --- New 10 engineered features ---
            "player_a_elo":                     round(elo_a_pre[i], 2),
            "player_b_elo":                     round(elo_b_pre[i], 2),
            "elo_diff":                         round(elo_a_pre[i] - elo_b_pre[i], 2),
            "player_a_ema_form":                round(ema_a_pre[i], 4),
            "player_b_ema_form":                round(ema_b_pre[i], 4),
            "h2h_last_winner":                  h2h_last,
            "player_a_win_streak":              streak_a_pre[i],
            "player_b_win_streak":              streak_b_pre[i],
            "player_a_matches_last_7_days":     a_matches_7,
            "player_b_matches_last_7_days":     b_matches_7,
            # New 4 — rolling score stats
            "player_a_avg_point_diff":          round(pdiff_a_pre[i], 4),
            "player_b_avg_point_diff":          round(pdiff_b_pre[i], 4),
            "player_a_avg_games_per_match":     round(gpm_a_pre[i], 4),
            "player_b_avg_games_per_match":     round(gpm_b_pre[i], 4),
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(f"Feature engineering complete: {len(result)} rows written to '{output_path}'.\n")

    display_cols = [
        "player_a", "player_b", "player_a_won",
        "elo_diff", "player_a_ema_form", "player_b_ema_form",
        "h2h_last_winner", "player_a_win_streak", "player_b_win_streak",
        "player_a_avg_point_diff", "player_b_avg_point_diff",
        "player_a_avg_games_per_match", "player_b_avg_games_per_match",
    ]
    print("TAIL (5) — new feature columns:")
    print(result[display_cols].tail(5).to_string(index=True))

    return result


if __name__ == "__main__":
    engineer_features()
