"""
Precompute every model output the dashboard shows, as static JSON.

Run from the project root:

    python src/serving/export_static.py                # incremental (default)
    python src/serving/export_static.py --force        # rebuild everything
    python src/serving/export_static.py --only 2026-07-21

Design notes
------------
* **No model ever ships.** A point-in-time model exists only to predict its own
  tournament's ~31 matches, so we export what it said, not the estimator.
* **Sharded by access pattern.** The index is small enough to load on first
  paint; tournament, player and matchup files are fetched on demand.
* **Incremental.** Each tournament file carries a fingerprint of the inputs that
  produced it. Unchanged inputs are skipped, so re-running costs almost nothing.
  Because the fingerprint covers the tournament's own rows, a live event
  re-exports automatically as results land - which is what lets the site show
  semi-final predictions once the quarter-finals are in.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

from src.modeling.dataset import CONT_COLS, encode_split, load_training_frame
from src.pipeline.feature_engineering import order_by_round
from src.pipeline.player_names import fold_ascii
from src.modeling.pit_model import train_point_in_time
from src.serving.simulate import (
    ROUND_ORDER,
    build_bracket,
    build_fixed_results,
    build_h2h_lookups,
    build_time_zero_state,
    is_placeholder,
    load_model,
    predict_match,
    run_monte_carlo,
)

DATA_PATH   = "data/processed/final_training_data.csv"
RAW_PATH    = "data/raw/raw_matches.csv"
CONFIG_PATH = "data/config/tournaments_config.csv"
OUT_DIR     = "site/data"

FIRST_YEAR   = 2018      # World Tour era; browsable scope, not training scope
N_SIMS       = 10_000
ACTIVE_SINCE = "2025-01-01"

# A draw with an unplayed match this long after it opened is an incomplete
# Wikipedia page, not a tournament in progress. Six events from 2021-2025 were
# each missing exactly one result and had been sitting on the site labelled
# "live", with a leaderboard conditioned on partial results.
STALE_AFTER_DAYS = 21

# Bump when the payload shape or any exported computation changes, so a
# rerun regenerates files that would otherwise look up to date.
EXPORT_VERSION = 10

FEATURE_NAMES = ["tier", "round", "player_a", "player_b"] + CONT_COLS

# 35 raw features → 9 drivers. SHAP is additive, so summing within a group is
# exact; the frontend never needs to know a feature name.
_DRIVERS = {
    "Rating": ["player_a_elo", "player_b_elo", "elo_diff", "elo_expected"],
    "Recent form": ["player_a_ema_form", "player_b_ema_form",
                    "player_a_recent_win_rate", "player_b_recent_win_rate",
                    "player_a_win_streak", "player_b_win_streak"],
    "Head-to-head": ["h2h_win_rate_a_vs_b", "h2h_last_winner"],
    "Seeding": ["player_a_seed", "player_b_seed"],
    "Rest & workload": ["player_a_days_since_last_match", "player_b_days_since_last_match",
                        "player_a_matches_last_7_days", "player_b_matches_last_7_days",
                        "player_a_matches_last_14_days", "player_b_matches_last_14_days"],
    "Scoring margin": ["player_a_avg_point_diff", "player_b_avg_point_diff",
                       "player_a_avg_victory_margin", "player_b_avg_victory_margin",
                       "player_a_avg_games_per_match", "player_b_avg_games_per_match",
                       "player_a_rubber_game_rate", "player_b_rubber_game_rate"],
    "Home & nation": ["player_a_is_home", "player_b_is_home", "same_nationality"],
    "Player identity": ["player_a", "player_b"],
    "Match context": ["tier", "round"],
}
DRIVER_OF = {f: d for d, fs in _DRIVERS.items() for f in fs}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def derive_status(n_played: int, n_pending: int, tour_date) -> str:
    """
    Where a tournament is in its life, from its own rows.

    Shared by the per-tournament shard and the index because they used to
    decide this separately, and drifted: the index went on calling eleven
    finished events "live" after the shards had been corrected.

    "live" gates the results-conditioned leaderboard in the frontend, so it has
    to mean the event is actually running. A draw still missing a result weeks
    after it opened is an incomplete Wikipedia page, not a match in progress.
    """
    if n_played == 0:
        return "upcoming"
    age = (pd.Timestamp.today().normalize() - pd.Timestamp(tour_date)).days
    return "live" if n_pending and age <= STALE_AFTER_DAYS else "complete"


def slugify(name: str) -> str:
    s = fold_ascii(name)
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")


def write_json(path: str, obj) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = json.dumps(obj, separators=(",", ":"))
    with open(path, "w") as f:
        f.write(blob)
    return len(blob)


def load_nat_map(raw: pd.DataFrame) -> dict:
    out = {}
    for side in ("a", "b"):
        sub = raw[[f"player_{side}", f"player_{side}_nat"]].dropna()
        for name, nat in sub.itertuples(index=False):
            out.setdefault(name, nat)
    return out


def dedupe_day(day: pd.DataFrame) -> pd.DataFrame:
    """One row per real match - the training frame carries both mirrorings."""
    seen, keep = set(), []
    for _, row in day.iterrows():
        k = (row["round"], frozenset((row["player_a"], row["player_b"])))
        if k not in seen:
            seen.add(k)
            keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame()


def fingerprint(day: pd.DataFrame, hist: pd.DataFrame) -> str:
    """
    Identify the inputs behind a tournament's export: its own rows (which change
    as a live event progresses) and all history before it (which changes on a
    rescrape). Cheap enough to compute for every tournament on every run.
    """
    h = hashlib.sha1()
    h.update(str(EXPORT_VERSION).encode())
    h.update(str(int(pd.util.hash_pandas_object(day, index=False).sum())).encode())
    h.update(str(len(hist)).encode())
    h.update(str(int(pd.util.hash_pandas_object(hist, index=False).sum())).encode())
    return h.hexdigest()[:16]


def group_shap(sv: np.ndarray) -> list:
    agg = {}
    for name, val in zip(FEATURE_NAMES[:len(sv)], sv):
        agg[DRIVER_OF[name]] = agg.get(DRIVER_OF[name], 0.0) + float(val)
    return [{"f": k, "s": round(v, 4)}
            for k, v in sorted(agg.items(), key=lambda t: -abs(t[1]))]


# ----------------------------------------------------------------------
# per-tournament export
# ----------------------------------------------------------------------
def export_tournament(cfg_row, df, raw, nat_map, fallback_payload, out_dir):
    """Returns (status, bytes) where status is 'skip' | 'write' | 'thin'."""
    import shap

    tour_date = pd.Timestamp(cfg_row["start_date"])
    date_key  = tour_date.strftime("%Y-%m-%d")
    name      = cfg_row["tournament_name"]
    tier      = int(cfg_row["tier"])
    # Keyed by name, not date: two tournaments can open on the same day.
    path      = os.path.join(out_dir, "tournament", f"{slugify(name)}.json")

    same_day = df["start_date"] == tour_date
    mine     = same_day & (df["tournament"] == name)
    day      = dedupe_day(df[mine])
    hist     = df[(df["start_date"] < tour_date) & (df["is_pending"] == 0)]
    if day.empty:
        return "thin", 0

    fp = fingerprint(day, hist)
    if os.path.exists(path):
        try:
            with open(path) as f:
                if json.load(f).get("fp") == fp:
                    return "skip", 0
        except (ValueError, OSError):
            pass  # unreadable or truncated - regenerate

    pit = train_point_in_time(df, date_key)
    if pit is None:
        payload, pre = fallback_payload
        model_label  = "preloaded (insufficient history)"
    else:
        payload, pre = pit
        model_label  = "point-in-time"

    df_t = df[~same_day | mine]          # drop a co-dated tournament's rows
    r1, stats = build_time_zero_state(df_t, date_key, tier)
    if r1.empty:
        return "thin", 0
    h2h_rate, h2h_last = build_h2h_lookups(df, date_key)
    scaler, p2i, t2i, r2i = (pre["scaler"], pre["player_to_id"],
                             pre["tier_to_id"], pre["round_to_id"])

    est = payload["model"]
    explainer = shap.TreeExplainer(est)
    n_feat = getattr(est, "n_features_in_", 0) or len(FEATURE_NAMES)

    # Seeds and scores live in the raw scrape, not the engineered frame.
    # Scores are stored winner-perspective, so they read correctly regardless
    # of which way round this frame happens to hold the pair.
    seeds, scores = {}, {}
    day_raw = raw[raw["tournament"] == name]
    for _, r in day_raw.iterrows():
        for side in ("a", "b"):
            s = r.get(f"player_{side}_seed")
            if pd.notna(s) and int(s) > 0:
                seeds[r[f"player_{side}"]] = int(s)
        sc = r.get("score")
        if pd.notna(sc):
            scores[(r["round"], frozenset((r["player_a"], r["player_b"])))] = str(sc)

    matches = []
    for _, row in day.iterrows():
        pa, pb, rnd = row["player_a"], row["player_b"], row["round"]
        # From this row's own features, not the tournament's Day-1 state, so a
        # later-round card reflects the rounds already played - see
        # row_player_state. `stats` stays Day-1 for the Monte Carlo below.
        rstats = row_player_state(row)
        p = predict_match(pa, pb, rnd, rstats, h2h_rate, h2h_last,
                          scaler, p2i, t2i, r2i, payload, tier=tier, nat_map=nat_map)

        cat, cont, _ = encode_split(row.to_frame().T, pre)
        X  = np.hstack([cat.astype(np.float64), cont])[:, :n_feat]
        sv = np.array(explainer.shap_values(X))
        if sv.ndim == 3:
            sv = sv[..., 1] if sv.shape[-1] == 2 else sv[0]

        pending = int(row.get("is_pending", 0)) == 1
        matches.append({
            "round": rnd, "a": pa, "b": pb,
            "a_nat": nat_map.get(pa, ""), "b_nat": nat_map.get(pb, ""),
            "a_seed": seeds.get(pa), "b_seed": seeds.get(pb),
            "a_elo": round(float(rstats[pa]["elo"]), 1) if pa in rstats else None,
            "b_elo": round(float(rstats[pb]["elo"]), 1) if pb in rstats else None,
            "p": round(float(p), 4),
            "pending": pending,
            "a_won": None if pending else bool(row["player_a_won"]),
            "score": "" if pending else scores.get((rnd, frozenset((pa, pb))), ""),
            # A match that ended in a retirement or a walkover. Without this the
            # frontend renders the partial score it left behind ("18-21, 6-2")
            # as if it were a completed result. These rows are excluded from
            # Elo, form, h2h and training, so the prediction beside them is
            # real but the result never fed anything back.
            "wo": int(row.get("is_walkover", 0)) == 1,
            "shap": group_shap(sv.reshape(-1)),
        })

    played = [m for m in matches if not m["pending"]]
    # Score the model only on matches that were actually contested. A retirement
    # is decided by injury, so counting it either way reads meaning into a result
    # the model's inputs could not speak to - and the rest of the pipeline
    # already treats these rows this way, excluding them from Elo, form, h2h and
    # training. Judged, not pending: they stay in `played` for status.
    judged = [m for m in played if not m["wo"]]
    hits = sum(1 for m in judged if (m["p"] > .5) == m["a_won"])
    n_pending = len(matches) - len(played)
    status = derive_status(len(played), n_pending, tour_date)

    def simulate(fixed, known=None):
        # A handful of draws are genuinely incomplete on Wikipedia. Ship the
        # bracket without a forecast rather than a fabricated one, and say so.
        try:
            counts, reached = run_monte_carlo(
                N_SIMS, r1, stats, h2h_rate, h2h_last,
                scaler, p2i, t2i, r2i, payload,
                np.random.default_rng(42), tier=tier,
                nat_map=nat_map, fixed_results=fixed, return_rounds=True,
                known_probs=known, bracket=build_bracket(day))
        except ValueError as e:
            print(f"    no simulation for {date_key}: {e}")
            return None
        # An unfilled slot can win a simulation, and did: Odisha Open 2022
        # shipped "TBD (Q1)" with a 0.6% title chance. Drop the placeholders and
        # renormalise, so the leaderboard reads as "given a real player wins".
        real  = {k: v for k, v in counts.items() if not is_placeholder(k)}
        total = sum(real.values())
        if not total:
            return None
        # Every round but the first: reaching round one is just being in the
        # draw. Deliberately NOT renormalised the way `p` is - "reaches the
        # semi-final" is a per-player marginal, not a distribution over players,
        # so it sums to the number of slots in that round rather than to 1.
        rounds_seen = [r for r in ROUND_ORDER if r in reached]
        adv_rounds  = rounds_seen[1:]

        # Everyone in the draw, not just everyone who won a simulation. Keying
        # the board off the champion counts dropped every player who never took
        # the title in 10,000 runs - fine when the only column was title odds,
        # wrong once the row also carries how far they got. Akita Masters 2018
        # lost two of its sixteen second-round entrants that way.
        entrants = [k for k in reached[rounds_seen[0]] if not is_placeholder(k)]
        board = sorted(
            ({"name": k, "nat": nat_map.get(k, ""),
              "p": round(counts.get(k, 0) / total, 4),
              "adv": [round(reached[r].get(k, 0) / N_SIMS, 4) for r in adv_rounds]}
             for k in entrants),
            key=lambda e: (-e["p"], *(-a for a in e["adv"])),
        )
        return adv_rounds, board

    # Always ship the pre-tournament forecast: conditioning a finished event on
    # its own results just returns the champion at 100%.
    pre = simulate({})
    adv_rounds, leaderboard = pre if pre else ([], None)

    # A live event additionally gets a forecast conditioned on what has actually
    # been played, so once the quarter-finals are in the site shows the model's
    # updated view of the semi-finals.
    #
    # Its still-to-play matches are quoted off the same numbers as the cards
    # above. Without this the two disagreed in public: down to a final, the
    # header's "Favourite now" and the final's own card describe the same single
    # match, and shipped 0.60 against 0.64. The simulation replays from Day 1 and
    # can only update what a scoreless result supports, so at the final it was
    # still carrying pre-tournament avg_point_diff and margins - see known_probs.
    # NOT applied to the pre-tournament board: a card's row state includes the
    # rounds already played, and `pre` is what the model thought before any of it.
    known = {(m["round"], frozenset((m["a"], m["b"]))): (m["a"], m["p"])
             for m in matches
             if m["pending"] and not is_placeholder(m["a"])
             and not is_placeholder(m["b"])}
    live = simulate(build_fixed_results(day), known) if status == "live" else None
    leaderboard_live = live[1] if live else None

    doc = {
        "fp": fp,
        "slug": slugify(name),
        "tournament": name,
        "date": date_key,
        "tier": tier,
        "host": cfg_row["host_country"],
        "model": model_label,
        # The estimator that actually produced these numbers. promote.py's
        # winner flips between libraries week to week, so it is read from
        # the payload rather than assumed anywhere downstream.
        "model_name": payload.get("name", ""),
        "trained_through": payload.get("trained_through"),
        "n_train_rows": payload.get("n_train_rows"),
        "sims": N_SIMS,
        "status": status,
        "accuracy": {"hit": hits, "n": len(judged)},
        "matches": matches,
        # Column headings for each leaderboard entry's `adv` list.
        "adv_rounds": adv_rounds,
        "leaderboard": leaderboard,
        "leaderboard_live": leaderboard_live,
    }
    return "write", write_json(path, doc)


# ----------------------------------------------------------------------
# players & matchups
# ----------------------------------------------------------------------
_STAT_COLS = {
    "is_home": "is_home", "matches_last_14_days": "matches_14d",
    "days_since_last_match": "days_since", "recent_win_rate": "recent_win_rate",
    "elo": "elo", "ema_form": "ema_form", "win_streak": "win_streak",
    "matches_last_7_days": "matches_7d", "avg_point_diff": "avg_point_diff",
    "avg_games_per_match": "avg_games_pm", "rubber_game_rate": "rubber_game_rate",
    "avg_victory_margin": "avg_margin", "seed": "seed",
}


def row_player_state(row) -> dict:
    """
    Both players' pre-match state as carried by one engineered row, in the
    shape build_time_zero_state returns.

    Day-1 state is the right starting point for the Monte Carlo, which replays
    a draw from the first round and applies its own in-bracket updates. It is
    the wrong input for a single match card: by the semi-final a player's rating
    has already moved three times, and the engineered row records exactly that.
    Predicting a card from Day-1 state threw those rounds away - a live event's
    pending semi-final, the one match the site exists to predict, was quoted off
    ratings from before the tournament began. It also disagreed with the SHAP
    attribution printed beside it, which was always built from this row.
    """
    out = {}
    for side in ("a", "b"):
        st = {}
        for col, key in _STAT_COLS.items():
            v = row.get(f"player_{side}_{col}")
            st[key] = float(v) if pd.notna(v) else 0.0
        for k in ("is_home", "matches_14d", "win_streak", "matches_7d"):
            st[k] = int(st[k])
        out[row[f"player_{side}"]] = st
    return out


def latest_player_state(df: pd.DataFrame) -> dict:
    """
    Each player's most recent feature row, in the shape build_time_zero_state
    returns. Features are pre-match values, so this is a player's form going
    into their latest appearance - the global analogue of Day-1 tournament state.

    Scanning slot A alone suffices: the frame is mirrored, so every player
    appears in slot A of some row for every match they played.

    Ordered by round, not by date alone. Every row of a tournament shares one
    `start_date`, so a plain chronological sort cannot separate a first round
    from a final and simply keeps the frame's own layout - which is
    `[originals][mirrors]` (data_loader.py concatenates the mirrored block on
    the end). "Last row seen" then landed on a player's latest appearance
    *within the mirror block*, i.e. their latest match played from slot B,
    which is a round or more behind their real latest match whenever the two
    differ. 33 of the 228 shipped players were carrying stale form, by 51 Elo
    points on average and up to 112 - and this state feeds both the player card
    and every matchup probability that player appears in.
    """
    ordered = order_by_round(df)
    out = {}
    for _, row in ordered.iterrows():
        name = row["player_a"]
        st = {}
        for col, key in _STAT_COLS.items():
            v = row.get(f"player_a_{col}")
            st[key] = float(v) if pd.notna(v) else 0.0
        for k in ("is_home", "matches_14d", "win_streak", "matches_7d"):
            st[k] = int(st[k])
        out[name] = st
    return out


def export_players(df, raw, nat_map, out_dir):
    """Current-form cards, the roster index, and each player's matchup row."""
    from src.modeling.dataset import get_train_val_datasets
    _, _, _, pre = get_train_val_datasets(DATA_PATH)
    payload = load_model()

    recent = raw[pd.to_datetime(raw["start_date"]) >= pd.Timestamp(ACTIVE_SINCE)]
    roster = sorted(set(recent["player_a"]) | set(recent["player_b"]))
    roster = [p for p in roster if isinstance(p, str) and p and p.upper() != "TBD"]

    stats = latest_player_state(df)
    roster = [p for p in roster if p in stats]

    as_of = (df["start_date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    h2h_rate, h2h_last = build_h2h_lookups(df, as_of)

    completed = df[df["is_pending"] == 0].sort_values("start_date")
    total = 0
    for name in roster:
        s = stats[name]
        mine = completed[completed["player_a"] == name].tail(5)
        card = {
            "name": name, "nat": nat_map.get(name, ""), "slug": slugify(name),
            "elo": round(s["elo"], 1),
            "ema": round(s["ema_form"], 4),
            "recent_win_rate": round(s["recent_win_rate"], 3),
            "avg_point_diff": round(s["avg_point_diff"], 2),
            "avg_margin": round(s["avg_margin"], 2),
            "rubber_rate": round(s["rubber_game_rate"], 3),
            "streak": int(s["win_streak"]),
            "form": [{"opp": r["player_b"], "won": bool(r["player_a_won"])}
                     for _, r in mine.iterrows()],
        }
        total += write_json(os.path.join(out_dir, "player", f"{slugify(name)}.json"), card)

    total += write_json(os.path.join(out_dir, "players.json"),
                        [{"name": n, "nat": nat_map.get(n, ""), "slug": slugify(n),
                          "elo": round(stats[n]["elo"], 1)} for n in roster])

    total += export_matchups(roster, stats, h2h_rate, h2h_last, pre,
                             payload, nat_map, out_dir)
    prune_stale(roster, out_dir)
    return roster, total


def prune_stale(roster, out_dir):
    """
    Drop player/matchup shards no longer in the roster.

    CI restores site/data from cache, so nothing is deleted implicitly: a
    player who drops out of the active window, or whose slug changes, would
    otherwise leave an orphan behind forever. That also breaks the publish
    gate, which asserts one matchup shard per player card.
    """
    keep = {slugify(n) + ".json" for n in roster}
    for sub in ("player", "matchup"):
        d = os.path.join(out_dir, sub)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".json") and fname not in keep:
                os.remove(os.path.join(d, fname))
                print(f"    pruned stale {sub}/{fname}")


def export_matchups(roster, stats, h2h_rate, h2h_last, pre, payload,
                    nat_map, out_dir):
    """
    Every pairing among the active roster, sharded one file per player.

    Order-invariance means P(a beats b) == 1 - P(b beats a) exactly, so each
    unordered pair is predicted once and written into both players' files.
    """
    scaler, p2i, t2i, r2i = (pre["scaler"], pre["player_to_id"],
                             pre["tier_to_id"], pre["round_to_id"])
    rows = {n: {} for n in roster}
    for i, a in enumerate(roster):
        for b in roster[i + 1:]:
            p = float(predict_match(a, b, "quarter-finals", stats,
                                    h2h_rate, h2h_last, scaler, p2i, t2i, r2i,
                                    payload, tier=750, nat_map=nat_map))
            rows[a][b] = round(p, 4)
            rows[b][a] = round(1.0 - p, 4)

    total = 0
    for name in roster:
        total += write_json(
            os.path.join(out_dir, "matchup", f"{slugify(name)}.json"),
            {"name": name,
             "vs": [{"slug": slugify(o), "name": o, "p": p}
                    for o, p in sorted(rows[name].items(), key=lambda t: -t[1])]})
    return total


# ----------------------------------------------------------------------
# index
# ----------------------------------------------------------------------
def export_index(cfg, df, out_dir):
    rows = []
    for _, c in cfg.iterrows():
        d = pd.Timestamp(c["start_date"])
        day = df[(df["start_date"] == d) & (df["tournament"] == c["tournament_name"])]
        if day.empty:
            continue
        pending = int((day["is_pending"] == 1).sum())
        played  = int((day["is_pending"] == 0).sum())
        rows.append({
            "name": c["tournament_name"],
            "slug": slugify(c["tournament_name"]),
            "date": d.strftime("%Y-%m-%d"),
            "tier": int(c["tier"]),
            "host": c["host_country"],
            "status": derive_status(played, pending, d),
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows, write_json(os.path.join(out_dir, "tournaments.json"), rows)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Export Deuce as static JSON")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--force", action="store_true", help="ignore fingerprints")
    ap.add_argument("--only", help="single tournament start date (YYYY-MM-DD)")
    ap.add_argument("--since", type=int, default=FIRST_YEAR)
    ap.add_argument("--skip-players", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    df  = load_training_frame(DATA_PATH, drop_pending=False)
    raw = pd.read_csv(RAW_PATH)
    cfg = pd.read_csv(CONFIG_PATH)
    cfg["start_date"] = pd.to_datetime(cfg["start_date"], errors="coerce")
    cfg = cfg.dropna(subset=["start_date"])
    cfg = cfg[cfg["start_date"].dt.year >= args.since].sort_values("start_date")
    if args.only:
        cfg = cfg[cfg["start_date"] == pd.Timestamp(args.only)]
    if args.force:
        tdir = os.path.join(args.out, "tournament")
        if os.path.isdir(tdir):
            for f in os.listdir(tdir):
                os.remove(os.path.join(tdir, f))

    nat_map = load_nat_map(raw)
    from src.modeling.dataset import get_train_val_datasets
    _, _, _, global_pre = get_train_val_datasets(DATA_PATH)
    fallback = (load_model(), global_pre)

    print(f"Exporting {len(cfg)} tournaments ({args.since}+) to {args.out}/")
    n_write = n_skip = n_thin = 0
    total_bytes = 0
    for _, row in cfg.iterrows():
        status, size = export_tournament(row, df, raw, nat_map, fallback, args.out)
        total_bytes += size
        if status == "write":
            n_write += 1
            print(f"  [write] {row['start_date'].date()}  {row['tournament_name']:<34} "
                  f"{size/1024:6.1f} KB")
        elif status == "skip":
            n_skip += 1
        else:
            n_thin += 1

    index, idx_bytes = export_index(cfg, df, args.out)
    total_bytes += idx_bytes

    if not args.skip_players:
        roster, pbytes = export_players(df, raw, nat_map, args.out)
        total_bytes += pbytes
        print(f"  {len(roster)} active players -> player/*.json ({pbytes/1024:.0f} KB)")

    print(f"\n{'='*62}")
    print(f"  written {n_write} · skipped {n_skip} · no data {n_thin}")
    print(f"  index: {len(index)} tournaments · total written {total_bytes/1024:.0f} KB")
    print(f"  {time.time()-t0:.1f}s")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
