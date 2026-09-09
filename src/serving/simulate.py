"""
Monte Carlo tournament simulation engine.

Used by the static exporter and runnable as a CLI:

    python3 src/serving/simulate.py --date 2026-02-24 --tier 300 --sims 10000

The engine is vectorised: each bracket round batches every pending match
across all simulations into a single predict_proba call (both slot
directions stacked into one matrix, for order-invariance), which is ~100x
faster than simulating brackets one at a time.

Completed matches of a live/partially-played tournament can be passed in
via `fixed_results` so simulations are conditioned on real outcomes.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import argparse
import pickle
import re

import numpy as np
import pandas as pd

from src.modeling.dataset import get_train_val_datasets, load_training_frame
from src.pipeline import elo as elo_model
from src.pipeline.feature_engineering import EMA_ALPHA

DATA_PATH  = "data/processed/final_training_data.csv"
MODEL_PATH = "models/best_model.pkl"

DEFAULT_TOUR_DATE = "2026-02-24"   # German Open 2026
DEFAULT_TIER      = 300
DEFAULT_N_SIMS    = 10_000

ROUND_ORDER = ["first round", "second round", "third round",
               "quarter-finals", "semi-finals", "final"]


def ladder_names(n_rounds: int) -> list[str]:
    """Name a knockout ladder of the given length.

    The last three rounds are always the quarter-final, semi-final and final;
    anything earlier is named by its position. Kept separate from
    `round_sequence` because a draw with a preliminary round has a ladder one
    longer than its opening round implies - see `build_bracket`.
    """
    n_rounds = max(1, n_rounds)
    if n_rounds <= 3:
        return ROUND_ORDER[-n_rounds:]
    return ROUND_ORDER[:n_rounds - 3] + ROUND_ORDER[-3:]


def round_sequence(n_first_round_matches: int) -> list[str]:
    """Round names for a knockout bracket that opens with the given number of
    first-round matches. 16 matches → 5 rounds, 32 matches → 6 rounds, etc.

    Rounds up: a draw missing a match (a walkover the scraper never recorded,
    say) leaves e.g. 15 openers, and truncating there returns one round too
    few - the bracket then never resolves to a single winner.
    """
    return ladder_names(int(np.ceil(np.log2(max(1, n_first_round_matches)))) + 1)


def is_placeholder(name: str) -> bool:
    """
    An unfilled draw slot ("TBD (Q1)", "Qualifier 3"), not a person.

    These reach the model with default Elo and everything else, so nothing
    downstream refuses them - they simply get predicted like anyone else. Kept
    in the bracket, because the pairing depends on the slot existing; excluded
    anywhere a name is presented as a player. The frontend applies the same
    rule when it renders a match.

    Lives here rather than in the exporter because `build_bracket` needs it to
    tell an unfilled feeder slot from a player entering the draw late.
    """
    return bool(re.search(r"\bTBD\b|qualifier", str(name), re.IGNORECASE))

def _feeder_plan(prev: list, nxt: list) -> list:
    """Slot list for a round fed by a preliminary round.

    Each of `nxt`'s two-per-match slots is either the winner of one of `prev`'s
    matches or a player entering the draw here. A slot naming someone who
    played in `prev` is that match's winner's slot; the rest of the feeders are
    still unplayed, so they show as unfilled slots and pair off against the
    leftover matches in bracket order - which is the order both lists arrive
    in, because the scraper emits cells by (column, row).
    """
    unclaimed = list(range(len(prev)))
    by_player = {}
    for j, (a, b) in enumerate(prev):
        by_player.setdefault(a, j)
        by_player.setdefault(b, j)

    slots, deferred = [], []
    for a, b in nxt:
        for name in (a, b):
            j = by_player.get(name)
            if j is not None and j in unclaimed:
                unclaimed.remove(j)
                slots.append(("w", j))
                continue
            if is_placeholder(name):
                deferred.append(len(slots))
            slots.append(("p", name))

    for pos, j in zip(deferred, unclaimed):
        slots[pos] = ("w", j)
    return slots


def build_bracket(day: pd.DataFrame):
    """
    The draw's true round ladder, and how each round's slots are filled.

    A knockout bracket normally halves - round k's winners are round k+1's
    entrants, paired in order - which is what the slot arithmetic in
    `run_monte_carlo` assumes by default. Super 100 and 300 draws break it.
    They open with a preliminary round only part of the field plays, and the
    seeds enter one round later, so 16 opening matches are followed by a
    16-match round of 32 rather than an 8-match round of 16.

    Halving from the opening round then gets the draw wrong twice: it drops a
    round off the ladder, and it deals every direct entrant out of the
    tournament altogether. Vietnam Open 2026 shipped a title race over its 32
    preliminary players with Lee Zii Jia - the highest-ranked man in the draw -
    absent, and a finished Guwahati Masters 2025 conditioned on its own results
    returned the real champion 5% of the time instead of 100%.

    Returns (rounds, plan). `rounds` is the ladder: the rounds Wikipedia has
    published, then halving on from the last of them. `plan[round]` appears
    only where a round is fed by something other than a straight halving, and
    is that round's slot list - each entry ("w", j) for the winner of match j
    of the previous round, or ("p", name) for a player entering here.
    """
    if day.empty or "round" not in day.columns:
        return [], {}
    present = [r for r in ROUND_ORDER if (day["round"] == r).any()]
    if not present:
        return [], {}

    matches = {
        r: [(row["player_a"], row["player_b"])
            for _, row in day[day["round"] == r].iterrows()]
        for r in present
    }

    # Keyed by position in the ladder, not by name: the name of a rung is not
    # known until the ladder's length is, below.
    plan = {}
    for i, (prev, nxt) in enumerate(zip(present, present[1:]), start=1):
        n_prev, n_next = len(matches[prev]), len(matches[nxt])
        # A normal round halves. It may round up - an odd opener leaves a bye,
        # which the engine already carries - so only a round *bigger* than that
        # is being fed from somewhere other than the round before it. It must
        # also be no smaller than the round it follows: a preliminary round
        # never shrinks going into the main draw. A classic-era page does,
        # because it repeats its semi-finals in both the half-bracket and the
        # Finals table and dedupe can leave three of them - All England Super
        # Series 2010 has exactly that, and without this it was read as a
        # feeder and its bracket then never resolved.
        halved = -(-n_prev // 2)
        slots = _feeder_plan(matches[prev], matches[nxt])
        wired = sorted(j for kind, j in slots if kind == "w")
        # Only trust a wiring that consumes every match of the previous round
        # exactly once. Anything else means the page did not say enough to
        # rewire this round, and the default order stands.
        if wired != list(range(n_prev)):
            continue
        if n_next >= n_prev:
            # Fed: the next round is bigger than this one, so part of its field
            # enters there. Its slot list is longer than a halving would give,
            # which is what adds the extra rung to the ladder below.
            plan[i] = slots
        elif n_next == halved and 2 * n_next == n_prev:
            # Same size as a halving, so the ladder is unaffected - but the page
            # may still pair the winners in an order the default [::2] gets
            # wrong. Akita Masters 2018 does: its round-2 rows are interleaved
            # across the two halves of the draw, so the real third round pairs
            # winner 0 with winner 2, not with winner 1. 15 finished draws have
            # a round like this. Only record it when it actually differs, so
            # the other 286 keep the exact numbers they already had.
            if slots != [("w", j) for j in range(n_prev)]:
                plan[i] = slots
        # Otherwise the page is malformed for this pair - a classic-era table
        # repeating its semi-finals leaves three of them against four
        # quarter-finals - and gets the default.

    # The ladder's length comes from the slot arithmetic rather than from how
    # many rounds the page lists: a fed round adds a rung, a duplicated row must
    # not add one, and the rounds after the last published one still have to be
    # counted. Each round halves, rounding up because an odd slot count leaves
    # a bye that the engine carries forward.
    slots, n_rungs = 2 * len(matches[present[0]]), 1
    while slots > 2 and n_rungs <= len(ROUND_ORDER) + 2:
        slots = len(plan[n_rungs]) if n_rungs in plan else -(-slots // 2)
        n_rungs += 1

    # Name the ladder by its length, exactly as round_sequence always has. The
    # published names are not used directly: a classic-era page can skip a round
    # entirely (Malaysia Open Super Series 2010 lists a first round and then
    # semi-finals), and splicing those onto a derived tail produces a ladder
    # with a repeated rung. Where the page is well formed its names *are* this
    # sequence, which `test_published_round_names_match_the_ladder` pins.
    rounds = ladder_names(n_rungs)

    # Only trust a slot plan when the page's own round names are the ladder's.
    # A few classic-era pages never got their rounds canonicalised - All England
    # Super Series Premier 2012 splits its opening round across "first round"
    # and "first round[2]" and spells the rest "quarterfinals"/"semifinals" - so
    # `present` has a hole in it, half the first round hides behind an
    # unrecognised label, and the remaining half looks like a feeder into the
    # second. Those draws were already simulated from a partial bracket; fall
    # back to that rather than invent a different wrong answer for them.
    if present != rounds[:len(present)]:
        return rounds, {}
    return rounds, {rounds[i]: v for i, v in plan.items() if i < len(rounds)}


# Order of the per-player slice of CONT_COLS held in the `static` matrix
STAT_KEYS = ["is_home", "matches_14d", "days_since", "recent_win_rate",
             "win_streak", "matches_7d", "avg_point_diff", "avg_games_pm",
             "rubber_game_rate", "avg_margin", "seed"]
STREAK_I = STAT_KEYS.index("win_streak")


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_n_features(payload):
    """Return the primary model's expected feature count, or None if unknown.
    CatBoost fitted on a plain numpy array leaves n_features_in_ at 0,
    fall back to feature_names_ in that case."""
    m = (payload["model"] if payload["type"] == "single"
         else next(iter(payload["models"].values())))
    n = getattr(m, "n_features_in_", None)
    if not n:
        names = getattr(m, "feature_names_", None)
        n = len(names) if names else None
    return n


def model_predict_proba(payload, X):
    """Supports both single-model and ensemble payloads.
    Auto-trims X to the model's expected feature count for backward compat."""
    n = get_n_features(payload)
    X_in = X[:, :n] if n is not None else X
    if payload["type"] == "ensemble":
        return sum(
            w * m.predict_proba(X_in)[:, 1]
            for w, m in zip(payload["weights"], payload["models"].values())
        )
    return payload["model"].predict_proba(X_in)[:, 1]


def build_time_zero_state(df, tour_date, tier=None):
    """
    Extract the tournament's first-round matchups plus each participant's
    pre-tournament stats exactly as they appear on Day 1.

    Player stats are collected from ALL of the tournament's rows (any round,
    completed or pending) so late-entering qualifiers are covered too.
    Mirrored duplicates are dropped: one canonical row per player pair.
    """
    day = df[df["start_date"] == pd.Timestamp(tour_date)]
    r1  = day[day["round"] == "first round"].copy()

    # Drop mirrored duplicates: keep the first row seen per unordered pair
    seen, keep = set(), []
    for _, row in r1.iterrows():
        pair = frozenset((row["player_a"], row["player_b"]))
        if pair not in seen:
            seen.add(pair)
            keep.append(row)
    r1_unique = pd.DataFrame(keep).reset_index(drop=True)

    player_stats = {}
    for _, row in day.iterrows():
        for side in ("a", "b"):
            name = row[f"player_{side}"]
            if name not in player_stats:
                player_stats[name] = {
                    "is_home":          int(row[f"player_{side}_is_home"]),
                    "matches_14d":      int(row[f"player_{side}_matches_last_14_days"]),
                    "days_since":       float(row[f"player_{side}_days_since_last_match"]),
                    "recent_win_rate":  float(row[f"player_{side}_recent_win_rate"]),
                    "elo":              float(row[f"player_{side}_elo"]),
                    "ema_form":         float(row[f"player_{side}_ema_form"]),
                    "win_streak":       int(row[f"player_{side}_win_streak"]),
                    "matches_7d":       int(row[f"player_{side}_matches_last_7_days"]),
                    "avg_point_diff":   float(row.get(f"player_{side}_avg_point_diff", 0.0)),
                    "avg_games_pm":     float(row.get(f"player_{side}_avg_games_per_match", 2.0)),
                    "rubber_game_rate": float(row.get(f"player_{side}_rubber_game_rate", 0.0)),
                    "avg_margin":       float(row.get(f"player_{side}_avg_victory_margin", 0.0)),
                    "seed":             float(row.get(f"player_{side}_seed", 0.0)),
                }

    return r1_unique, player_stats


def build_fixed_results(day_rows: pd.DataFrame) -> dict:
    """
    Map (round_name, frozenset({player_a, player_b})) → actual winner for
    every completed match of a tournament. Passed to run_monte_carlo so
    simulations of a live tournament are conditioned on real results.
    """
    out = {}
    for _, r in day_rows.iterrows():
        if int(r.get("is_pending", 0)) == 1:
            continue
        winner = r["player_a"] if r["player_a_won"] == 1 else r["player_b"]
        out[(r["round"], frozenset((r["player_a"], r["player_b"])))] = winner
    return out


def build_h2h_lookups(df, tour_date):
    """
    Pre-compute two H2H signals from all completed rows strictly before the
    given tournament date.

    Returns:
        h2h_rate_fn(pa, pb)  → float win rate of pa vs pb in [0, 1]
        h2h_last_fn(pa, pb)  → 1.0 if pa won last meeting, 0.0 if pb did, 0.5 if none
    """
    hist = df[df["start_date"] < pd.Timestamp(tour_date)]
    if "is_pending" in hist.columns:
        hist = hist[hist["is_pending"] == 0]
    # Walkovers are kept in the frame for bracket topology but were never
    # counted as history during feature engineering - match that here.
    if "is_walkover" in hist.columns:
        hist = hist[hist["is_walkover"] == 0]
    hist = hist.sort_values("start_date")

    # One pass over history, indexed by UNORDERED pair. These used to be two
    # closures that scanned the whole ~20k-row frame per call and memoised on
    # the ordered pair (pa, pb) - so a caller asking both directions of a
    # matchup, which every order-invariant prediction does, missed the cache on
    # every single call. The static export asks for 26k pairs both ways: 105k
    # full-frame scans, 191 s of the 237 s run. Indexing here makes each lookup
    # a dict read.
    #
    # `key` is the pair sorted, and `wins` counts them for the FIRST name in
    # that key, so both directions read off one entry.
    index = {}
    for pa, pb, won in zip(
        hist["player_a"].to_numpy(),
        hist["player_b"].to_numpy(),
        hist["player_a_won"].to_numpy(),
    ):
        key = (pa, pb) if pa <= pb else (pb, pa)
        winner = pa if won == 1 else pb
        entry = index.get(key)
        if entry is None:
            index[key] = [1 if winner == key[0] else 0, 1, winner]
        else:
            entry[0] += winner == key[0]
            entry[1] += 1
            entry[2] = winner          # hist is date-sorted, so last row wins

    def h2h_rate(pa, pb):
        key = (pa, pb) if pa <= pb else (pb, pa)
        entry = index.get(key)
        if entry is None:
            return 0.5
        wins = entry[0] if pa == key[0] else entry[1] - entry[0]
        return float(wins / entry[1])

    def h2h_last(pa, pb):
        key = (pa, pb) if pa <= pb else (pb, pa)
        entry = index.get(key)
        if entry is None:
            return 0.5
        return 1.0 if entry[2] == pa else 0.0

    return h2h_rate, h2h_last


def _same_nationality(pa, pb, nat_map):
    if not nat_map:
        return 0.0
    na, nb = nat_map.get(pa), nat_map.get(pb)
    return 1.0 if (na is not None and na == nb) else 0.0


def _cont_matrix(SA, SB, eA, eB, mA, mB, same, rate, last):
    """Assemble the (R, 31) continuous-feature matrix in CONT_COLS order.
    SA/SB are (R, 11) static-stat slices in STAT_KEYS order."""
    return np.column_stack([
        same, rate,
        SA[:, 0], SA[:, 1], SA[:, 2], SA[:, 3],
        SB[:, 0], SB[:, 1], SB[:, 2], SB[:, 3],
        eA, eB, eA - eB, elo_model.expected(eA, eB), mA, mB, last,
        SA[:, 4], SB[:, 4], SA[:, 5], SB[:, 5],
        SA[:, 6], SB[:, 6], SA[:, 7], SB[:, 7],
        SA[:, 8], SB[:, 8], SA[:, 9], SB[:, 9],
        SA[:, 10], SB[:, 10],
    ])


def _predict_one_direction(
    pa, pb, round_name, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, tier=None, nat_map=None,
):
    """Raw model call with pa in the player_a slot (30-feature cont vector)."""
    t        = DEFAULT_TIER if tier is None else tier
    tier_id  = tier_to_id.get(t, 0)
    round_id = round_to_id.get(round_name, 0)
    pa_id    = player_to_id.get(pa, 0)
    pb_id    = player_to_id.get(pb, 0)

    sa = player_stats[pa]
    sb = player_stats[pb]
    SA = np.array([[sa[k] for k in STAT_KEYS]], dtype=np.float64)
    SB = np.array([[sb[k] for k in STAT_KEYS]], dtype=np.float64)

    cont_raw = _cont_matrix(
        SA, SB,
        np.array([sa["elo"]]), np.array([sb["elo"]]),
        np.array([sa["ema_form"]]), np.array([sb["ema_form"]]),
        np.array([_same_nationality(pa, pb, nat_map)]),
        np.array([h2h_rate_fn(pa, pb)]),
        np.array([h2h_last_fn(pa, pb)]),
    )

    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_id, round_id, pa_id, pb_id]], dtype=np.int64)
    X   = np.hstack([cat, cont_scaled])
    return float(model_predict_proba(model_payload, X)[0])


def predict_match(
    pa, pb, round_name, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, tier=None, nat_map=None,
):
    """
    Order-invariant win probability for pa beating pb.
    Averages both slot assignments so P(A beats B) == 1 - P(B beats A) exactly.
    """
    p_ab = _predict_one_direction(
        pa, pb, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
        scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier, nat_map,
    )
    p_ba = _predict_one_direction(
        pb, pa, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
        scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier, nat_map,
    )
    return (p_ab + (1.0 - p_ba)) / 2.0


def run_monte_carlo(
    n_sims, r1_matchups, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, rng, tier=None,
    nat_map=None, fixed_results=None, progress_cb=None,
    return_rounds=False, known_probs=None, bracket=None,
):
    """
    Vectorised Monte Carlo over n_sims brackets.

    Per round, every match across all simulations is batched into one
    predict_proba call (both slot directions stacked, order-invariant
    averaging). In-bracket Elo/EMA/streak updates are applied per simulation
    via (n_sims, n_players) arrays so form carries into later rounds.

    fixed_results: {(round_name, frozenset({a, b})): winner} - real outcomes
    of already-played matches; these override the model and are applied
    deterministically in every simulation.

    known_probs: {(round_name, frozenset({a, b})): (player, p)} - the
    probability a *scheduled but unplayed* match was already quoted at, with
    `p` given from `player`'s side. This engine reconstructs a player's state
    by replaying the bracket from Day 1, which is all it can do for a match
    between two hypothetical winners; but once a pairing is real the pipeline
    has engineered a row for it, and that row's state beats anything this
    reconstruction can produce. Where such a row exists, use its number.
    Overridden by fixed_results: a played match is certain.

    progress_cb(round_name, round_idx, n_rounds): optional UI hook.

    return_rounds: also return how often each player *reached* each round.
    The simulation already knows this - the slot array at the top of a round
    is exactly its entrants - but the title count alone throws it away.

    Returns: {player_name: n_titles_won}, or that plus
    {round_name: {player_name: n_sims_reached}} when return_rounds is set.
    """
    t = DEFAULT_TIER if tier is None else tier
    # Everyone in a draw is an established player by definition, so the
    # provisional-K branch never applies to an in-bracket update.
    K = elo_model.k_for(t, elo_model.PROVISIONAL_N)
    fixed_results = fixed_results or {}
    known_probs   = known_probs or {}

    players = sorted(player_stats)
    P = len(players)
    pidx = {p: i for i, p in enumerate(players)}

    static   = np.array([[player_stats[p][k] for k in STAT_KEYS] for p in players],
                        dtype=np.float64)
    vocab_id = np.array([player_to_id.get(p, 0) for p in players], dtype=np.int64)

    E = np.tile(np.array([player_stats[p]["elo"] for p in players]), (n_sims, 1))
    M = np.tile(np.array([player_stats[p]["ema_form"] for p in players]), (n_sims, 1))
    # Win streak carries through the bracket for the same reason Elo and EMA do:
    # a simulation that has a player winning four straight matches must not go
    # on telling the model they are on the losing run they arrived with. Unlike
    # the scoreline-derived stats below it needs no score to update, only a
    # winner, so it is exactly the pipeline's rule rather than an approximation.
    W = np.tile(np.array([player_stats[p]["win_streak"] for p in players],
                         dtype=np.float64), (n_sims, 1))

    slots = []
    for _, row in r1_matchups.iterrows():
        slots += [pidx[row["player_a"]], pidx[row["player_b"]]]
    current = np.tile(np.array(slots, dtype=np.int64), (n_sims, 1))

    tier_id = tier_to_id.get(t, 0)
    # `bracket` carries the ladder read off the page plus, for any round fed by
    # something other than a straight halving, that round's slot list. Without
    # it the ladder is derived by halving from the opening round, which is right
    # for a draw where everybody enters in round one - see build_bracket.
    if bracket is not None and bracket[0]:
        rounds, slot_plan = bracket
    else:
        rounds, slot_plan = round_sequence(len(r1_matchups)), {}
    n_rounds_total = len(rounds)

    # None until a round actually reduces the bracket to one slot. Seeding this
    # with current[:, 0] would report the first player of the first match as a
    # 100%-certain champion whenever the bracket fails to resolve - a confident
    # wrong answer instead of a visible failure.
    champions = None
    reached = {}
    for round_i, round_name in enumerate(rounds):
        # Entrants of this round, counted before any carry trim. A player holds
        # at most one slot per simulation - winners of distinct matches are
        # distinct people - so a plain bincount over the slot array is the
        # number of simulations in which they got this far.
        if return_rounds:
            reached[round_name] = np.bincount(current.ravel(), minlength=P)

        # Defensive: an odd slot count means the bracket is malformed
        # (a dropped slot somewhere) - give the trailing player a bye
        # rather than crashing on mismatched pairing arrays.
        carry = None
        if current.shape[1] % 2 == 1:
            carry = current[:, -1]
            current = current[:, :-1]
        n_matches = current.shape[1] // 2
        if n_matches == 0:
            break

        A = current[:, 0::2].ravel()          # (R,) player indices in slot a
        B = current[:, 1::2].ravel()
        R = A.shape[0]
        sim_idx = np.repeat(np.arange(n_sims), n_matches)

        # Pair-level features computed once per unique (a, b) pair
        key = A.astype(np.int64) * P + B
        uniq, inv = np.unique(key, return_inverse=True)
        n_u = len(uniq)
        rate_ab = np.empty(n_u); last_ab = np.empty(n_u)
        rate_ba = np.empty(n_u); last_ba = np.empty(n_u)
        same_u  = np.empty(n_u)
        fixed_u = np.full(n_u, -1.0)          # -1 = no real result on record
        known_u = np.full(n_u, -1.0)          # -1 = no quoted card for this pair
        for k, kk in enumerate(uniq):
            a, b = divmod(int(kk), P)
            pa_n, pb_n = players[a], players[b]
            rate_ab[k] = h2h_rate_fn(pa_n, pb_n)
            last_ab[k] = h2h_last_fn(pa_n, pb_n)
            rate_ba[k] = h2h_rate_fn(pb_n, pa_n)
            last_ba[k] = h2h_last_fn(pb_n, pa_n)
            same_u[k]  = _same_nationality(pa_n, pb_n, nat_map)
            winner = fixed_results.get((round_name, frozenset((pa_n, pb_n))))
            if winner is not None:
                fixed_u[k] = 1.0 if winner == pa_n else 0.0
            quote = known_probs.get((round_name, frozenset((pa_n, pb_n))))
            if quote is not None:
                ref, prob = quote
                known_u[k] = prob if ref == pa_n else 1.0 - prob

        SA, SB = static[A].copy(), static[B].copy()
        SA[:, STREAK_I] = W[sim_idx, A]
        SB[:, STREAK_I] = W[sim_idx, B]
        eA, eB = E[sim_idx, A], E[sim_idx, B]
        mA, mB = M[sim_idx, A], M[sim_idx, B]

        cont1 = _cont_matrix(SA, SB, eA, eB, mA, mB, same_u[inv], rate_ab[inv], last_ab[inv])
        cont2 = _cont_matrix(SB, SA, eB, eA, mB, mA, same_u[inv], rate_ba[inv], last_ba[inv])

        round_id = round_to_id.get(round_name, 0)
        cat1 = np.column_stack([np.full(R, tier_id), np.full(R, round_id),
                                vocab_id[A], vocab_id[B]])
        cat2 = np.column_stack([np.full(R, tier_id), np.full(R, round_id),
                                vocab_id[B], vocab_id[A]])

        cont = scaler.transform(np.vstack([cont1, cont2]))
        X    = np.hstack([np.vstack([cat1, cat2]).astype(np.float64), cont])
        probs = model_predict_proba(model_payload, X)
        p = (probs[:R] + (1.0 - probs[R:])) / 2.0

        # A scheduled-but-unplayed match already has an engineered row, and the
        # probability quoted off it is strictly better informed than the one
        # reconstructed here - see known_probs.
        kn = known_u[inv]
        p = np.where(kn >= 0.0, kn, p)

        # Real results (live/finished tournaments) override the model
        fx = fixed_u[inv]
        p = np.where(fx >= 0.0, fx, p)

        a_wins  = rng.random(R) < p
        winners = np.where(a_wins, A, B)
        losers  = np.where(a_wins, B, A)

        # In-bracket Elo/EMA/streak updates (each player plays once per
        # round/sim, so the fancy-indexed assignments never collide)
        # The margin-of-victory multiplier has no counterpart here: a simulated
        # match has a winner but no scoreline, so the update uses the plain K.
        elo_w, elo_l = E[sim_idx, winners], E[sim_idx, losers]
        exp_w = elo_model.expected(elo_w, elo_l)
        E[sim_idx, winners] = elo_w + K * (1.0 - exp_w)
        E[sim_idx, losers]  = elo_l - K * (1.0 - exp_w)
        M[sim_idx, winners] = EMA_ALPHA + (1 - EMA_ALPHA) * M[sim_idx, winners]
        M[sim_idx, losers]  = (1 - EMA_ALPHA) * M[sim_idx, losers]
        # Same rule as _elo_prepass: a win extends a winning run or starts one,
        # a loss extends a losing run or starts one.
        str_w, str_l = W[sim_idx, winners], W[sim_idx, losers]
        W[sim_idx, winners] = np.maximum(str_w, 0.0) + 1.0
        W[sim_idx, losers]  = np.minimum(str_l, 0.0) - 1.0

        won = winners.reshape(n_sims, n_matches)
        nxt = rounds[round_i + 1] if round_i + 1 < len(rounds) else None
        slots_next = slot_plan.get(nxt) if nxt else None
        if slots_next is None:
            current = won
            if carry is not None:
                current = np.column_stack([current, carry])
        else:
            # This round feeds only part of the next one; its other slots hold
            # players entering the draw here. Assemble the next round from the
            # published pairing instead of from this round's winners alone -
            # halving would drop those entrants out of the tournament.
            cols = []
            for kind, val in slots_next:
                if kind == "w":
                    if not 0 <= val < n_matches:
                        raise ValueError(
                            f"{nxt} is fed by match {val} of {round_name}, "
                            f"which only has {n_matches}")
                    cols.append(won[:, val])
                else:
                    seat = pidx.get(val)
                    if seat is None:
                        raise ValueError(
                            f"{nxt} names {val!r}, who has no state in this draw")
                    cols.append(np.full(n_sims, seat, dtype=np.int64))
            current = np.column_stack(cols)
        if progress_cb:
            progress_cb(round_name, round_i + 1, n_rounds_total)
        if current.shape[1] == 1:
            champions = current[:, 0]
            break

    if champions is None:
        raise ValueError(
            f"Bracket never resolved to a single winner: {len(r1_matchups)} "
            f"first-round matchups left {current.shape[1]} slots after "
            f"{len(rounds)} rounds. The draw is incomplete."
        )

    idx, counts = np.unique(champions, return_counts=True)
    titles = {players[i]: int(c) for i, c in zip(idx, counts)}
    if not return_rounds:
        return titles
    return titles, {
        rnd: {players[i]: int(c) for i, c in enumerate(vec) if c}
        for rnd, vec in reached.items()
    }


def run(tour_date: str, tier: int, n_sims: int,
        data_path: str = DATA_PATH, model_path: str = MODEL_PATH,
        condition: bool = False):
    print("Loading data and model...")
    df = load_training_frame(data_path, drop_pending=False)

    model_payload = load_model(model_path)
    model_name    = model_payload.get("name", model_payload.get("type", "unknown"))
    print(f"Model: {model_name}")

    _, _, _, preprocessors = get_train_val_datasets(data_path)

    r1_matchups, player_stats = build_time_zero_state(df, tour_date, tier)
    if r1_matchups.empty:
        print(f"ERROR: no first-round rows found for {tour_date}. "
              f"Is the tournament in {data_path}?")
        sys.exit(1)

    h2h_rate_fn, h2h_last_fn = build_h2h_lookups(df, tour_date)
    fixed = {}
    if condition:
        day   = df[df["start_date"] == pd.Timestamp(tour_date)]
        fixed = build_fixed_results(day)
        print(f"Conditioning on {len(fixed)} real results already on record.")

    print(f"\n{'='*62}")
    print(f"  {tour_date} - First Round Bracket ({len(r1_matchups)} matchups)")
    print(f"{'='*62}")
    for _, row in r1_matchups.iterrows():
        p = predict_match(
            row["player_a"], row["player_b"], "first round",
            player_stats, h2h_rate_fn, h2h_last_fn,
            preprocessors["scaler"], preprocessors["player_to_id"],
            preprocessors["tier_to_id"], preprocessors["round_to_id"],
            model_payload, tier=tier,
        )
        print(f"  {row['player_a']:30s} vs {row['player_b']:30s}  | P(A wins)={p:.3f}")
    print(f"{'='*62}")

    print(f"\nRunning {n_sims:,} simulations...")
    rng = np.random.default_rng(42)
    win_counts = run_monte_carlo(
        n_sims, r1_matchups, player_stats,
        h2h_rate_fn, h2h_last_fn,
        preprocessors["scaler"], preprocessors["player_to_id"],
        preprocessors["tier_to_id"], preprocessors["round_to_id"],
        model_payload, rng, tier=tier, fixed_results=fixed,
        progress_cb=lambda name, i, n: print(f"  [{i}/{n}] {name} simulated"),
    )

    leaderboard = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*54}")
    print(f"  Championship Probability Leaderboard ({n_sims:,} sims)")
    print(f"{'='*54}")
    print(f"  {'Player':<32} {'Win %':>7}")
    print(f"  {'-'*32}  {'-'*7}")
    for name, wins in leaderboard:
        print(f"  {name:<32} {wins/n_sims*100:>6.2f}%")
    print(f"{'='*54}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo BWF tournament simulation")
    parser.add_argument("--date",  default=DEFAULT_TOUR_DATE,
                        help=f"Tournament start date YYYY-MM-DD (default: {DEFAULT_TOUR_DATE})")
    parser.add_argument("--tier",  type=int, default=DEFAULT_TIER,
                        help=f"Tournament tier (default: {DEFAULT_TIER})")
    parser.add_argument("--sims",  type=int, default=DEFAULT_N_SIMS,
                        help=f"Number of simulations (default: {DEFAULT_N_SIMS:,})")
    parser.add_argument("--data",  default=DATA_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--condition", action="store_true",
                        help="Fix already-played matches to their real outcome "
                             "(default: pure pre-tournament forecast)")
    args = parser.parse_args()
    run(args.date, args.tier, args.sims, args.data, args.model, args.condition)
