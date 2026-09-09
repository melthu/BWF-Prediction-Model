"""
Data pipeline invariants: mirroring, leakage, and the calendar gate.
"""
import pandas as pd
import pytest

from src.pipeline.data_checks import check_config
from src.pipeline.data_loader import SWAP_PAIRS


def test_mirroring_swaps_per_player_features_without_negating(df):
    """
    The bug this pins down: `player_a_avg_point_diff` was negated *on top of*
    the A<->B swap. It is a per-player stat, so the swap alone is correct; the
    extra negation left slot A wrong-signed on every mirrored row while slot B
    was untouched, breaking the order-invariance mirroring exists to create.

    Every row has a mirrored counterpart, so for each per-player pair the
    multiset of (A value, B value) must equal the multiset of (B value, A value).
    """
    for col_a, col_b in SWAP_PAIRS:
        if col_a not in df.columns:
            continue
        rounded = (lambda s: s.round(6)) if pd.api.types.is_numeric_dtype(df[col_a]) \
            else (lambda s: s)
        a = rounded(df[col_a]).sort_values().reset_index(drop=True)
        b = rounded(df[col_b]).sort_values().reset_index(drop=True)
        pd.testing.assert_series_equal(
            a, b, check_names=False,
            obj=f"{col_a} vs {col_b} - mirrored per-player values must match",
        )


def test_pair_level_features_are_inverted_not_swapped(df):
    """
    The other half of the rule: features describing the *pair* do get inverted.
    elo_diff is A minus B, so its distribution must be symmetric about zero
    once every row is mirrored.
    """
    assert abs(df["elo_diff"].sum()) < 1e-6, "elo_diff is not sign-symmetric across mirrors"
    assert df["player_a_won"].mean() == pytest.approx(0.5, abs=1e-9), (
        "labels are not balanced - mirroring should make each match appear as both outcomes"
    )


def test_pending_rows_carry_features_but_are_excluded_from_training(df):
    """
    Pending matches are published draws with no result. They must get features
    (the site predicts them) but never become training rows.
    """
    from src.modeling.dataset import load_training_frame

    pending = df[df["is_pending"] == 1]
    if pending.empty:
        pytest.skip("no pending matches in the current data")
    assert pending["player_a_elo"].notna().all(), "pending rows must still get features"
    assert (load_training_frame()["is_pending"] == 0).all(), (
        "load_training_frame leaked pending rows into training"
    )


def test_config_check_accepts_the_real_calendar():
    assert check_config() == []


def test_config_check_catches_a_single_lost_season(cfg, tmp_path, monkeypatch):
    """
    The gap this closes. build_config.py refuses a config that shrank by more
    than 5%, but 2021 is 16 of 357 rows - 4.5%, under the threshold. A whole
    season could vanish while the total still looked healthy.
    """
    from src.pipeline import data_checks

    years = pd.to_datetime(cfg["start_date"]).dt.year
    gutted = cfg[years != 2021]
    assert len(gutted) >= 0.95 * len(cfg), (
        "premise no longer holds: losing 2021 now trips build_config's own guard"
    )

    path = tmp_path / "cfg.csv"
    gutted.to_csv(path, index=False)
    monkeypatch.setattr(data_checks, "CONFIG_PATH", str(path))

    errors = data_checks.check_config()
    assert any("2021" in e for e in errors), errors


def test_a_players_rounds_are_scanned_in_playing_order(raw):
    """
    A Wikipedia bracket page carries its Finals table (semi-finals + final)
    ABOVE the section tables, so the scraper emits those rows first. Every row
    of a tournament shares one start_date, so the stable chronological sort
    could not separate them, and the Elo prepass - a single sequential scan -
    reached a semi-final before the quarter-final that produced it: 217 of 310
    tournaments had at least one player whose rows were scanned out of order.

    Ordering is a property of the frame the prepasses consume, so this asserts
    on the output of order_by_round rather than on the raw scrape.
    """
    from src.pipeline.feature_engineering import order_by_round, _round_rank

    ordered = order_by_round(raw.assign(
        start_date=pd.to_datetime(raw["start_date"])))

    offenders = []
    for (name, date), g in ordered.groupby(["tournament", "start_date"], sort=False):
        ranks = [_round_rank(r) for r in g["round"]]
        if any(r is None for r in ranks):
            continue          # not a knockout ladder; deliberately left alone
        seen = {}
        for pos, (_, row) in enumerate(g.iterrows()):
            rank = _round_rank(row["round"])
            for side in ("a", "b"):
                p = row[f"player_{side}"]
                if p in seen and rank < seen[p]:
                    offenders.append(f"{name} {date:%Y-%m-%d}: {p} "
                                     f"{row['round']!r} scanned after a later round")
                seen[p] = max(seen.get(p, rank), rank)
    assert not offenders, offenders[:10]


def test_order_by_round_preserves_first_round_bracket_order(raw):
    """
    run_monte_carlo derives the whole draw's topology from the order of the
    first-round pairings, so the round sort must be stable within a round.
    """
    from src.pipeline.feature_engineering import order_by_round

    src = raw.assign(start_date=pd.to_datetime(raw["start_date"]))
    ordered = order_by_round(src)
    for (name, date), g in src.groupby(["tournament", "start_date"], sort=False):
        before = [(r.player_a, r.player_b) for r in
                  g[g["round"].str.lower().str.startswith("first")].itertuples()]
        after = [(r.player_a, r.player_b) for r in
                 ordered[(ordered["tournament"] == name)
                         & (ordered["start_date"] == date)
                         & (ordered["round"].str.lower().str.startswith("first"))
                         ].itertuples()]
        assert before == after, f"{name} {date:%Y-%m-%d}: first-round order changed"


def test_bracket_rounds_covers_every_alias():
    """
    The scraper's whole-page mode keeps only rows whose round is a real rung of
    a knockout ladder, and it matches on the header text as scraped - lowercased
    but NOT canonicalised. So BRACKET_ROUNDS has to name every spelling that
    ROUND_ALIASES would canonicalise into a rung, not just the six canonical
    ones. Writing it in canonical names alone silently deleted all 48 quarter-
    and semi-finals of the 2010-2019 World Championships, which spell them
    "Quarterfinals"/"Semifinals"; the filter dropped them and the corpus simply
    had no such matches. Nothing else would have noticed - the rows were never
    wrong, they were absent.
    """
    from src.pipeline.feature_engineering import ROUND_ALIASES, ROUND_RANK
    from src.pipeline.scraper_wiki_single import BRACKET_ROUNDS

    missing = {
        raw for raw, canon in ROUND_ALIASES.items()
        if canon in ROUND_RANK and raw not in BRACKET_ROUNDS
    }
    assert not missing, (
        "ROUND_ALIASES canonicalises these into ladder rungs, but the scraper's "
        "whole-page filter would drop them: " + ", ".join(sorted(missing))
    )
    # ...and the canonical names themselves, which pages also use verbatim.
    assert set(ROUND_RANK) <= set(BRACKET_ROUNDS)
