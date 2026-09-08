"""
The serving path: prediction, simulation, and SHAP attribution.

This is where the retired Streamlit dashboard used to act as a second opinion.
With `app.py` gone the exporter is the only implementation, so these tests
assert the invariants that a disagreement between the two would have exposed.
"""
import numpy as np
import pandas as pd
import pytest

from src.serving.export_static import FEATURE_NAMES, DRIVER_OF, group_shap
from src.serving.simulate import (
    ROUND_ORDER, build_fixed_results, predict_match, round_sequence, run_monte_carlo,
)

SIMS = 200  # enough to check invariants; the real export uses 10,000


def _predict(f, pa, pb, rnd="first round"):
    return predict_match(
        pa, pb, rnd, f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], tier=f["tier"], nat_map=f["nat_map"],
    )


def test_prediction_is_order_invariant(fitted):
    """
    P(A beats B) must equal 1 - P(B beats A) exactly. The model itself is not
    symmetric, so this holds only because predict_match averages both slot
    assignments; if that averaging is dropped, the same match gets two
    different answers depending on which way the scraper happened to store it.
    """
    for _, row in fitted["day"].head(12).iterrows():
        pa, pb, rnd = row["player_a"], row["player_b"], row["round"]
        assert _predict(fitted, pa, pb, rnd) + _predict(fitted, pb, pa, rnd) == pytest.approx(1.0)


def test_predictions_are_probabilities(fitted):
    for _, row in fitted["day"].iterrows():
        p = _predict(fitted, row["player_a"], row["player_b"], row["round"])
        assert 0.0 <= p <= 1.0


def test_same_nationality_is_actually_reaching_the_model(fitted):
    """
    Omitting nat_map silently zeroes the same_nationality feature instead of
    failing, so the only way to notice is that predictions stop moving.
    """
    f = fitted
    pairs = [(r["player_a"], r["player_b"]) for _, r in f["day"].iterrows()]
    with_nat = [_predict(f, a, b) for a, b in pairs]
    without = [predict_match(a, b, "first round", f["stats"], f["h2h_rate"], f["h2h_last"],
                             f["pre"]["scaler"], f["pre"]["player_to_id"],
                             f["pre"]["tier_to_id"], f["pre"]["round_to_id"],
                             f["payload"], tier=f["tier"], nat_map=None)
               for a, b in pairs]
    assert with_nat != without, "nat_map made no difference - same_nationality is not wired up"


@pytest.mark.parametrize("n_first_round, expected_rounds", [
    (16, 5), (32, 6), (8, 4), (4, 3), (2, 2),
    # A draw missing a match must round *up*: truncating leaves the bracket
    # one round short and it never resolves to a single winner.
    (15, 5), (31, 6),
])
def test_round_sequence_resolves_to_one_winner(n_first_round, expected_rounds):
    seq = round_sequence(n_first_round)
    assert len(seq) == expected_rounds
    assert seq[-1] == ROUND_ORDER[-1]
    assert all(r in ROUND_ORDER for r in seq)


def test_monte_carlo_is_a_distribution_over_the_draw(fitted):
    f = fitted
    counts = run_monte_carlo(
        SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
        tier=f["tier"], nat_map=f["nat_map"],
    )
    assert sum(counts.values()) == SIMS, "championship probabilities must sum to 1"

    entrants = set(f["r1"].to_numpy().ravel()) if hasattr(f["r1"], "to_numpy") else set()
    if entrants:
        assert set(counts) <= entrants, "a player who is not in the draw won it"

    # The exact failure of the old seeding bug: `champions = current[:, 0]` left
    # an unresolved bracket reporting the first player as a 100% champion.
    assert max(counts.values()) < SIMS, "one player won every simulation - bracket did not resolve"


def test_monte_carlo_round_counts_fill_every_slot(fitted):
    """
    `return_rounds` reports who reached each round. A round has a fixed number
    of slots and every one of them is occupied in every simulation, so the
    counts for a round must total exactly slots x sims - the check that
    separates "entrants of this round" from "winners of this round", which is
    an off-by-one that would otherwise look plausible.
    """
    f = fitted
    titles, reached = run_monte_carlo(
        SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
        tier=f["tier"], nat_map=f["nat_map"], return_rounds=True,
    )
    assert sum(titles.values()) == SIMS

    slots = 2 * len(f["r1"])
    for rnd in ROUND_ORDER:
        if rnd not in reached:
            continue
        assert sum(reached[rnd].values()) == slots * SIMS, (
            f"{rnd}: counted {sum(reached[rnd].values())} appearances, "
            f"expected {slots * SIMS} ({slots} slots x {SIMS} sims)")
        slots //= 2

    # Reaching the final is implied by winning it.
    final = reached.get("final", {})
    for player, n in titles.items():
        assert final.get(player, 0) >= n, f"{player} won more finals than they reached"


def test_monte_carlo_is_deterministic_under_a_fixed_seed(fitted):
    """The exporter seeds with 42; a rerun that shifts the numbers would churn
    every shard's fingerprint and republish the whole site."""
    f = fitted
    args = (SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
            f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
            f["pre"]["round_to_id"], f["payload"])
    kw = dict(tier=f["tier"], nat_map=f["nat_map"])
    a = run_monte_carlo(*args, np.random.default_rng(42), **kw)
    b = run_monte_carlo(*args, np.random.default_rng(42), **kw)
    assert a == b


def test_fixed_results_override_the_model(fitted):
    """
    Conditioning a finished tournament on its own results must return the real
    champion at 100% - that is what makes a live draw's odds trustworthy.
    """
    f = fitted
    fixed = build_fixed_results(f["day"])
    counts = run_monte_carlo(
        SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
        tier=f["tier"], nat_map=f["nat_map"], fixed_results=fixed,
    )
    final = f["day"][f["day"]["round"] == ROUND_ORDER[-1]].iloc[0]
    champion = final["player_a"] if final["player_a_won"] == 1 else final["player_b"]
    assert counts.get(champion, 0) == SIMS, (
        f"expected {champion} to win every conditioned simulation, got {counts}"
    )


def test_a_preliminary_round_does_not_swallow_the_draw(bye_draw):
    """
    build_bracket must find the extra round and account for every slot.

    A draw with a preliminary round runs 16 -> 16 -> 8 -> 4 -> 2 -> 1, not the
    16 -> 8 -> 4 -> 2 -> 1 that halving from the opener gives. Getting it wrong
    drops a round *and* deals out everyone who entered after round one.
    """
    from src.serving.simulate import build_bracket, round_sequence

    _, day = bye_draw
    rounds, plan = build_bracket(day)

    n1 = int((day["round"] == "first round").sum())
    assert len(rounds) == len(round_sequence(n1)) + 1, (
        f"a fed round adds a rung: got {rounds}")
    assert rounds[-3:] == ["quarter-finals", "semi-finals", "final"]
    assert "second round" in plan, "the round after the opener is the fed one"

    slots = plan["second round"]
    assert len(slots) == 2 * int((day["round"] == "second round").sum())
    fed = [j for kind, j in slots if kind == "w"]
    assert sorted(fed) == list(range(n1)), (
        "every opening match must feed exactly one slot, none twice")
    entering = [v for kind, v in slots if kind == "p"]
    assert len(entering) == len(slots) - n1
    assert len(set(entering)) == len(entering), "a player cannot hold two slots"


def test_a_bye_draw_conditioned_on_its_results_returns_its_real_champion(fitted_bye,
                                                                        bye_draw):
    """
    The same invariant as test_fixed_results_override_the_model, on the draw
    shape that used to break it.

    Before build_bracket this failed loudly on real data: Guwahati Masters 2025
    gave its actual champion 109 sims in 2000, Odisha Masters 2025 gave its
    champion none at all, because the direct entrants were never dealt into the
    bracket and the fixed results were being matched against the wrong rounds.
    """
    from src.serving.simulate import build_bracket

    f = fitted_bye
    _, day = bye_draw
    counts = run_monte_carlo(
        SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
        tier=f["tier"], nat_map=f["nat_map"],
        fixed_results=build_fixed_results(f["day"]),
        bracket=build_bracket(day),
    )
    final = f["day"][f["day"]["round"] == ROUND_ORDER[-1]].iloc[0]
    champion = final["player_a"] if final["player_a_won"] == 1 else final["player_b"]
    assert counts.get(champion, 0) == SIMS, (
        f"expected {champion} to win every conditioned simulation, got "
        f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")


def test_a_full_draw_bracket_is_unchanged(tournament):
    """
    build_bracket must be a no-op where the old halving was already right.

    The fix is meant to reach 30 tournaments, not all 226 - if it moved a
    normal draw it would rewrite every shard's numbers for no reason.
    """
    from src.serving.simulate import build_bracket, round_sequence

    _, day = tournament
    rounds, plan = build_bracket(day)
    n1 = int((day["round"] == "first round").sum())
    assert rounds == round_sequence(n1)
    assert plan == {}, f"a full draw needs no slot plan, got {list(plan)}"


def test_every_feature_maps_to_exactly_one_driver():
    """
    A feature added to CONT_COLS without a driver raises KeyError mid-export,
    after work is already on disk. Catch it here instead.
    """
    missing = [f for f in FEATURE_NAMES if f not in DRIVER_OF]
    assert not missing, f"features with no SHAP driver: {missing}"
    # 35 since elo_expected joined the Rating driver; update deliberately, so
    # that a feature appearing by accident still trips this.
    assert len(FEATURE_NAMES) == 35


def test_grouped_shap_is_exact():
    """SHAP is additive, so grouping must preserve the total exactly."""
    rng = np.random.default_rng(0)
    sv = rng.normal(size=len(FEATURE_NAMES))
    grouped = group_shap(sv)
    assert sum(g["s"] for g in grouped) == pytest.approx(sv.sum(), abs=1e-3)
    assert [abs(g["s"]) for g in grouped] == sorted((abs(g["s"]) for g in grouped), reverse=True)


def test_known_probs_are_what_the_board_quotes(fitted):
    """
    A scheduled-but-unplayed match must simulate at the probability its own
    card carries, from either slot.

    Down to a final the leaderboard's top entry and the final's card describe
    one single match, and they disagreed in public - the header said 0.60 while
    the card beside it said 0.64. The simulation replays from Day 1 and can only
    update what a scoreless result supports, so at the final it was still
    carrying pre-tournament scoring margins. Where the pipeline has engineered a
    row for the pairing, that row wins.
    """
    f = fitted
    fixed = build_fixed_results(f["day"])
    final = f["day"][f["day"]["round"] == ROUND_ORDER[-1]].iloc[0]
    pa, pb = final["player_a"], final["player_b"]
    fixed.pop((ROUND_ORDER[-1], frozenset((pa, pb))), None)

    def titles(known):
        counts = run_monte_carlo(
            SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
            f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
            f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
            tier=f["tier"], nat_map=f["nat_map"], fixed_results=fixed,
            known_probs=known,
        )
        return counts.get(pa, 0) / SIMS

    quoted = 0.8
    key = (ROUND_ORDER[-1], frozenset((pa, pb)))
    got_a = titles({key: (pa, quoted)})
    got_b = titles({key: (pb, 1.0 - quoted)})

    # 200 draws of a coin: 3 binomial standard errors is ~0.085
    assert abs(got_a - quoted) < 0.085, f"quoted {quoted}, simulated {got_a}"
    assert got_a == got_b, (
        f"the same quote from the other slot gave {got_b} instead of {got_a}"
    )


def test_in_bracket_win_streak_follows_the_pipeline_rule(fitted):
    """
    A simulation that has a player win four straight matches must not go on
    telling the model they are on the losing run they arrived with. Streak is
    the one stale stat the engine can fix exactly: it needs a winner, not a
    scoreline. Checked through the champion's feature vector at the final.
    """
    from src.serving import simulate as sim

    f = fitted
    seen = []
    orig = sim._cont_matrix

    def spy(SA, SB, *rest):
        seen.append((SA[:, sim.STREAK_I].copy(), SB[:, sim.STREAK_I].copy()))
        return orig(SA, SB, *rest)

    sim._cont_matrix = spy
    try:
        run_monte_carlo(
            SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
            f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
            f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
            tier=f["tier"], nat_map=f["nat_map"],
            fixed_results=build_fixed_results(f["day"]),
        )
    finally:
        sim._cont_matrix = orig

    # Both finalists reached the final on four straight wins, whatever they
    # walked into the draw on, so the streak the model sees there is >= 4.
    fa, fb = seen[-2]          # last round, slot-a direction
    assert fa.min() >= 4 and fb.min() >= 4, (
        f"streaks at the final were {fa.min()} / {fb.min()}, not the four wins "
        "it took to get there - the in-bracket update is not being applied"
    )
