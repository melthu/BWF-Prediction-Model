import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.dataset import get_train_val_datasets, CONT_COLS
from src.simulate_german_open import (
    ROUND_ORDER,
    model_predict_proba,
    build_time_zero_state,
    build_h2h_lookups,
    predict_match,
    simulate_bracket,
    _predict_one_direction,
)

DATA_PATH   = "data/processed/final_training_data.csv"
CONFIG_PATH = "data/config/tournaments_config.csv"
MODEL_PATH  = "models/best_model.pkl"

# Human-readable names for the 24 model features (4 cat IDs + 20 cont)
FEATURE_NAMES = [
    "tier_id", "round_id", "player_a_id", "player_b_id",
] + CONT_COLS

st.set_page_config(
    page_title="BWF Match Predictor",
    page_icon="🏸",
    layout="wide",
)


# ------------------------------------------------------------------
# Cached resource loaders
# ------------------------------------------------------------------

@st.cache_resource
def load_resources():
    with open(MODEL_PATH, "rb") as f:
        model_payload = pickle.load(f)
    _, _, _, preprocessors = get_train_val_datasets(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower()
    return model_payload, preprocessors, df


@st.cache_resource
def get_shap_explainer():
    """Create a SHAP TreeExplainer for the underlying tree model."""
    model_payload, _, _ = load_resources()
    if model_payload["type"] == "single":
        tree_model = model_payload["model"]
    else:
        # For ensemble: prefer XGBoost, otherwise use the first model
        models = model_payload["models"]
        tree_model = models.get("xgb", next(iter(models.values())))
    return shap.TreeExplainer(tree_model)


@st.cache_data(show_spinner=False)
def get_2026_tournaments():
    cfg = pd.read_csv(CONFIG_PATH)
    cfg2026 = cfg[cfg["start_date"].str.startswith("2026")].copy()
    cfg2026 = cfg2026.sort_values("start_date")
    return cfg2026[["tournament_name", "tier", "start_date"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_simulation(tour_date: str, tier: int, n_sims: int, seed: int = 42):
    """Cached simulation — re-runs only when parameters change."""
    model_payload, preprocessors, df = load_resources()
    scaler       = preprocessors["scaler"]
    player_to_id = preprocessors["player_to_id"]
    tier_to_id   = preprocessors["tier_to_id"]
    round_to_id  = preprocessors["round_to_id"]

    r32_matchups, player_stats = build_time_zero_state(df, tour_date, tier)
    h2h_rate_fn, h2h_last_fn  = build_h2h_lookups(df, tour_date)

    if r32_matchups.empty:
        return None, None

    bracket_rows = []
    for _, row in r32_matchups.iterrows():
        p = predict_match(
            row["player_a"], row["player_b"], "first round",
            player_stats, h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
        )
        bracket_rows.append({
            "Player A":  row["player_a"],
            "Player B":  row["player_b"],
            "P(A wins)": round(p, 3),
        })
    bracket_df = pd.DataFrame(bracket_rows)

    rng = np.random.default_rng(seed)
    win_counts = {}
    for _ in range(n_sims):
        champion = simulate_bracket(
            r32_matchups, player_stats,
            h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id,
            model_payload, rng, tier,
        )
        win_counts[champion] = win_counts.get(champion, 0) + 1

    leaderboard = (
        pd.DataFrame(win_counts.items(), columns=["Player", "Wins"])
        .sort_values("Wins", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard["Win %"] = (leaderboard["Wins"] / n_sims * 100).round(2)
    return bracket_df, leaderboard


def build_shap_input(pa, pb, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
                     scaler, player_to_id, tier_to_id, round_to_id, tier):
    """
    Build the 24-element feature vector for one direction (pa → slot_a)
    as a named DataFrame so SHAP waterfall labels are human-readable.
    """
    UNK      = 0
    tier_id  = tier_to_id.get(tier, 0)
    round_id = round_to_id.get(round_name, 0)
    pa_id    = player_to_id.get(pa, UNK)
    pb_id    = player_to_id.get(pb, UNK)
    sa, sb   = player_stats[pa], player_stats[pb]

    cont_raw = np.array([[
        0.0,
        h2h_rate_fn(pa, pb),
        float(sa["is_home"]),
        float(sa["matches_14d"]),
        float(sa["days_since"]),
        float(sa["recent_win_rate"]),
        float(sb["is_home"]),
        float(sb["matches_14d"]),
        float(sb["days_since"]),
        float(sb["recent_win_rate"]),
        float(sa["elo"]),
        float(sb["elo"]),
        float(sa["elo"] - sb["elo"]),
        float(sa["ema_form"]),
        float(sb["ema_form"]),
        h2h_last_fn(pa, pb),
        float(sa["win_streak"]),
        float(sb["win_streak"]),
        float(sa["matches_7d"]),
        float(sb["matches_7d"]),
    ]], dtype=np.float32)

    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_id, round_id, pa_id, pb_id]], dtype=np.float64)
    X = np.hstack([cat, cont_scaled])
    return X


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------

st.title("🏸 BWF Men's Singles — Monte Carlo Bracket Simulator")

tournaments = get_2026_tournaments()

with st.sidebar:
    st.header("Settings")
    options  = tournaments["tournament_name"].tolist()
    selected = st.selectbox("Tournament", options, index=len(options) - 1)

    t_row     = tournaments[tournaments["tournament_name"] == selected].iloc[0]
    tour_date = t_row["start_date"]
    tier      = int(t_row["tier"])

    st.caption(f"Date: {tour_date}  |  Tier: {tier}")
    n_sims  = st.slider("Simulations", min_value=1_000, max_value=50_000,
                        value=10_000, step=1_000)
    run_btn = st.button("▶  Run Simulation", use_container_width=True, type="primary")

# Pre-load the tournament roster (fast — just a DataFrame filter)
model_payload, preprocessors, df = load_resources()
scaler       = preprocessors["scaler"]
player_to_id = preprocessors["player_to_id"]
tier_to_id   = preprocessors["tier_to_id"]
round_to_id  = preprocessors["round_to_id"]

r32_matchups, player_stats = build_time_zero_state(df, tour_date, tier)
h2h_rate_fn, h2h_last_fn  = build_h2h_lookups(df, tour_date)
roster = sorted(player_stats.keys())

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_sim, tab_shap = st.tabs(["🏆 Monte Carlo Bracket", "🔍 Matchup Explainer"])

# ── Tab 1: Monte Carlo Bracket ─────────────────────────────────────
with tab_sim:
    if not run_btn:
        st.info("Select a tournament in the sidebar and click **▶ Run Simulation** to begin.")
    else:
        if r32_matchups.empty:
            st.error(f"No first-round data found for **{selected}** ({tour_date}). "
                     "The dataset may not cover this tournament.")
        else:
            with st.spinner(f"Running {n_sims:,} simulations for {selected}..."):
                bracket_df, leaderboard = run_simulation(tour_date, tier, n_sims)

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader(f"First Round Bracket ({len(bracket_df)} matchups)")
                styled = (
                    bracket_df.style
                    .format({"P(A wins)": "{:.3f}"})
                    .background_gradient(subset=["P(A wins)"], cmap="RdYlGn",
                                         vmin=0.3, vmax=0.7)
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

            with col_right:
                st.subheader(f"Championship Probability ({n_sims:,} sims)")
                top_n      = min(16, len(leaderboard))
                chart_data = leaderboard.head(top_n).set_index("Player")["Win %"]
                st.bar_chart(chart_data, horizontal=True)
                st.dataframe(
                    leaderboard[["Player", "Win %"]]
                    .style.format({"Win %": "{:.2f}%"})
                    .background_gradient(subset=["Win %"], cmap="Blues"),
                    use_container_width=True,
                    hide_index=True,
                )
                model_name = model_payload.get("name", model_payload.get("type", "?"))
                st.caption(f"Model: **{model_name}**  |  Val AUC: 0.7754")

# ── Tab 2: Matchup Explainer ────────────────────────────────────────
with tab_shap:
    st.subheader("SHAP Matchup Explainer")
    st.markdown(
        "Select any two players from the tournament roster to see exactly which "
        "features push the model's prediction above or below its baseline."
    )

    if not roster:
        st.warning(f"No bracket data found for **{selected}**. "
                   "Select a different tournament.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            default_a = 0
            player_a  = st.selectbox("Player A", roster, index=default_a, key="shap_pa")
        with col_b:
            default_b = min(1, len(roster) - 1)
            player_b  = st.selectbox("Player B", roster, index=default_b, key="shap_pb")

        analyze_btn = st.button("🔍 Analyze Matchup", type="primary")

        if player_a == player_b:
            st.warning("Please select two **different** players.")
        elif analyze_btn:
            # ── Build feature vector ──────────────────────────────
            X = build_shap_input(
                player_a, player_b, "first round",
                player_stats, h2h_rate_fn, h2h_last_fn,
                scaler, player_to_id, tier_to_id, round_to_id, tier,
            )

            # ── Win probability (order-invariant) ─────────────────
            p_win = predict_match(
                player_a, player_b, "first round",
                player_stats, h2h_rate_fn, h2h_last_fn,
                scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
            )

            st.metric(
                label=f"P({player_a} beats {player_b})",
                value=f"{p_win*100:.1f}%",
                delta=f"{(p_win - 0.5)*100:+.1f}pp vs 50/50",
            )

            # ── SHAP analysis ─────────────────────────────────────
            with st.spinner("Computing SHAP values..."):
                explainer  = get_shap_explainer()
                shap_vals  = explainer(X)
                # Attach human-readable feature names to the Explanation object
                shap_vals.feature_names = FEATURE_NAMES

            st.markdown(
                f"**Waterfall plot** — how each feature shifts the model's output "
                f"(log-odds) from the base value to the final prediction for "
                f"**{player_a}** in the Player A slot."
            )

            # ── Render waterfall via matplotlib ──────────────────
            # shap.plots.waterfall draws onto the current figure.
            # We capture it with plt.gcf() AFTER the call, then hand it to
            # st.pyplot() and close it to prevent memory leaks.
            shap.plots.waterfall(shap_vals[0], max_display=20, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ── Raw feature value table ───────────────────────────
            with st.expander("Raw feature values"):
                feat_df = pd.DataFrame({
                    "Feature":      FEATURE_NAMES,
                    "Scaled value": X[0].tolist(),
                    "SHAP value":   shap_vals.values[0].tolist(),
                })
                feat_df["SHAP value"] = feat_df["SHAP value"].round(4)
                feat_df["Scaled value"] = feat_df["Scaled value"].round(4)
                feat_df = feat_df.sort_values("SHAP value", key=abs, ascending=False)
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
