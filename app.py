import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import pickle
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st
from streamlit_calendar import calendar as st_calendar

from src.dataset import get_train_val_datasets, CONT_COLS
from src.simulate_german_open import (
    ROUND_ORDER,
    build_time_zero_state,
    build_h2h_lookups,
    predict_match,
    simulate_bracket,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DATA_PATH   = "data/processed/final_training_data.csv"
CONFIG_PATH = "data/config/tournaments_config.csv"
MODEL_PATH  = "models/best_model.pkl"

FEATURE_NAMES = ["tier_id", "round_id", "player_a_id", "player_b_id"] + CONT_COLS

TIER_LABELS: dict[int, str] = {
    1500: "Finals",
    1000: "Super 1000",
    750:  "Super 750",
    500:  "Super 500",
    300:  "Super 300",
    100:  "Super 100",
}

PLAYER_FLAGS: dict[str, str] = {
    # Denmark
    "Viktor Axelsen":                    "🇩🇰",
    "Anders Antonsen":                   "🇩🇰",
    "Rasmus Gemke":                      "🇩🇰",
    "Jan Ø. Jørgensen":                  "🇩🇰",
    "Jan O. Jørgensen":                  "🇩🇰",
    "Peter Gade":                        "🇩🇰",
    # Norway
    "Hans-Kristian Solberg Vittersheim": "🇳🇴",
    # Indonesia
    "Jonatan Christie":                  "🇮🇩",
    "Anthony Sinisuka Ginting":          "🇮🇩",
    "Tommy Sugiarto":                    "🇮🇩",
    "Ihsan Maulana Mustofa":             "🇮🇩",
    "Taufik Hidayat":                    "🇮🇩",
    "Sony Dwi Kuncoro":                  "🇮🇩",
    # China
    "Lin Dan":                           "🇨🇳",
    "Chen Long":                         "🇨🇳",
    "Shi Yuqi":                          "🇨🇳",
    "Li Shifeng":                        "🇨🇳",
    "Lu Guang Zu":                       "🇨🇳",
    "Weng Hong Yang":                    "🇨🇳",
    "Chen Xiaoxin":                      "🇨🇳",
    # Japan
    "Kento Momota":                      "🇯🇵",
    "Kodai Naraoka":                     "🇯🇵",
    "Kanta Tsuneyama":                   "🇯🇵",
    # Malaysia
    "Lee Chong Wei":                     "🇲🇾",
    "Lee Zii Jia":                       "🇲🇾",
    "Daren Liew":                        "🇲🇾",
    # Thailand
    "Kunlavut Vitidsarn":                "🇹🇭",
    "Tanongsak Saensomboonsuk":          "🇹🇭",
    "Kantaphon Wangcharoen":             "🇹🇭",
    # Taiwan
    "Chou Tien-chen":                    "🇹🇼",
    "Wang Tzu-wei":                      "🇹🇼",
    # India
    "Prannoy H. S.":                     "🇮🇳",
    "Lakshya Sen":                       "🇮🇳",
    "Srikanth Kidambi":                  "🇮🇳",
    "Kidambi Srikanth":                  "🇮🇳",
    "Parupalli Kashyap":                 "🇮🇳",
    "Sai Praneeth B.":                   "🇮🇳",
    # South Korea
    "Son Wan Ho":                        "🇰🇷",
    "Lee Hyun Il":                       "🇰🇷",
    "Jeon Hyuk Jin":                     "🇰🇷",
    # France
    "Toma Junior Popov":                 "🇫🇷",
    "Christo Popov":                     "🇫🇷",
    "Brice Leverdez":                    "🇫🇷",
    # Singapore
    "Loh Kean Yew":                      "🇸🇬",
    # Hong Kong
    "Ng Ka Long Angus":                  "🇭🇰",
    # Netherlands
    "Mark Caljouw":                      "🇳🇱",
    # Canada
    "Brian Yang":                        "🇨🇦",
    # Ireland
    "Nhat Nguyen":                       "🇮🇪",
    # Germany
    "Marc Zwiebler":                     "🇩🇪",
    "Fabian Roth":                       "🇩🇪",
}

TOURNAMENT_HOSTS: dict[str, str] = {
    "China":                  "🇨🇳",
    "Japan":                  "🇯🇵",
    "South Korea":            "🇰🇷",
    "Korea":                  "🇰🇷",
    "Indonesia":              "🇮🇩",
    "Malaysia":               "🇲🇾",
    "Thailand":               "🇹🇭",
    "India":                  "🇮🇳",
    "Denmark":                "🇩🇰",
    "France":                 "🇫🇷",
    "Germany":                "🇩🇪",
    "England":                "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "United Kingdom":         "🇬🇧",
    "Australia":              "🇦🇺",
    "New Zealand":            "🇳🇿",
    "Canada":                 "🇨🇦",
    "United States":          "🇺🇸",
    "Singapore":              "🇸🇬",
    "Hong Kong":              "🇭🇰",
    "Taiwan":                 "🇹🇼",
    "Switzerland":            "🇨🇭",
    "Spain":                  "🇪🇸",
    "Netherlands":            "🇳🇱",
    "Finland":                "🇫🇮",
    "Philippines":            "🇵🇭",
    "Vietnam":                "🇻🇳",
    "Bangladesh":             "🇧🇩",
    "Sri Lanka":              "🇱🇰",
    "United Arab Emirates":   "🇦🇪",
    "Austria":                "🇦🇹",
    "Poland":                 "🇵🇱",
    "Sweden":                 "🇸🇪",
    "Ireland":                "🇮🇪",
    "Bahrain":                "🇧🇭",
    "Macau":                  "🇲🇴",
    "Norway":                 "🇳🇴",
}

TODAY = date.today()


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------

def format_name(name: str) -> str:
    return f"{PLAYER_FLAGS.get(name, '🏸')} {name}"


def format_tier(tier: int) -> str:
    return TIER_LABELS.get(tier, f"Tier {tier}")


def get_actual_winner(df: pd.DataFrame, tour_date: str):
    rows = df[
        (df["start_date"] == pd.Timestamp(tour_date)) &
        (df["round"] == "final")
    ]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return r["player_a"] if r["player_a_won"] == 1 else r["player_b"]


def build_calendar_events(
    all_tours: pd.DataFrame,
    selected_key: str,
    today: date,
) -> list:
    """
    Build FullCalendar event dicts for all BWF tournaments.
    Each tournament spans 6 days (end is exclusive in FullCalendar).
    Colors: selected = navy, past = muted gray, upcoming = vibrant blue.
    """
    events = []
    for _, r in all_tours.iterrows():
        tour_dt  = r["start_date"]
        tour_key = tour_dt.strftime("%Y-%m-%d")
        end_key  = (tour_dt + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        flag     = TOURNAMENT_HOSTS.get(str(r["host_country"]), "🌐")
        is_past  = tour_dt.date() < today
        is_sel   = tour_key == selected_key

        if is_sel:
            bg, border = "#1a3a5c", "#1a3a5c"
        elif is_past:
            bg, border = "#9e9e9e", "#757575"
        else:
            bg, border = "#1e88e5", "#1565c0"

        events.append({
            "id":              tour_key,
            "title":           f"{flag} {r['tournament_name']}",
            "start":           tour_key,
            "end":             end_key,
            "backgroundColor": bg,
            "borderColor":     border,
        })
    return events


def build_shap_input(pa, pb, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
                     scaler, player_to_id, tier_to_id, round_to_id, tier):
    """Build (1, 34) feature array: 4 cat IDs + 30 scaled cont features."""
    sa, sb   = player_stats[pa], player_stats[pb]
    tier_id  = tier_to_id.get(tier, 0)
    round_id = round_to_id.get(round_name, 0)
    pa_id    = player_to_id.get(pa, 0)
    pb_id    = player_to_id.get(pb, 0)

    cont_raw = np.array([[
        0.0, h2h_rate_fn(pa, pb),
        float(sa["is_home"]),    float(sa["matches_14d"]),  float(sa["days_since"]),   float(sa["recent_win_rate"]),
        float(sb["is_home"]),    float(sb["matches_14d"]),  float(sb["days_since"]),   float(sb["recent_win_rate"]),
        float(sa["elo"]),        float(sb["elo"]),           float(sa["elo"] - sb["elo"]),
        float(sa["ema_form"]),   float(sb["ema_form"]),     h2h_last_fn(pa, pb),
        float(sa["win_streak"]), float(sb["win_streak"]),
        float(sa["matches_7d"]), float(sb["matches_7d"]),
        float(sa["avg_point_diff"]),   float(sb["avg_point_diff"]),
        float(sa["avg_games_pm"]),     float(sb["avg_games_pm"]),
        float(sa["rubber_game_rate"]), float(sb["rubber_game_rate"]),
        float(sa["avg_margin"]),       float(sb["avg_margin"]),
        float(sa["seed"]),             float(sb["seed"]),
    ]], dtype=np.float32)

    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_id, round_id, pa_id, pb_id]], dtype=np.float64)
    return np.hstack([cat, cont_scaled])


def compute_likely_bracket(
    r32_matchups, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, tier,
) -> dict[str, list[str]]:
    """Greedily advance the higher-probability winner through each round."""
    current = [(row["player_a"], row["player_b"]) for _, row in r32_matchups.iterrows()]
    round_winners: dict[str, list[str]] = {}
    for rnd in ROUND_ORDER:
        if not current:
            break
        winners = [
            pa if predict_match(pa, pb, rnd, player_stats, h2h_rate_fn, h2h_last_fn,
                                scaler, player_to_id, tier_to_id, round_to_id,
                                model_payload, tier) >= 0.5 else pb
            for pa, pb in current
        ]
        round_winners[rnd] = winners
        current = list(zip(winners[::2], winners[1::2]))
    return round_winners


def render_bracket_figure(round_winners: dict[str, list[str]]) -> go.Figure:
    """Plotly table for the most-likely bracket path (post-simulation)."""
    LABELS = {
        "first round":    "R1",
        "second round":   "R2",
        "quarter-finals": "QF",
        "semi-finals":    "SF",
        "final":          "🏆 Final",
    }
    rounds_present = [r for r in ROUND_ORDER if r in round_winners]
    headers    = [LABELS.get(r, r.title()) for r in rounds_present]
    max_rows   = max(len(round_winners[r]) for r in rounds_present)

    col_data, col_colors = [], []
    for rnd in rounds_present:
        players  = [format_name(p) for p in round_winners[rnd]]
        n        = len(players)
        is_final = rnd == rounds_present[-1]
        colors   = [
            ("#fff3cd" if is_final else ("#dceefb" if j % 2 == 0 else "#ffffff"))
            if j < n else "#f5f5f5"
            for j in range(max_rows)
        ]
        col_data.append(players + [""] * (max_rows - n))
        col_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#1a3a5c", font=dict(color="white", size=13),
            align="center", height=34,
        ),
        cells=dict(
            values=col_data, fill_color=col_colors,
            align="center", font=dict(size=11), height=26,
        ),
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=4, b=0),
                      height=max(180, 30 * max_rows + 70))
    return fig



def build_radar_chart(pa: str, pb: str, player_stats: dict, h2h_rate_fn) -> go.Figure:
    extractors = {
        "Elo":       lambda s: s["elo"],
        "Form":      lambda s: s["ema_form"],
        "Streak":    lambda s: s["win_streak"],
        "Pt Diff":   lambda s: s["avg_point_diff"],
        "Freshness": lambda s: -s["days_since"],
    }

    def _norm(key, player):
        vals = [extractors[key](s) for s in player_stats.values()]
        lo, hi = min(vals), max(vals)
        v = extractors[key](player_stats[player])
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    dims = list(extractors.keys())
    fig  = go.Figure()
    for player, color, fill in [
        (pa, "#1f77b4", "rgba(31,119,180,0.20)"),
        (pb, "#d62728", "rgba(214,39,40,0.20)"),
    ]:
        r_vals = [_norm(d, player) for d in dims] + [_norm(dims[0], player)]
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=dims + [dims[0]], fill="toself",
            name=format_name(player),
            line=dict(color=color), fillcolor=fill,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, legend=dict(x=0.8, y=1.1),
        margin=dict(l=30, r=30, t=30, b=30), height=380,
    )
    return fig


def build_form_chart(player: str, tour_date_str: str, df: pd.DataFrame,
                     scaler, player_to_id, tier_to_id, round_to_id,
                     model_payload, tier: int):
    cutoff = pd.Timestamp(tour_date_str)
    mask   = (
        ((df["player_a"] == player) | (df["player_b"] == player)) &
        (df["start_date"] < cutoff)
    )
    hist = df[mask].sort_values("start_date").tail(5).reset_index(drop=True)
    if hist.empty:
        return None

    records = []
    for _, mrow in hist.iterrows():
        m_date   = mrow["start_date"]
        is_a     = mrow["player_a"] == player
        opp      = mrow["player_b"] if is_a else mrow["player_a"]
        won      = bool(mrow["player_a_won"] == 1) if is_a else bool(mrow["player_a_won"] == 0)
        side, os = ("a", "b") if is_a else ("b", "a")

        def _f(col, default):
            return mrow.get(col, default)

        mini = {
            player: {
                "is_home":          int(_f(f"player_{side}_is_home", 0)),
                "matches_14d":      int(_f(f"player_{side}_matches_last_14_days", 0)),
                "days_since":       float(_f(f"player_{side}_days_since_last_match", 100)),
                "recent_win_rate":  float(_f(f"player_{side}_recent_win_rate", 0.5)),
                "elo":              float(_f(f"player_{side}_elo", 1500)),
                "ema_form":         float(_f(f"player_{side}_ema_form", 0.5)),
                "win_streak":       int(_f(f"player_{side}_win_streak", 0)),
                "matches_7d":       int(_f(f"player_{side}_matches_last_7_days", 0)),
                "avg_point_diff":   float(_f(f"player_{side}_avg_point_diff", 0.0)),
                "avg_games_pm":     float(_f(f"player_{side}_avg_games_per_match", 2.0)),
                "rubber_game_rate": float(_f(f"player_{side}_rubber_game_rate", 0.0)),
                "avg_margin":       float(_f(f"player_{side}_avg_victory_margin", 0.0)),
                "seed":             float(_f(f"player_{side}_seed", 0.0)),
            },
            opp: {
                "is_home":          int(_f(f"player_{os}_is_home", 0)),
                "matches_14d":      int(_f(f"player_{os}_matches_last_14_days", 0)),
                "days_since":       float(_f(f"player_{os}_days_since_last_match", 100)),
                "recent_win_rate":  float(_f(f"player_{os}_recent_win_rate", 0.5)),
                "elo":              float(_f(f"player_{os}_elo", 1500)),
                "ema_form":         float(_f(f"player_{os}_ema_form", 0.5)),
                "win_streak":       int(_f(f"player_{os}_win_streak", 0)),
                "matches_7d":       int(_f(f"player_{os}_matches_last_7_days", 0)),
                "avg_point_diff":   float(_f(f"player_{os}_avg_point_diff", 0.0)),
                "avg_games_pm":     float(_f(f"player_{os}_avg_games_per_match", 2.0)),
                "rubber_game_rate": float(_f(f"player_{os}_rubber_game_rate", 0.0)),
                "avg_margin":       float(_f(f"player_{os}_avg_victory_margin", 0.0)),
                "seed":             float(_f(f"player_{os}_seed", 0.0)),
            },
        }
        m_tier  = int(_f("tier", tier))
        m_round = str(_f("round", "first round")).lower()
        hb = df[df["start_date"] < m_date]
        h2h_r, h2h_l = build_h2h_lookups(
            hb.assign(start_date=hb["start_date"]),
            m_date.strftime("%Y-%m-%d"),
        )
        p = predict_match(player, opp, m_round, mini, h2h_r, h2h_l,
                          scaler, player_to_id, tier_to_id, round_to_id, model_payload, m_tier)
        records.append({
            "Date":     m_date.strftime("%Y-%m-%d"),
            "Opponent": format_name(opp),
            "Win Prob": round(p, 3),
            "Result":   "W" if won else "L",
        })
    return pd.DataFrame(records)


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
    df["round"]      = df["round"].str.lower()
    return model_payload, preprocessors, df


@st.cache_resource
def get_shap_explainer():
    model_payload, _, _ = load_resources()

    def _pick_tree(payload):
        if payload["type"] == "single":
            return payload["model"]
        for name in ("xgb", "lgbm", "catboost"):
            if name in payload.get("models", {}):
                return payload["models"][name]
        return None

    m = _pick_tree(model_payload)
    return shap.TreeExplainer(m) if m and not hasattr(m, "parameters") else None


@st.cache_data(show_spinner=False)
def get_all_tournaments() -> pd.DataFrame:
    cfg = pd.read_csv(CONFIG_PATH)
    cfg["start_date"] = pd.to_datetime(cfg["start_date"], errors="coerce")
    cfg = cfg.dropna(subset=["start_date"]).sort_values("start_date", ascending=False)
    return cfg[["tournament_name", "tier", "start_date", "host_country"]].reset_index(drop=True)


# ------------------------------------------------------------------
# App bootstrap
# ------------------------------------------------------------------

st.set_page_config(page_title="BWF Live Terminal", page_icon="🏸", layout="wide")

all_tours = get_all_tournaments()

# Session state initialisation
_default_row = all_tours.iloc[0]
_default_key = _default_row["start_date"].strftime("%Y-%m-%d")

if "sim_results"       not in st.session_state:
    st.session_state["sim_results"]       = {}
if "shap_analyzed"     not in st.session_state:
    st.session_state["shap_analyzed"]     = None
if "selected_tour_key" not in st.session_state:
    st.session_state["selected_tour_key"] = _default_key
if "cal_initial_date"  not in st.session_state:
    st.session_state["cal_initial_date"]  = _default_row["start_date"].strftime("%Y-%m-01")

st.title("🏸 BWF Men's Singles — Live Point-in-Time Terminal")

# Derive active tournament from session state (needed in sidebar and main tabs)
tour_date = st.session_state["selected_tour_key"]
t_match   = all_tours[all_tours["start_date"] == pd.Timestamp(tour_date)]
if t_match.empty:
    t_match = all_tours.iloc[0:1]
t_row_sel = t_match.iloc[0]
selected  = t_row_sel["tournament_name"]
tier      = int(t_row_sel["tier"])
host_flag = TOURNAMENT_HOSTS.get(str(t_row_sel["host_country"]), "🌐")

# ------------------------------------------------------------------
# Sidebar — calendar + tournament picker
# ------------------------------------------------------------------

with st.sidebar:
    cal_events = build_calendar_events(
        all_tours, st.session_state["selected_tour_key"], TODAY
    )

    cal_options = {
        "initialView":   "dayGridMonth",
        "initialDate":   st.session_state["cal_initial_date"],
        "headerToolbar": {
            "left":   "prev,next today",
            "center": "title",
            "right":  "",
        },
        "height":       580,
        "navLinks":     False,
        "editable":     False,
        "selectable":   False,
        "dayMaxEvents": 2,
    }

    cal_state = st_calendar(
        events=cal_events,
        options=cal_options,
        callbacks=["eventClick"],
        custom_css="""
            .fc-event { cursor: pointer; font-size: 10px; }
            .fc-toolbar-title { font-size: 1rem !important; }
            .fc-button { font-size: 0.72rem !important; padding: 2px 6px !important; }
            .fc-daygrid-event-dot { display: none; }
        """,
        key=f"bwf_cal_{st.session_state['cal_initial_date']}",
    )

    # Handle event click — update selected tournament and remount calendar at that month
    if cal_state and cal_state.get("eventClick"):
        clicked_id = cal_state["eventClick"]["event"]["id"]
        if clicked_id != st.session_state["selected_tour_key"]:
            clicked_dt = pd.Timestamp(clicked_id)
            st.session_state["cal_initial_date"]  = clicked_dt.strftime("%Y-%m-01")
            st.session_state["selected_tour_key"] = clicked_id
            st.rerun()

    # Currently selected tournament label
    st.markdown(
        f"<div style='margin:8px 0 2px;font-size:0.82rem;color:#888'>Selected</div>"
        f"<div style='font-weight:700'>{host_flag} {selected}</div>"
        f"<div style='font-size:0.8rem;color:#666'>{format_tier(tier)}  ·  {tour_date}</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem;color:#888'>"
        "🔵 Upcoming &nbsp; ⚫ Past &nbsp; 🔷 Selected — click any block to select"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    n_sims  = st.slider("Monte Carlo Simulations", 1_000, 50_000, 10_000, 1_000)
    run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")

# ------------------------------------------------------------------
# Load data & build tournament state
# ------------------------------------------------------------------

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

tab_engine, tab_matchup = st.tabs(["🔬 Live Engine", "⚡ Matchup Analyzer"])

# ── Tab 1: Live Engine ─────────────────────────────────────────────

sim_key = f"{tour_date}|{tier}|{n_sims}"

with tab_engine:
    if r32_matchups.empty:
        st.error(
            f"No first-round data for **{selected}** ({tour_date}).  \n"
            "Try a different month, or re-run `make features` + `make train`."
        )

    elif run_btn:
        with st.status("🔬 Point-in-Time Engine", expanded=True) as sim_status:
            st.write(f"📅 **{selected}** — data cut-off at `{tour_date}`")
            st.write(
                f"👥 Roster: **{len(player_stats)}** players  ·  "
                f"**{len(r32_matchups)}** first-round matchups"
            )
            actual_winner = get_actual_winner(df, tour_date)
            if actual_winner:
                st.write(f"📋 Actual winner on record: **{format_name(actual_winner)}**")
            st.write(f"🎲 Launching **{n_sims:,}** Monte Carlo iterations…")
            sim_status.update(label=f"Simulating {selected}…",
                              state="running", expanded=False)

            # ── Live ticker ─────────────────────────────────────
            ticker_ph  = st.empty()
            win_counts: dict[str, int] = {}
            rng = np.random.default_rng(42)
            t0  = time.time()

            for i in range(n_sims):
                champion = simulate_bracket(
                    r32_matchups, player_stats,
                    h2h_rate_fn, h2h_last_fn,
                    scaler, player_to_id, tier_to_id, round_to_id,
                    model_payload, rng, tier,
                )
                win_counts[champion] = win_counts.get(champion, 0) + 1

                if (i + 1) % 500 == 0 or (i + 1) == n_sims:
                    n_done  = i + 1
                    lb_live = (
                        pd.DataFrame(win_counts.items(), columns=["Player", "Wins"])
                        .sort_values("Wins", ascending=False).head(10)
                        .reset_index(drop=True)
                    )
                    lb_live["Win %"]  = (lb_live["Wins"] / n_done * 100).round(1)
                    lb_live["Player"] = lb_live["Player"].apply(format_name)
                    ticker_ph.dataframe(
                        lb_live[["Player", "Win %"]].style.format({"Win %": "{:.1f}%"}),
                        use_container_width=True, hide_index=True,
                    )

            elapsed = time.time() - t0

            # ── Build final results ──────────────────────────────
            leaderboard = (
                pd.DataFrame(win_counts.items(), columns=["Player", "Wins"])
                .sort_values("Wins", ascending=False).reset_index(drop=True)
            )
            leaderboard["Win %"] = (leaderboard["Wins"] / n_sims * 100).round(2)
            if actual_winner:
                leaderboard["Actual Result"] = leaderboard["Player"].apply(
                    lambda p: "🥇 Winner" if p == actual_winner else ""
                )

            bracket_rows = []
            for _, row in r32_matchups.iterrows():
                p = predict_match(
                    row["player_a"], row["player_b"], "first round",
                    player_stats, h2h_rate_fn, h2h_last_fn,
                    scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
                )
                bracket_rows.append({
                    "Player A": row["player_a"], "Player B": row["player_b"],
                    "P(A wins)": round(p, 3),
                })
            bracket_df = pd.DataFrame(bracket_rows)

            round_winners = compute_likely_bracket(
                r32_matchups, player_stats, h2h_rate_fn, h2h_last_fn,
                scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
            )

            st.session_state["sim_results"][sim_key] = {
                "bracket_df":    bracket_df,
                "leaderboard":   leaderboard,
                "round_winners": round_winners,
                "actual_winner": actual_winner,
                "elapsed":       elapsed,
            }
            sim_status.update(
                label=f"✅ {n_sims:,} sims complete — {elapsed:.1f}s",
                state="complete", expanded=False,
            )
        st.rerun()

    elif sim_key in st.session_state["sim_results"]:
        # ── Show stored results ──────────────────────────────────
        res           = st.session_state["sim_results"][sim_key]
        bracket_df    = res["bracket_df"]
        leaderboard   = res["leaderboard"]
        round_winners = res["round_winners"]
        actual_winner = res["actual_winner"]
        elapsed       = res["elapsed"]

        st.success(
            f"✅ **{selected}** — {n_sims:,} sims  ·  {elapsed:.1f}s  ·  "
            f"Model: **{model_payload.get('name', '?')}**  ·  Val AUC: **0.7872**"
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(f"First Round ({len(bracket_df)} matchups)")
            disp_br = bracket_df.copy()
            disp_br["Player A"] = disp_br["Player A"].apply(format_name)
            disp_br["Player B"] = disp_br["Player B"].apply(format_name)
            st.dataframe(
                disp_br.style
                .format({"P(A wins)": "{:.3f}"})
                .background_gradient(subset=["P(A wins)"], cmap="RdYlGn", vmin=0.3, vmax=0.7),
                use_container_width=True, hide_index=True,
            )

        with col_right:
            st.subheader(f"Championship Probability ({n_sims:,} sims)")
            disp_cols = ["Player", "Win %"] + (
                ["Actual Result"] if "Actual Result" in leaderboard.columns else []
            )
            disp_lb = leaderboard[disp_cols].copy()
            disp_lb["Player"] = disp_lb["Player"].apply(format_name)

            chart_lb = leaderboard.head(min(12, len(leaderboard))).set_index("Player")["Win %"]
            chart_lb.index = chart_lb.index.map(format_name)
            st.bar_chart(chart_lb, horizontal=True)

            st.dataframe(
                disp_lb.style
                .format({"Win %": "{:.2f}%"})
                .background_gradient(subset=["Win %"], cmap="Blues"),
                use_container_width=True, hide_index=True,
            )

        if round_winners:
            st.subheader("🌳 Most Likely Bracket Path")
            st.plotly_chart(render_bracket_figure(round_winners), use_container_width=True)

    else:
        # ── Pre-run: show bracket matchups with instant probabilities ──
        is_upcoming = pd.Timestamp(tour_date).date() >= TODAY
        tag = "🔮 Upcoming" if is_upcoming else "📜 Historical"
        st.subheader(f"{host_flag} {selected}  ·  {format_tier(tier)}  ·  {tag}")
        st.caption(
            f"{len(roster)} players in bracket  ·  "
            "Click **▶ Run Simulation** in the sidebar to launch Monte Carlo analysis."
        )



# ── Tab 2: Matchup Analyzer ────────────────────────────────────────

with tab_matchup:
    st.subheader("⚡ Matchup Analyzer")

    if not roster:
        st.warning(f"No bracket data for **{selected}**. Pick another tournament.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("Player A", roster, index=0,
                                format_func=format_name, key="shap_pa")
    with col_b:
        player_b = st.selectbox("Player B", roster, index=min(1, len(roster) - 1),
                                format_func=format_name, key="shap_pb")

    analyze_btn = st.button("🔍 Analyze Matchup", type="primary")
    shap_key    = f"{player_a}|{player_b}|{tour_date}"

    if analyze_btn:
        st.session_state["shap_analyzed"] = shap_key

    if player_a == player_b:
        st.warning("Select two **different** players.")
    elif st.session_state.get("shap_analyzed") == shap_key:
        sa = player_stats[player_a]
        sb = player_stats[player_b]

        p_win = predict_match(
            player_a, player_b, "first round",
            player_stats, h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
        )
        st.metric(
            label=f"P({format_name(player_a)} beats {format_name(player_b)})",
            value=f"{p_win * 100:.1f}%",
            delta=f"{(p_win - 0.5) * 100:+.1f}pp vs 50/50",
        )

        # ── Tale of the Tape ─────────────────────────────────────
        st.subheader("📋 Tale of the Tape")
        tape_df = pd.DataFrame({
            "Stat": [
                "Elo Rating", "EMA Form", "Win Streak",
                "Days Since Last Match", "Matches (Last 14d)",
                "H2H Win Rate", "Avg Point Diff", "Seed",
            ],
            format_name(player_a): [
                f"{sa['elo']:.0f}",    f"{sa['ema_form']:.3f}",  f"{sa['win_streak']}",
                f"{sa['days_since']:.0f}",  f"{sa['matches_14d']}",
                f"{h2h_rate_fn(player_a, player_b):.3f}",
                f"{sa['avg_point_diff']:+.2f}", f"{int(sa['seed'])}",
            ],
            format_name(player_b): [
                f"{sb['elo']:.0f}",    f"{sb['ema_form']:.3f}",  f"{sb['win_streak']}",
                f"{sb['days_since']:.0f}",  f"{sb['matches_14d']}",
                f"{h2h_rate_fn(player_b, player_a):.3f}",
                f"{sb['avg_point_diff']:+.2f}", f"{int(sb['seed'])}",
            ],
        })
        st.dataframe(tape_df, use_container_width=True, hide_index=True)

        # ── Radar chart ──────────────────────────────────────────
        st.subheader("🕸️ Stat Radar")
        st.plotly_chart(
            build_radar_chart(player_a, player_b, player_stats, h2h_rate_fn),
            use_container_width=True,
        )

        # ── SHAP waterfall ───────────────────────────────────────
        st.subheader("🔍 SHAP Feature Attribution")
        explainer = get_shap_explainer()
        if explainer is None:
            st.info("SHAP unavailable — no tree model loaded.")
        else:
            X = build_shap_input(
                player_a, player_b, "first round",
                player_stats, h2h_rate_fn, h2h_last_fn,
                scaler, player_to_id, tier_to_id, round_to_id, tier,
            )
            with st.spinner("Computing SHAP values…"):
                _m = (model_payload["model"] if model_payload["type"] == "single"
                      else next(iter(model_payload["models"].values())))
                _n        = getattr(_m, "n_features_in_", X.shape[1])
                X_shap    = X[:, :_n]
                feat_names = FEATURE_NAMES[:_n]
                shap_vals  = explainer(X_shap)
                shap_vals.feature_names = feat_names

            st.caption(
                f"How each feature shifts the log-odds for "
                f"**{format_name(player_a)}** in the Player A slot."
            )
            shap.plots.waterfall(shap_vals[0], max_display=20, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Raw feature values"):
                feat_df = pd.DataFrame({
                    "Feature":      feat_names,
                    "Scaled value": X_shap[0].round(4).tolist(),
                    "SHAP value":   shap_vals.values[0].round(4).tolist(),
                }).sort_values("SHAP value", key=abs, ascending=False)
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # ── Player Form ──────────────────────────────────────────
        st.divider()
        st.subheader("📈 Recent Form — Last 5 Matches")
        st.caption("Win probability estimated strictly from pre-match data (no leakage).")
        form_col_a, form_col_b = st.columns(2)
        for col_w, player_name in [(form_col_a, player_a), (form_col_b, player_b)]:
            with col_w:
                st.markdown(f"**{format_name(player_name)}**")
                fdf = build_form_chart(
                    player_name, tour_date, df,
                    scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
                )
                if fdf is None or fdf.empty:
                    st.caption("No match history before this tournament.")
                else:
                    fig2, ax = plt.subplots(figsize=(5, 3))
                    colors = ["green" if r == "W" else "red" for r in fdf["Result"]]
                    ax.plot(range(len(fdf)), fdf["Win Prob"].values,
                            color="steelblue", linewidth=1.5, zorder=1)
                    ax.scatter(range(len(fdf)), fdf["Win Prob"].values,
                               c=colors, s=60, zorder=2)
                    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks(range(len(fdf)))
                    ax.set_xticklabels(fdf["Date"].tolist(), rotation=30,
                                       ha="right", fontsize=7)
                    ax.set_ylabel("Win Probability")
                    ax.set_title(f"Last {len(fdf)} matches")
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                    st.dataframe(
                        fdf.style.map(
                            lambda v: "color: green" if v == "W" else "color: red",
                            subset=["Result"],
                        ),
                        use_container_width=True, hide_index=True,
                    )
    else:
        st.info("Select two players above and click **🔍 Analyze Matchup**.")
