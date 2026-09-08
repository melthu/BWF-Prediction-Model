"""
Shared fixtures.

These tests run against the real scraped data rather than a synthetic frame.
That is deliberate: the bugs this suite exists to catch (a mirrored column with
the wrong sign, a bracket whose rounds don't resolve, a name that folds to the
empty string) were all invisible on tidy made-up input and only showed up on
the actual corpus.
"""
import os
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH  = "data/raw/raw_matches.csv"
DATA_PATH = "data/processed/final_training_data.csv"
CFG_PATH  = "data/config/tournaments_config.csv"


def pytest_configure():
    """Run from the repo root, so the relative data paths resolve."""
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def raw():
    if not os.path.exists(RAW_PATH):
        pytest.skip(f"{RAW_PATH} missing")
    return pd.read_csv(RAW_PATH)


@pytest.fixture(scope="session")
def cfg():
    if not os.path.exists(CFG_PATH):
        pytest.skip(f"{CFG_PATH} missing")
    return pd.read_csv(CFG_PATH)


@pytest.fixture(scope="session")
def df():
    """The mirrored training frame, rebuilt if it is missing (it is not tracked)."""
    if not os.path.exists(DATA_PATH):
        if not os.path.exists("data/interim/engineered_matches.csv"):
            subprocess.run([sys.executable, "src/pipeline/feature_engineering.py"],
                           check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "src/pipeline/data_loader.py"],
                       check=True, stdout=subprocess.DEVNULL)
    frame = pd.read_csv(DATA_PATH)
    frame["start_date"] = pd.to_datetime(frame["start_date"])
    return frame


@pytest.fixture(scope="session")
def tournament(cfg, df):
    """
    A completed tournament with a full 32-player draw, chosen from the data
    rather than hardcoded so the suite keeps working as the corpus grows.
    """
    from src.serving.export_static import dedupe_day

    completed = df[df["is_pending"] == 0]
    for _, row in cfg.sort_values("start_date", ascending=False).iterrows():
        date = pd.Timestamp(row["start_date"])
        day = dedupe_day(completed[(completed["start_date"] == date)
                                   & (completed["tournament"] == row["tournament_name"])])
        if len(day) == 31 and (day["round"] == "first round").sum() == 16:
            return row, day
    pytest.skip("no completed 32-draw tournament found")


@pytest.fixture(scope="session")
def bye_draw(cfg, df):
    """
    A completed tournament whose opening round feeds only part of the next one.

    Super 100 and 300 draws run a preliminary round that some of the field
    skips, so a 16-match opener is followed by a 16-match round of 32 rather
    than an 8-match round of 16. The `tournament` fixture above screens these
    out - it wants exactly 31 matches - which is precisely why the bracket bug
    they carry survived: 30 tournaments in the corpus have this shape and the
    suite had never simulated one.
    """
    from src.serving.export_static import dedupe_day

    completed = df[df["is_pending"] == 0]
    for _, row in cfg.sort_values("start_date", ascending=False).iterrows():
        date = pd.Timestamp(row["start_date"])
        day = dedupe_day(completed[(completed["start_date"] == date)
                                   & (completed["tournament"] == row["tournament_name"])])
        if day.empty or "round" not in day.columns:
            continue
        n1 = (day["round"] == "first round").sum()
        n2 = (day["round"] == "second round").sum()
        if n1 and n2 > -(-n1 // 2) and (day["round"] == "final").sum() == 1:
            return row, day
    pytest.skip("no completed tournament with a preliminary round found")


@pytest.fixture(scope="session")
def fitted_bye(df, raw, bye_draw):
    """A point-in-time model and preprocessors for the `bye_draw` tournament."""
    return _fit_for(df, raw, bye_draw)


@pytest.fixture(scope="session")
def fitted(df, raw, tournament):
    """A point-in-time model and its paired preprocessors for `tournament`."""
    return _fit_for(df, raw, tournament)


def _fit_for(df, raw, pair):
    from src.modeling.pit_model import train_point_in_time
    from src.serving.export_static import load_nat_map
    from src.serving.simulate import build_h2h_lookups, build_time_zero_state

    cfg_row, day = pair
    date_key = pd.Timestamp(cfg_row["start_date"]).strftime("%Y-%m-%d")
    tier = int(cfg_row["tier"])

    pit = train_point_in_time(df, date_key)
    assert pit is not None, "expected enough history to fit a point-in-time model"
    payload, pre = pit

    same_day = df["start_date"] == pd.Timestamp(date_key)
    mine = same_day & (df["tournament"] == cfg_row["tournament_name"])
    r1, stats = build_time_zero_state(df[~same_day | mine], date_key, tier)
    h2h_rate, h2h_last = build_h2h_lookups(df, date_key)

    return {
        "cfg_row": cfg_row, "day": day, "date_key": date_key, "tier": tier,
        "payload": payload, "pre": pre, "r1": r1, "stats": stats,
        "h2h_rate": h2h_rate, "h2h_last": h2h_last,
        "nat_map": load_nat_map(raw),
    }
