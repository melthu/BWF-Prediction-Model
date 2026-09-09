import argparse
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root
from src.pipeline.player_names import canonicalise
from src.pipeline.scraper_wiki_single import scrape_wiki_single

CONFIG_PATH = "data/config/tournaments_config.csv"
OUTPUT_PATH = "data/raw/raw_matches.csv"

# Incremental mode rescrapes any tournament that started within this many
# days in the past (results often land on Wikipedia days after the event)
# or starts within LOOKAHEAD_DAYS (draws are published shortly before).
RESCRAPE_WINDOW_DAYS = 21
LOOKAHEAD_DAYS       = 7


def _select_incremental(config: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Pick the subset of tournaments worth (re)scraping:
    missing from the CSV, containing pending matches, or recent/imminent."""
    today  = date.today()
    lo     = pd.Timestamp(today - timedelta(days=RESCRAPE_WINDOW_DAYS))
    hi     = pd.Timestamp(today + timedelta(days=LOOKAHEAD_DAYS))
    dates  = pd.to_datetime(config["start_date"])

    have = set(existing["tournament"].unique())
    # Pending rows trigger a rescrape only for recent events. Old tournaments
    # can carry permanent pending rows (group-stage matches where Wikipedia
    # never marks a winner) - rescraping those weekly would be pointless.
    if "is_pending" in existing.columns:
        recent_cutoff = (today - timedelta(days=60)).isoformat()
        pending = set(existing.loc[
            (existing["is_pending"] == 1) &
            (existing["start_date"].astype(str) >= recent_cutoff),
            "tournament",
        ].unique())
    else:
        pending = set()

    mask = (
        ~config["tournament_name"].isin(have)
        | config["tournament_name"].isin(pending)
        | ((dates >= lo) & (dates <= hi))
    )
    # Never try to scrape tournaments that haven't reached the lookahead window
    mask &= dates <= hi
    return config[mask]


def run_orchestrator(
    config_path: str = CONFIG_PATH,
    output_path: str = OUTPUT_PATH,
    incremental: bool = False,
) -> pd.DataFrame:
    config = pd.read_csv(config_path)

    existing = None
    if incremental and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if "is_pending" not in existing.columns:
            existing["is_pending"] = 0
        todo = _select_incremental(config, existing)
        print(f"Incremental mode: {len(todo)} of {len(config)} tournaments selected "
              f"(missing, pending, or within ±{RESCRAPE_WINDOW_DAYS}/{LOOKAHEAD_DAYS} days).\n")
    else:
        todo = config
        print(f"Full scrape: {len(config)} tournaments.\n")

    all_frames = []
    scraped_names = []

    for _, row in todo.iterrows():
        print(f"[{row['start_date']}] Scraping: {row['tournament_name']} (Tier {row['tier']}) ...")

        try:
            df = scrape_wiki_single(
                url=row["url"],
                tournament_name=row["tournament_name"],
                tier=int(row["tier"]),
            )
        except Exception as e:
            print(f"  WARNING: scrape failed ({e}) - skipping.\n")
            continue

        if df.empty:
            print(f"  WARNING: No matches extracted - skipping.\n")
            continue

        df.insert(3, "start_date", row["start_date"])
        df.insert(4, "host_country", row["host_country"])
        all_frames.append(df)
        scraped_names.append(row["tournament_name"])
        n_pending = int(df["is_pending"].sum()) if "is_pending" in df.columns else 0
        suffix = f" ({n_pending} pending)" if n_pending else ""
        print(f"  OK: {len(df)} matches extracted{suffix}.\n")

        time.sleep(2)

    if not all_frames and existing is None:
        print("ERROR: No data was collected. Check config and scraper.")
        return pd.DataFrame()

    if existing is not None:
        # Replace rescraped tournaments' rows; keep everything else untouched.
        kept = existing[~existing["tournament"].isin(scraped_names)]
        new = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

        # Safety gate: a rescrape should never lose completed matches.
        for name in scraped_names:
            old_done = (
                (existing["tournament"] == name) & (existing["is_pending"] == 0)
            ).sum()
            new_done = (
                (new["tournament"] == name) & (new["is_pending"] == 0)
            ).sum() if not new.empty else 0
            if new_done < old_done:
                print(f"ABORT: '{name}' shrank from {old_done} to {new_done} completed "
                      f"matches - keeping the existing CSV untouched.")
                return existing

        master = pd.concat([kept, new], ignore_index=True)
        master["start_date"] = master["start_date"].astype(str)
        # Stable sort - preserves in-tournament bracket order (MC topology)
        master = (master.sort_values(["start_date", "tournament"], kind="stable")
                  .reset_index(drop=True))
    else:
        master = pd.concat(all_frames, ignore_index=True)

    # Fold Wikipedia's spelling variants together before anything downstream
    # sees them. Doing it here, at the single point raw_matches.csv is written,
    # means every consumer (features, the app, the exporter) is consistent
    # without each having to remember to canonicalise.
    master = canonicalise(master)

    # Canonicalising can *create* duplicates, so the de-dupe has to follow it.
    # The scraper already drops a match repeated across a page's half-bracket
    # and "Finals" tables, but it does that on the raw names - and a classic-era
    # page can spell the same player two ways in those two tables, which is one
    # pair to us and two pairs to the scraper. The 2013 World Championships
    # writes "Jan Ø. Jørgensen" in one and "Jan O. Jorgensen" in the other, and
    # shipped its quarter-final twice the moment those were aliased together.
    # Prefer the row carrying a score, matching the scraper's own rule.
    if {"tournament", "round", "player_a", "player_b"} <= set(master.columns):
        pair = master.apply(
            lambda r: (r["tournament"], r["round"],
                       frozenset((r["player_a"], r["player_b"]))), axis=1)
        scored = (master["score"].notna()
                  & (master["score"].astype(str).str.strip() != "")).astype(int)
        keep = (master.assign(_pair=pair, _scored=scored)
                .sort_values("_scored", ascending=False, kind="stable")
                .drop_duplicates("_pair", keep="first").index)
        master = master.loc[sorted(keep)].reset_index(drop=True)

    master.to_csv(output_path, index=False)

    n_pending = int(master["is_pending"].sum()) if "is_pending" in master.columns else 0
    print(f"{'='*60}")
    print(f"Done. Total matches: {len(master)}  (pending: {n_pending})")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")

    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape BWF match results from Wikipedia")
    parser.add_argument("--incremental", action="store_true",
                        help="Only scrape missing/pending/recent tournaments and merge "
                             "into the existing CSV (default: full rescrape)")
    args = parser.parse_args()
    run_orchestrator(incremental=args.incremental)
