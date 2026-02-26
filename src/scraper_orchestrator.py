import time

import pandas as pd

from scraper_wiki_single import scrape_wiki_single

CONFIG_PATH = "data/config/tournaments_config.csv"
OUTPUT_PATH = "data/raw/raw_matches.csv"


def run_orchestrator(config_path: str = CONFIG_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    # --- Load config ---
    config = pd.read_csv(config_path)
    print(f"Loaded config: {len(config)} tournaments to scrape.\n")

    all_frames = []

    for _, row in config.iterrows():
        print(f"[{row['start_date']}] Scraping: {row['tournament_name']} (Tier {row['tier']}) ...")

        df = scrape_wiki_single(
            url=row["url"],
            tournament_name=row["tournament_name"],
            tier=int(row["tier"]),
        )

        if df.empty:
            print(f"  WARNING: No matches extracted — skipping.\n")
            continue

        df.insert(3, "start_date", row["start_date"])
        df.insert(4, "host_country", row["host_country"])
        all_frames.append(df)
        print(f"  OK: {len(df)} matches extracted.\n")

        time.sleep(2)

    if not all_frames:
        print("ERROR: No data was collected. Check config and scraper.")
        return pd.DataFrame()

    # --- Compile and save ---
    master = pd.concat(all_frames, ignore_index=True)
    master.to_csv(output_path, index=False)

    print(f"{'='*60}")
    print(f"Done. Total matches: {len(master)}")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")
    print("HEAD (10):")
    print(master.head(10).to_string(index=True))
    print("\nTAIL (10):")
    print(master.tail(10).to_string(index=True))

    return master


if __name__ == "__main__":
    run_orchestrator()
