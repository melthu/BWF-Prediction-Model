import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


def scrape_wiki_single(url: str, tournament_name: str, tier: int) -> pd.DataFrame:
    """
    Scrapes Men's Singles match results from a BWF tournament Wikipedia page.

    Uses the 'Section & Bold' strategy:
    1. Isolates the Men's Singles section by navigating mw-heading2 divs.
    2. Maps each bracket table's columns to round names via <th> header cells.
    3. Tracks true column indices (accounting for rowspan/colspan) to assign rounds.
    4. Extracts player nationality from the <a title> inside each flagicon span.
    5. Determines the winner by checking if the flagicon's parent is a <b> tag.
    """
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        timeout=15,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # --- Step 1: Find the mw-heading2 div wrapping the Men's Singles h2 ---
    ms_heading_div = None
    for div in soup.find_all("div", class_="mw-heading"):
        h = div.find(["h2", "h3"])
        if h and re.search(r"men.?s singles", h.get_text(), re.IGNORECASE):
            ms_heading_div = div
            break

    EMPTY_COLS = ["tournament", "tier", "round", "player_a", "player_a_nat", "player_b", "player_b_nat", "player_a_won"]

    if ms_heading_div is None:
        print("ERROR: Could not find a 'Men's Singles' section header on this page.")
        return pd.DataFrame(columns=EMPTY_COLS)

    stop_pattern = re.compile(r"(women|doubles|mixed)", re.IGNORECASE)
    ms_tables = []
    for sib in ms_heading_div.find_next_siblings():
        if sib.name == "div" and "mw-heading2" in sib.get("class", []):
            if stop_pattern.search(sib.get_text()):
                break
        if sib.name == "table":
            ms_tables.append(sib)

    if not ms_tables:
        print("ERROR: Found the Men's Singles header but no bracket tables beneath it.")
        return pd.DataFrame(columns=EMPTY_COLS)

    # --- Step 2: Build column→round map from the header row ---
    def build_round_ranges(table):
        """
        Parse the first row of a bracket table to produce a list of
        (start_col, end_col, round_name) for every non-empty header cell.
        """
        rows = table.find_all("tr")
        if not rows:
            return []
        col = 0
        ranges = []
        for cell in rows[0].find_all(["th", "td"]):
            cs = int(cell.get("colspan", 1))
            text = cell.get_text().strip()
            if text:
                ranges.append((col, col + cs - 1, text))
            col += cs
        return ranges

    def classify_table(table):
        """Returns 'bracket', 'group_match', or 'skip'."""
        rows = table.find_all("tr")
        if not rows:
            return "skip"
        headers = {cell.get_text().strip() for cell in rows[0].find_all(["th", "td"])}
        if "Player 1" in headers and "Player 2" in headers:
            return "group_match"
        if headers & {"Seeds", "Rank", "NOCs", "W", "L", "Pld", "Pts", "Nation"}:
            return "skip"
        return "bracket"

    def col_to_round(col_idx, ranges):
        for start, end, name in ranges:
            if start <= col_idx <= end:
                return name
        return "Unknown"

    # --- Step 3: Walk each table row-by-row, tracking true column positions ---
    def extract_player_cells(table):
        """
        Returns an ordered list of (col_idx, player_name, nationality, is_winner).

        Uses the standard HTML table rendering algorithm: maintain a set of
        column indices already claimed by rowspan cells from prior rows, so
        each cell gets its true visual column position despite colspan/rowspan.

        Wikipedia bracket structure for a player cell:
          Winner:     <td> <b> <span class="flagicon">...</span> <a title="Name">…</a> </b> </td>
          Non-winner: <td>     <span class="flagicon">...</span> <a title="Name">…</a>     </td>

        Nationality: the <a title> inside the flagicon (e.g., "China", "South Korea").
        """
        rows = table.find_all("tr")
        col_occupancy = {}  # row_idx -> set of occupied column indices
        result = []

        for ri, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(["td", "th"]):
                # Advance past any columns claimed by rowspans from earlier rows
                while col_idx in col_occupancy.get(ri, set()):
                    col_idx += 1

                cs = int(cell.get("colspan", 1))
                rs = int(cell.get("rowspan", 1))

                # Reserve this cell's entire span in the occupancy map
                for r in range(ri, ri + rs):
                    for c in range(col_idx, col_idx + cs):
                        col_occupancy.setdefault(r, set()).add(c)

                # Extract player info if this cell contains a flagicon
                flagicon = cell.find("span", class_="flagicon")
                if flagicon:
                    is_winner = flagicon.parent.name == "b"

                    # Nationality: <a title> inside the flagicon (e.g. "China", "South Korea")
                    # WTF group tables sometimes link to "Indonesia national badminton team"
                    # instead of the country directly — clean that up as a fallback.
                    flag_link = flagicon.find("a")
                    nationality = None
                    if flag_link:
                        raw_nat = flag_link.get("title") or (flag_link.find("img") or {}).get("alt", "")
                        nationality = re.sub(r"national badminton team", "", raw_nat, flags=re.IGNORECASE).strip() or None

                    # Player link = first <a> in the cell NOT inside the flagicon span
                    # AND whose title does not contain "national badminton team"
                    # (WTF group tables inject a team link adjacent to the player name)
                    player_link = None
                    for a in cell.find_all("a"):
                        if a.find_parent("span", class_="flagicon"):
                            continue
                        if "national badminton team" in (a.get("title") or "").lower():
                            continue
                        player_link = a
                        break

                    if player_link:
                        name = player_link.get("title") or player_link.get_text().strip()
                        # Strip Wikipedia parenthetical disambiguations e.g. "(badminton)"
                        name = re.sub(r"\s*\(.*?\)", "", name).strip()
                        if name:
                            result.append((col_idx, name, nationality, is_winner))

                col_idx += cs

        return result

    # --- Step 4: Assemble players with round labels, then pair sequentially ---
    all_players = []  # (round_name, player_name, nationality, is_winner)
    for table in ms_tables:
        table_type = classify_table(table)
        if table_type == "skip":
            continue
        if table_type == "group_match":
            for _, name, nationality, is_winner in extract_player_cells(table):
                all_players.append(("Group stage", name, nationality, is_winner))
        else:
            round_ranges = build_round_ranges(table)
            for col_idx, name, nationality, is_winner in extract_player_cells(table):
                round_name = col_to_round(col_idx, round_ranges)
                all_players.append((round_name, name, nationality, is_winner))

    matches = []
    for i in range(0, len(all_players) - 1, 2):
        round_a, player_a, nat_a, a_wins = all_players[i]
        round_b, player_b, nat_b, _ = all_players[i + 1]

        # Both players in a pair share the same round; prefer the non-Unknown one
        round_name = round_a if round_a != "Unknown" else round_b

        if player_a == player_b:
            continue

        matches.append(
            {
                "tournament": tournament_name,
                "tier": tier,
                "round": round_name,
                "player_a": player_a,
                "player_a_nat": nat_a,
                "player_b": player_b,
                "player_b_nat": nat_b,
                "player_a_won": 1 if a_wins else 0,
            }
        )

    return pd.DataFrame(matches)


if __name__ == "__main__":
    test_url = "https://en.wikipedia.org/wiki/2026_Malaysia_Open_(badminton)"
    df = scrape_wiki_single(url=test_url, tournament_name="Malaysia Open 2026", tier=1000)

    if df.empty:
        print("Extraction failed or returned empty DataFrame.")
    else:
        print(f"Success! Extracted {len(df)} Men's Singles matches.\n")
        print(df.to_string(index=True))
