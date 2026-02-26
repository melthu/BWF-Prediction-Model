import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUTPUT_PATH = "data/config/tournaments_config.csv"
MIN_TIER = 100  # Keep Super 100 and above; drop everything else

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

LEVEL_MAP = {
    "World Tour Finals": 1500,
    "Super 1000":        1000,
    "Super 750":          750,
    "Super 500":          500,
    "Super 300":          300,
    "Super 100":          100,
}

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_start_date(date_text: str, year: int) -> str | None:
    """
    Convert date strings like '7–12 January', '14–19 January' → YYYY-MM-DD.
    Takes the first (start) day of the range.
    Normalises en-dash, em-dash, and minus variants before parsing.
    """
    text = re.sub(r"[–—−]", "-", date_text.strip())
    m = re.match(r"(\d+)\s*(?:-\s*\d+\s*)?([A-Za-z]+)", text)
    if m:
        day = int(m.group(1))
        month = MONTH_MAP.get(m.group(2).lower())
        if month:
            try:
                return pd.Timestamp(year=year, month=month, day=day).strftime("%Y-%m-%d")
            except Exception:
                return None
    return None


def get_tier(cell) -> int | None:
    """Read tier from '<li><b>Level:</b> Super 1000</li>' inside the tournament cell."""
    for li in cell.find_all("li"):
        b = li.find("b")
        if b and "level" in b.get_text().lower():
            text = re.sub(r"Level\s*:\s*", "", li.get_text(), flags=re.IGNORECASE).strip()
            for label, value in LEVEL_MAP.items():
                if label.lower() in text.lower():
                    return value
    return None


def get_host_country(cell) -> str | None:
    """
    Read host country from '<li><b>Host:</b> Kuala Lumpur, Malaysia</li>'.
    Takes everything after the last comma as the country.
    """
    for li in cell.find_all("li"):
        b = li.find("b")
        if b and "host" in b.get_text().lower():
            text = re.sub(r"Host\s*:\s*", "", li.get_text(), flags=re.IGNORECASE).strip()
            if "," in text:
                return text.split(",")[-1].strip()
            return text.strip()
    return None


def get_draw_url(cell, year: int) -> str | None:
    """
    Primary:  <a> with visible text 'Draw' that points to /wiki/ (not a redlink).
    Fallback: first <a> whose href matches /wiki/{year}_ (the year-specific page).
    Redlinks (/w/index.php?...&redlink=1) are intentionally ignored — the page
    doesn't exist yet and can't be scraped.
    """
    for a in cell.find_all("a"):
        if a.get_text().strip().lower() == "draw":
            href = a.get("href", "")
            if href.startswith("/wiki/"):
                return "https://en.wikipedia.org" + href
    # Fallback: year-specific link on the tournament name itself (2021-era pages)
    for a in cell.find_all("a"):
        href = a.get("href", "")
        if re.match(rf"/wiki/{year}_", href):
            return "https://en.wikipedia.org" + href
    return None


def get_tournament_name(cell, year: int) -> str | None:
    """
    Extract the bolded tournament name link, then append the year.

    Handles two Wikipedia structures:
      2025+: <b><a href="/wiki/Malaysia_Open">Malaysia Open</a></b> (<a>Draw</a>)
      2021–: <b><span class="flagicon">…</span> <a href="/wiki/Swiss_Open">Swiss Open</a></b>

    In both cases: find first <a> inside <b> that is NOT inside a flagicon span.
    """
    for b_tag in cell.find_all("b"):
        for a in b_tag.find_all("a"):
            if a.find_parent("span", class_="flagicon"):
                continue
            name = a.get_text().strip()
            if name and name.lower() not in ("draw", "report", "results"):
                return f"{name} {year}"
    return None


def scrape_year(year: int) -> list[dict]:
    url = f"https://en.wikipedia.org/wiki/{year}_BWF_World_Tour"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            print(f"  {year}: page not found (404) — skipping.")
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  {year}: request failed — {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tournaments = []
    seen_urls: set[str] = set()

    # Each tournament block has exactly one <td> containing '<li><b>Level:</b> ...>'
    # That cell also holds the Draw link, host info, and sits in the same <tr> as the date cell.
    for cell in soup.find_all("td"):
        if "Level" not in cell.get_text():
            continue

        tier = get_tier(cell)
        if tier is None or tier < MIN_TIER:
            continue

        draw_url = get_draw_url(cell, year)
        if not draw_url or draw_url in seen_urls:
            continue

        tournament_name = get_tournament_name(cell, year)
        host_country    = get_host_country(cell)
        if not tournament_name or not host_country:
            continue

        # Date lives in the first <td> of the same <tr> (both use rowspan, so they
        # share the anchor row in the HTML even though they span many visual rows).
        parent_tr  = cell.find_parent("tr")
        start_date = None
        if parent_tr:
            date_cell = parent_tr.find("td")
            if date_cell:
                start_date = parse_start_date(date_cell.get_text().strip(), year)

        if not start_date:
            continue

        seen_urls.add(draw_url)
        tournaments.append({
            "url":             draw_url,
            "tournament_name": tournament_name,
            "tier":            tier,
            "start_date":      start_date,
            "host_country":    host_country,
        })

    print(f"  {year}: {len(tournaments)} Super 100+ tournaments found.")
    return tournaments


def build_config(output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    all_rows = []
    for year in range(2021, 2027):  # 2021 – 2026
        print(f"Scraping {year} BWF World Tour...")
        rows = scrape_year(year)
        all_rows.extend(rows)
        time.sleep(2)

    if not all_rows:
        print("ERROR: No tournaments found.")
        return pd.DataFrame()

    df = (
        pd.DataFrame(all_rows)
        .sort_values("start_date")
        .reset_index(drop=True)
    )
    df.to_csv(output_path, index=False)

    print(f"\nFinal config shape : {df.shape}")
    print(f"Tier breakdown     :\n{df['tier'].value_counts().sort_index()}")
    print(f"\nSaved to: {output_path}\n")
    print("HEAD (10):")
    print(df.head(10).to_string(index=True))

    return df


if __name__ == "__main__":
    build_config()
