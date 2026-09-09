import os
import re
import time
from datetime import date
from urllib.parse import quote, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUTPUT_PATH = "data/config/tournaments_config.csv"
MIN_TIER = 100  # Keep Super 100 and above; drop everything else

# World Tour era runs 2018 → present. The +2 end bound lets next season's
# calendar page be picked up as soon as Wikipedia publishes it; years whose
# page doesn't exist yet 404 and are skipped gracefully.
WORLD_TOUR_FIRST_YEAR = 2018
WORLD_TOUR_LAST_YEAR  = date.today().year + 1

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# BWF World Tour (2018-present)
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
    Convert date strings to YYYY-MM-DD (takes the first/start day of a range).

    Handles two formats:
      '7–12 January'  → day-first   (World Tour 2018+)
      'January 18'    → month-first  (Super Series 2010-2017)
    """
    # Fold the non-breaking space to a plain one FIRST, and keep a space out of
    # the dash class below. Wikipedia used to write these cells "7–12&nbsp;January",
    # where the nbsp survived the sub and \s* absorbed it; it now serves a plain
    # U+0020, which the class rewrote to "-" - "7-12-January" matches neither
    # pattern. Every year of both eras then scraped zero tournaments while
    # build_config still exited 0, so CI stayed green for five weeks on a
    # calendar frozen at 2026-07-28.
    text = re.sub(r"[–−—,]", "-", date_text.replace("\xa0", " ").strip())
    # Day-first: "18 January" or "18-23 January"
    m = re.match(r"(\d+)\s*(?:-\s*\d+\s*)?\s*([A-Za-z]+)", text)
    if m:
        day = int(m.group(1))
        month = MONTH_MAP.get(m.group(2).lower())
        if month:
            try:
                return pd.Timestamp(year=year, month=month, day=day).strftime("%Y-%m-%d")
            except Exception:
                return None
    # Month-first: "January 18"
    m = re.match(r"([A-Za-z]+)\s+(\d+)", text)
    if m:
        month = MONTH_MAP.get(m.group(1).lower())
        day = int(m.group(2))
        if month:
            try:
                return pd.Timestamp(year=year, month=month, day=day).strftime("%Y-%m-%d")
            except Exception:
                return None
    return None


def get_tier(cell, level_map: dict = LEVEL_MAP) -> int | None:
    """Read tier from '<li><b>Level:</b> Super 1000</li>' inside the tournament cell."""
    for li in cell.find_all("li"):
        b = li.find("b")
        if b and "level" in b.get_text().lower():
            text = re.sub(r"Level\s*:\s*", "", li.get_text(), flags=re.IGNORECASE).strip()
            for label, value in level_map.items():
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


def article_url(a) -> str | None:
    """
    Absolute URL of an <a> that points at a real Wikipedia article, else None.

    Wikipedia serves three href flavours depending on which parser rendered the
    page: legacy relative ('/wiki/Foo'), protocol-relative
    ('//en.wikipedia.org/wiki/Foo') and Parsoid absolute
    ('https://en.wikipedia.org/wiki/Foo'). Accept all three and normalise to
    https. Redlinks ('/w/index.php?...&redlink=1') point at pages that don't
    exist yet and are rejected either way.
    """
    href = a.get("href", "") or ""
    if "redlink" in href or "/w/index.php" in href:
        return None
    path = None
    if href.startswith("/wiki/"):
        path = href
    else:
        m = re.match(r"(?:https?:)?//en\.wikipedia\.org(/wiki/.+)", href)
        if m:
            path = m.group(1)
    if not path:
        return None
    # Parsoid serves non-ASCII titles unescaped; percent-encode so the config
    # holds one canonical spelling regardless of which parser rendered the page.
    return "https://en.wikipedia.org" + quote(unquote(path), safe="/:()_,'!.-")


def get_draw_url(cell, year: int) -> str | None:
    """
    Primary:  <a> with visible text 'Draw' pointing at an existing article.
    Fallback: first <a> whose target page is year-specific ('.../{year}_…'),
    i.e. the tournament-name link on 2021-era pages.
    """
    for a in cell.find_all("a"):
        if a.get_text().strip().lower() == "draw":
            url = article_url(a)
            if url:
                return url
    # Fallback: year-specific link on the tournament name itself (2021-era pages)
    for a in cell.find_all("a"):
        url = article_url(a)
        if url and url.startswith(f"https://en.wikipedia.org/wiki/{year}_"):
            return url
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


def _scrape_calendar_page(url: str, year: int, level_map: dict) -> list[dict]:
    """
    Core scraping logic for any BWF calendar page.
    Finds tournament cells containing '<li><b>Level:</b>…</li>' and extracts
    tier, draw URL, tournament name, host country, and start date.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            print(f"  {year}: page not found (404) - skipping.")
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  {year}: request failed - {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tournaments = []
    seen_urls: set[str] = set()

    for cell in soup.find_all("td"):
        if "Level" not in cell.get_text():
            continue

        tier = get_tier(cell, level_map)
        if tier is None or tier < MIN_TIER:
            continue

        draw_url = get_draw_url(cell, year)
        if not draw_url or draw_url in seen_urls:
            continue

        tournament_name = get_tournament_name(cell, year)
        host_country    = get_host_country(cell)
        if not tournament_name or not host_country:
            continue

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


def scrape_year(year: int) -> list[dict]:
    """Scrape a BWF World Tour calendar page (2018+)."""
    return _scrape_calendar_page(
        f"https://en.wikipedia.org/wiki/{year}_BWF_World_Tour",
        year, LEVEL_MAP,
    )


def scrape_superseries_year(year: int) -> list[dict]:
    """
    Scrape a BWF Super Series calendar page (2010-2017).

    These pages use a flat wikitable with columns:
      Tour# | Official title | Venue | City | Start | Finish | Prize | Report

    Unlike World Tour pages (2018+), there is no 'Level:' field - tier is
    inferred from the tournament name:
      'Finals'          → 1500
      'Premier'         → 750
      'Super Series'    → 500
    The 'Report' link IS the draw page passed to scraper_wiki_single.
    """
    url = f"https://en.wikipedia.org/wiki/{year}_BWF_Super_Series"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            print(f"  {year}: page not found (404) - skipping.")
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  {year}: request failed - {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    tournaments = []
    seen_urls: set[str] = set()

    table = soup.find("table", class_="wikitable")
    if not table:
        print(f"  {year}: no wikitable found - skipping.")
        return []

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue

        title_cell  = cells[1]
        date_cell   = cells[4]
        report_cell = cells[7]

        # Host country from flagicon in title cell
        host_country = None
        flagicon = title_cell.find("span", class_="flagicon")
        if flagicon:
            fa = flagicon.find("a")
            if fa:
                host_country = fa.get("title", "").strip()
        if not host_country:
            continue

        # Tournament name from first <a> NOT inside flagicon
        tournament_name = None
        for a in title_cell.find_all("a"):
            if a.find_parent("span", class_="flagicon"):
                continue
            name = a.get_text().strip()
            if name:
                tournament_name = f"{name} {year}"
                break
        if not tournament_name:
            continue

        # Tier from name keywords
        nl = tournament_name.lower()
        if "final" in nl:
            tier = 1500
        elif "premier" in nl:
            tier = 750
        elif "super series" in nl or "superseries" in nl:
            tier = 500
        else:
            continue

        # Start date ("January 18" format)
        start_date = parse_start_date(date_cell.get_text().strip(), year)
        if not start_date:
            continue

        # Draw URL = Report link
        draw_url = None
        for a in report_cell.find_all("a"):
            draw_url = article_url(a)
            if draw_url:
                break
        if not draw_url or draw_url in seen_urls:
            continue

        seen_urls.add(draw_url)
        tournaments.append({
            "url":             draw_url,
            "tournament_name": tournament_name,
            "tier":            tier,
            "start_date":      start_date,
            "host_country":    host_country,
        })

    print(f"  {year}: {len(tournaments)} Super Series tournaments found.")
    return tournaments


def scrape_world_championships(year: int) -> list[dict]:
    """
    The BWF World Championships, which no calendar page carries.

    `_scrape_calendar_page` only ever sees the World Tour and Super Series
    calendars, and the Championships is on neither - it is not a tour stop and
    has no "Level: Super NNN" line to be found by. It was therefore missing from
    the corpus entirely, along with every other major, so the sport's biggest
    title each year contributed nothing to anyone's Elo, form or head-to-head.

    Men's singles has its own page, which `scrape_wiki_single` reads in whole-
    page mode. Dates and host come off the parent page's infobox. Not held in
    Olympic years; those pages 404 and are skipped.

    Tier 1500, the same as the World Tour Finals - the top tier already in the
    vocabulary. A new value would be an unseen category for every model fitted
    before it, and the fitted Elo tier exponent is 0.06, so tier barely moves a
    rating anyway.
    """
    parent = f"https://en.wikipedia.org/wiki/{year}_BWF_World_Championships"
    try:
        resp = requests.get(parent, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            print(f"  {year}: no World Championships page - skipping.")
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  {year}: World Championships request failed - {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    box = soup.find("table", class_=re.compile("infobox"))
    if box is None:
        print(f"  {year}: World Championships page has no infobox - skipping.")
        return []

    dates = host = None
    for row in box.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if not (th and td):
            continue
        label = th.get_text(" ", strip=True).lower()
        if "date" in label and dates is None:
            dates = td.get_text(" ", strip=True)
        elif "location" in label and host is None:
            text = td.get_text(" ", strip=True)
            host = text.split(",")[-1].strip() or None

    start_date = parse_start_date(dates, year) if dates else None
    if not start_date or not host:
        print(f"  {year}: World Championships dates/host unreadable - skipping.")
        return []

    print(f"  {year}: World Championships on {start_date} in {host}.")
    return [{
        "url": (f"https://en.wikipedia.org/wiki/{year}_BWF_World_Championships"
                f"_%E2%80%93_Men%27s_singles"),
        "tournament_name": f"World Championships {year}",
        "tier": 1500,
        "start_date": start_date,
        "host_country": host,
    }]


def build_config(output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    all_rows = []

    # BWF Super Series era: 2010–2017
    for year in range(2010, 2018):
        print(f"Scraping {year} BWF Super Series...")
        rows = scrape_superseries_year(year)
        all_rows.extend(rows)
        time.sleep(2)

    # BWF World Tour era: 2018 → present (+1 season lookahead)
    for year in range(WORLD_TOUR_FIRST_YEAR, WORLD_TOUR_LAST_YEAR + 1):
        print(f"Scraping {year} BWF World Tour...")
        rows = scrape_year(year)
        all_rows.extend(rows)
        time.sleep(2)

    # Majors, which appear on no calendar page.
    for year in range(2010, WORLD_TOUR_LAST_YEAR + 1):
        print(f"Checking {year} BWF World Championships...")
        all_rows.extend(scrape_world_championships(year))
        time.sleep(2)

    if not all_rows:
        # A total failure has to be loud. Returning an empty frame here left the
        # existing config untouched and exited 0, so the workflow's "Rebuild
        # tournament calendar" step passed, `git diff -- data/` saw nothing, and
        # the scrape, retrain and commit all skipped without anything going red.
        # The 5% shrink guard below cannot catch this - it only ever runs when
        # the scrape returned *something*.
        print("ERROR: No tournaments found - refusing to continue.")
        raise SystemExit(1)

    df = (
        pd.DataFrame(all_rows)
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    # A partially-failed scrape (Wikipedia markup change, rate limiting, a run
    # of 404s) still yields *some* rows. Writing those would silently gut the
    # calendar the dashboard reads - refuse rather than overwrite.
    if os.path.exists(output_path):
        old = pd.read_csv(output_path)
        if len(df) < 0.95 * len(old):
            print(
                f"ERROR: scrape returned {len(df)} tournaments vs {len(old)} in the "
                f"existing config - looks like a partial failure. Not overwriting "
                f"{output_path}."
            )
            raise SystemExit(1)

    df.to_csv(output_path, index=False)

    print(f"\nFinal config shape : {df.shape}")
    print(f"Tier breakdown     :\n{df['tier'].value_counts().sort_index()}")
    print(f"\nSaved to: {output_path}\n")
    print("HEAD (10):")
    print(df.head(10).to_string(index=True))

    return df


if __name__ == "__main__":
    build_config()
