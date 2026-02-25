# Spec: `scraper_wiki_single.py` (BWF Wikipedia MS Scraper V2)

## 1. Context & Objective
Extract strictly **Men's Singles** match outcomes from a BWF Wikipedia page. 
We are refining the output to serve as direct inputs for a machine learning model by enforcing a binary target label and extracting the tournament round.

## 2. Extraction Logic 
**Step 1: Isolate the Men's Singles Section**
* Locate the Men's Singles section header (e.g., `id` containing "Men's_singles").
* Iterate through sibling elements to capture the bracket tables (`<table class="wikitable">` or tables with font-size styling), breaking when you hit the next discipline header (e.g., "Women's singles").

**Step 2: Round Extraction via Table Headers**
* For each bracket table, extract the text of the `<th>` tags in the first row. These typically contain the round names (e.g., "Semi-finals", "Final", "First round", "Quarter-finals").
* Because Wikipedia uses `rowspan` to draw brackets, depth mapping can be tricky. Attempt to track the horizontal depth/column index of the `<td>` containing the player to map it to the corresponding `<th>` round name. If a clean mapping fails, fallback to a sequential deduction or mark as "Unknown".

**Step 3: Match Pairing & Binary Outcome**
* Scan the table to find players via `<span class="flagicon">` tags preceding an `<a>` tag.
* Group them sequentially into pairs (Player A vs Player B).
* **The Binary Target:** Check which player has the `<b>` (bold) tag formatting. 
  * If Player A is bolded, `player_a_won = 1`. 
  * If Player B is bolded, `player_a_won = 0`.

**Step 4: Clean Disambiguation**
* Apply a regex to strip Wikipedia parenthetical disambiguations from player names: `re.sub(r'\s*\(.*?\)', '', name).strip()`

## 3. Output Schema
Return a Pandas DataFrame with the exact columns:
1. `tournament` (str)
2. `tier` (int)
3. `round` (str - e.g., "First round", "Quarter-finals")
4. `player_a` (str)
5. `player_b` (str)
6. `player_a_won` (int: 1 or 0)

## 4. Execution Request
Update the script and run the `if __name__ == "__main__":` block on `https://en.wikipedia.org/wiki/2026_Malaysia_Open_(badminton)`. Print the DataFrame to verify the binary labels and rounds are correct.


# Spec: `scraper_orchestrator.py` (BWF Macro-Scraper)

## 1. Context & Objective
Scale the validated `scraper_wiki_single.py` script. Create a configuration-driven orchestrator that loops through multiple Wikipedia tournament pages, extracts the Men's Singles matches, injects the tournament's start date, and outputs a single master dataset.

## 2. The Configuration File (`tournaments_config.csv`)
Create this CSV file in the root directory. It dictates what the orchestrator scrapes. 
Include the following columns:
1. `url`: The Wikipedia URL.
2. `tournament_name`: String (e.g., "Malaysia Open 2026").
3. `tier`: Integer (e.g., 1000, 750, 500).
4. `start_date`: Format YYYY-MM-DD (Crucial for time-series feature engineering later).

*Action:* Please populate this file with 4 to 5 actual recent BWF World Tour tournaments (e.g., 2026 Malaysia Open, 2025 World Tour Finals, 2025 China Masters) to serve as our initial test batch.

## 3. Orchestrator Logic
Write `scraper_orchestrator.py` to perform the following:
1. **Load Config:** Read `tournaments_config.csv` using pandas.
2. **Initialize:** Create an empty list to store the match DataFrames.
3. **Loop:** Iterate through each row in the config. For each tournament:
   * Print a console log indicating which tournament is being scraped.
   * Call the `scrape_wiki_single(url, tournament_name, tier)` function from our micro-scraper.
   * Create a new column in the resulting DataFrame called `start_date` and assign the value from the config row.
   * Append the DataFrame to the list.
   * **Politeness:** Implement `time.sleep(2)` to avoid rate-limiting from Wikipedia.
4. **Compile:** Concatenate all DataFrames in the list into a single master DataFrame.

## 4. Final Output & Validation
* Save the concatenated DataFrame to a file named `raw_matches.csv` in the root directory.
* Print the total number of matches extracted across all tournaments.
* Print the `.head(10)` and `.tail(10)` of the master DataFrame to verify the dates and concatenation worked properly.