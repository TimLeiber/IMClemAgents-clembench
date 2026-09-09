"""
Fetch historical events from Wikidata across multiple categories.

Runs SPARQL queries sequentially, merges and deduplicates results, requires
each event to have an English Wikipedia article (needed downstream for
summary/page-view fetching), maps Wikidata categories onto the three
Chronicle game categories, and writes resources/events/events.json.

Each event keeps its sitelinks count (number of Wikipedia language editions
with an article on it) in the output — a better obscurity signal than
English page views alone, since it is independent of English-speaking-world
attention. Queries fetch the complete matching population per category (no
server-side LIMIT/ORDER BY) so this signal reflects the true distribution;
events.json is sorted ascending by sitelinks as a convenience, but any
future game mode can re-filter/re-sort on the stored field directly.

Usage:
    pip install requests pandas
    python fetch_events.py

Runtime: a few minutes for all 18 categories (largest single category,
"battle", fetches ~11k rows in under 10s once unsorted; "trial" and
"migration" each return high-thousands). Output is cached to events.json
and only needs regenerating if you want a fresh sample.

Output:
    resources/events/events.json — {"battles": [...], "discoveries": [...], "political": [...]}
"""

import os
import time
import json
import requests
import pandas as pd

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "historical-events-benchmark/1.0 (research project)"}

OUT_PATH = os.path.join(os.path.dirname(__file__), "resources", "events", "events.json")

# Each category: (name, Wikidata Q-id). Every Q-id below was verified by
# resolving Special:EntityData/{qid}.json and checking the label/description
# actually matches the category name, then test-querying for real row counts
# and spot-checking sample titles (2026-06-17) — the original list had 9/20
# Q-ids pointing at unrelated items (e.g. scientific_discovery -> "string" the
# datatype, expedition -> an anime film, protest -> a fish family), which
# silently filled "discoveries" with linguistic/computing terms and leaked
# footbridges/canal locks into "political" via trade_route. "empire" and
# "exploration_voyage" were dropped entirely: no Wikidata class exists for
# either as an *event* (the former is a state, the latter overlaps with the
# fixed "expedition" class) after a real search.
CATEGORIES = [
    ("battle",               "Q178561"),
    ("war",                  "Q198"),
    ("revolution",           "Q10931"),
    ("coup",                 "Q45382"),
    ("treaty",               "Q131569"),
    ("expedition",           "Q2401485"),   # "expedition": discovery/research trip
    ("scientific_discovery", "Q12772819"),  # "discovery": act of detecting something new
    ("trial",                "Q8016240"),   # "trial": tribunal dispute resolution
    ("coronation",           "Q209715"),    # "coronation"
    ("rebellion",            "Q124734"),
    ("siege",                "Q188055"),
    ("diplomatic_conference","Q1072326"),   # "summit": meeting of heads of state/government
    ("election",             "Q40231"),
    ("assassination",        "Q3882219"),
    ("protest",              "Q273120"),    # "protest" — see NON_TRANSITIVE below
    ("colonial_settlement",  "Q133156"),
    ("migration",            "Q177626"),    # "human migration"
    ("trade_route",          "Q1574938"),   # "trade route"
]

# Categories queried with direct P31 only (no P279* subclass closure).
# "protest" (Q273120) has a broken Wikidata subclass chain: "The Holocaust"
# (Q2763) is itself transitively P279* under "protest", which drags Operation
# Reinhard / Kristallnacht / massacre-type articles into protest results.
# Direct P31 avoids that without losing real protest articles (verified:
# 200+ genuine protests/demonstrations, zero genocide-related hits).
# "expedition" (Q2401485) has the same problem: WWII army formations like
# "8th Army (Wehrmacht)" are transitively P279* under "expedition" and were
# confirmed (live transcript, 2026-06-17) to surface as "discoveries"
# instances. Direct P31 keeps real expeditions (Franklin's lost expedition,
# Japanese Antarctic Expedition, Coppermine expedition, etc. all verified).
NON_TRANSITIVE = {"protest", "expedition"}

# Maps each Wikidata category onto one of the three Chronicle game categories.
GAME_CATEGORY_MAP = {
    "battle": "battles",
    "war": "battles",
    "siege": "battles",
    "rebellion": "battles",
    "scientific_discovery": "discoveries",
    "expedition": "discoveries",
    "revolution": "political",
    "coup": "political",
    "treaty": "political",
    "trial": "political",
    "coronation": "political",
    "diplomatic_conference": "political",
    "election": "political",
    "assassination": "political",
    "protest": "political",
    "colonial_settlement": "political",
    "migration": "political",
    "trade_route": "political",
}

# Requires a real English Wikipedia article (schema:isPartOf en.wikipedia.org)
# so every result can be fetched by instancegenerator.py's Wikipedia summary step.
QUERY_TEMPLATE = """
SELECT DISTINCT ?event ?eventLabel ?wikipediaTitle ?date ?sitelinks WHERE {{
  ?event wdt:P31{transitive} wd:{qid} .
  OPTIONAL {{ ?event wdt:P585 ?date . }}
  OPTIONAL {{ ?event wdt:P580 ?date . }}
  ?event wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= 2)
  ?article schema:about ?event ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?wikipediaTitle .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
}}
"""
# No ORDER BY / LIMIT: sorting server-side forces Wikidata to fully sort every
# matching row before truncating (measured ~50s for "battle" vs ~9s unsorted
# and uncapped), and a LIMIT without a sort would keep an arbitrary slice with
# no relationship to obscurity. Fetching everything is fast enough (a few
# minutes across all categories) and keeps the full sitelinks distribution
# intact for any future obscurity-based filtering.


def run_query(category_name: str, qid: str, retries: int = 2) -> pd.DataFrame:
    transitive = "" if category_name in NON_TRANSITIVE else "/wdt:P279*"
    query = QUERY_TEMPLATE.format(qid=qid, transitive=transitive)
    for attempt in range(retries + 1):
        try:
            r = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=60,
            )
            r.raise_for_status()
            bindings = r.json()["results"]["bindings"]

            rows = []
            for b in bindings:
                rows.append({
                    "uri":             b["event"]["value"],
                    "label":           b.get("eventLabel", {}).get("value", ""),
                    "wikipedia_title": b.get("wikipediaTitle", {}).get("value", ""),
                    "date":            b.get("date", {}).get("value", ""),
                    "sitelinks":       int(b.get("sitelinks", {}).get("value", 0)),
                    "category":        category_name,
                })

            print(f"  [{category_name}] {len(rows)} results")
            return pd.DataFrame(rows)

        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"  [{category_name}] timed out, retrying ({attempt + 1}/{retries})...")
                continue
            print(f"  [{category_name}] TIMED OUT after {retries} retries — skipping")
            return pd.DataFrame()
        except Exception as e:
            print(f"  [{category_name}] ERROR: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def assign_game_category(categories_str: str) -> str | None:
    """Pick a single game category from a '|'-joined set of matched Wikidata categories."""
    for cat in categories_str.split("|"):
        if cat in GAME_CATEGORY_MAP:
            return GAME_CATEGORY_MAP[cat]
    return None


def main():
    all_frames = []

    for category_name, qid in CATEGORIES:
        print(f"Querying: {category_name} ({qid})")
        df = run_query(category_name, qid)
        if not df.empty:
            all_frames.append(df)
        time.sleep(2)  # be polite to the endpoint

    if not all_frames:
        print("No results fetched. Exiting.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows before dedup: {len(combined)}")

    # Deduplicate by URI, keep the union of all matched categories
    combined["category"] = combined.groupby("uri")["category"].transform(
        lambda x: "|".join(sorted(set(x)))
    )
    combined = combined.drop_duplicates(subset="uri")
    print(f"Total rows after dedup:  {len(combined)}")

    # Drop entries with no usable label or no English Wikipedia title
    combined = combined[combined["label"].str.strip() != ""]
    combined = combined[~combined["label"].str.startswith("Q")]  # unlabeled Q-items
    combined = combined[combined["wikipedia_title"].str.strip() != ""]
    print(f"Total rows after label/title filter: {len(combined)}")

    # Different Wikidata items can point at the same Wikipedia article — keep one
    combined = combined.drop_duplicates(subset="wikipedia_title")

    # Drop numbered-series entries (e.g. "Expedition 7", "Expedition 31") —
    # the only fact distinguishing one from its siblings is the number itself,
    # which a description stripped of identifying details can't convey, so
    # these are effectively unguessable (confirmed live, 2026-06-17: three
    # different ISS "Expedition N" targets all lost, detective could only
    # ever guess "the ISS program" in general).
    combined = combined[~combined["wikipedia_title"].str.contains(r"\b\d+$", regex=True)]
    print(f"Total rows after numbered-series filter: {len(combined)}")

    combined["game_category"] = combined["category"].apply(assign_game_category)
    combined = combined.dropna(subset=["game_category"])
    print(f"Total rows after game-category mapping: {len(combined)}")

    combined = combined.sort_values("sitelinks")

    events_by_category = {"battles": [], "discoveries": [], "political": []}
    for _, row in combined.iterrows():
        events_by_category[row["game_category"]].append({
            "event_name": row["wikipedia_title"],
            "wikipedia_title": row["wikipedia_title"],
            "sitelinks": int(row["sitelinks"]),
            "wikidata_categories": row["category"].split("|"),
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(events_by_category, f, indent=2, ensure_ascii=False)

    for cat, events in events_by_category.items():
        print(f"  {cat}: {len(events)} events")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
