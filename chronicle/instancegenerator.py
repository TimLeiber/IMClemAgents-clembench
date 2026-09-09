"""
Instance generator for the Chronicle game.

Fetches event summaries from Wikipedia, extracts forbidden keywords via spaCy NER,
assigns difficulty by page-view count, and writes instances.json.

Usage:
    python instancegenerator.py [--seed SEED] [--no-cache]

The generated instances.json will contain 6 experiments x 10 instances = 60 instances.
Raw Wikipedia data is cached to resources/events/events_cache.json to avoid re-fetching
on subsequent runs. Use --no-cache to force a fresh fetch.
"""

import os
import re
import json
import time
import random
import logging
import argparse
import requests
from collections import Counter

from clemcore.clemgame import GameInstanceGenerator

N_INSTANCES = 10       # instances per experiment
N_EXPERIMENTS = 6      # 3 categories x 2 difficulties
MAX_TURNS = 10         # rounds per episode
MAX_DETECTIVE_WORDS = 100
CANDIDATES_PER_CATEGORY = 80   # randomly sampled from the pool before enrichment

EVENT_POOL_FILE = "resources/events/events.json"   # output of fetch_events.py
CACHE_FILE = "resources/events/events_cache.json"

CATEGORIES = ["battles", "discoveries", "political"]
DIFFICULTIES = ["easy", "hard"]

# Continents/broad regions/hemispheres/cardinal-direction adjectives — too generic
# to forbid (e.g. "South America" for a Paraguay expedition still leaves thousands
# of possible events, but makes the description nearly impossible to narrow down).
# Only applied to NER-derived keywords from the Wikipedia summary, NOT to words
# drawn from the event name itself (those are deliberately specific/identifying).
GEO_STOPLIST = {
    "africa", "asia", "europe", "north america", "south america", "central america",
    "the americas", "oceania", "antarctica", "australia", "middle east",
    "western europe", "eastern europe", "northern europe", "southern europe",
    "north", "south", "east", "west", "northern", "southern", "eastern", "western",
    "northeast", "northwest", "southeast", "southwest",
}

logger = logging.getLogger(__name__)


class ChronicleInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))
        self.nlp = None  # spaCy model, loaded lazily

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def on_generate(self, seed: int, **kwargs):
        random.seed(seed)
        use_cache = not kwargs.get("no_cache", False)

        event_pool = self.load_json(EVENT_POOL_FILE)

        cache_path = os.path.join(os.path.dirname(__file__), CACHE_FILE)
        if use_cache and os.path.exists(cache_path):
            print("Loading events from cache...")
            with open(cache_path, "r", encoding="utf-8") as f:
                events_by_category = json.load(f)
        else:
            sampled_pool = {}
            for category, entries in event_pool.items():
                n = min(CANDIDATES_PER_CATEGORY, len(entries))
                sampled_pool[category] = random.sample(entries, n)
                print(f"  {category}: sampled {n} of {len(entries)} candidates")

            print("Fetching events from Wikipedia...")
            events_by_category = self._fetch_all_events(sampled_pool)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(events_by_category, f, indent=2, ensure_ascii=False)
            print(f"Cache written to {cache_path}")

        narrator_template = self.load_template("resources/initial_prompts/narrator_initial")
        detective_template = self.load_template("resources/initial_prompts/detective_initial")

        for category in CATEGORIES:
            events = events_by_category.get(category, [])
            if not events:
                print(f"WARNING: no events for category '{category}', skipping.")
                continue

            # Sort by page views descending; assign difficulty by thirds. The middle
            # third is intentionally dropped (only easy/hard are generated) to keep a
            # clear page-view gap between the easy and hard tiers.
            events_sorted = sorted(events, key=lambda e: e.get("page_views", 0), reverse=True)
            n = len(events_sorted)
            tier_size = n // 3

            tiers = {
                "easy": events_sorted[:tier_size],
                "hard": events_sorted[2 * tier_size:],
            }

            for difficulty in DIFFICULTIES:
                exp_name = f"{category}_{difficulty}"
                pool = tiers[difficulty]

                if len(pool) < N_INSTANCES:
                    print(f"WARNING: only {len(pool)} events for {exp_name}, need {N_INSTANCES}.")

                selected = random.sample(pool, min(N_INSTANCES, len(pool)))

                experiment = self.add_experiment(exp_name)
                experiment["max_turns"] = MAX_TURNS
                experiment["max_detective_words"] = MAX_DETECTIVE_WORDS
                experiment["narrator_initial_prompt"] = narrator_template
                experiment["detective_initial_prompt"] = detective_template

                for instance_id, event in enumerate(selected):
                    game_instance = self.add_game_instance(experiment, instance_id)
                    game_instance["target_event"] = event["event_name"]
                    game_instance["event_category"] = category
                    game_instance["difficulty"] = difficulty
                    game_instance["event_summary"] = event.get("summary", "")
                    game_instance["forbidden_keywords"] = event.get("forbidden_keywords", [])
                    game_instance["acceptable_answers"] = event.get("acceptable_answers", [])

        print("Done.")

    # ------------------------------------------------------------------
    # Wikipedia fetching
    # ------------------------------------------------------------------

    def _fetch_all_events(self, seed_events: dict) -> dict:
        result = {}
        for category, entries in seed_events.items():
            print(f"  Fetching {len(entries)} events for category '{category}'...")
            result[category] = []
            for entry in entries:
                event_name = entry["event_name"]
                wiki_title = entry["wikipedia_title"]
                print(f"    {event_name} ...", end=" ", flush=True)
                event_data = self._fetch_event(event_name, wiki_title)
                if event_data:
                    result[category].append(event_data)
                    print("OK")
                else:
                    print("SKIP (fetch failed)")
                time.sleep(0.5)  # be polite to Wikipedia API
        return result

    def _fetch_event(self, event_name: str, wiki_title: str) -> dict | None:
        summary = self._fetch_wikipedia_summary(wiki_title)
        if not summary:
            return None

        page_views = self._fetch_page_views(wiki_title)
        redirects = self._fetch_redirects(wiki_title)
        forbidden_keywords = self._extract_forbidden_keywords(summary, event_name)
        acceptable_answers = self._build_acceptable_answers(event_name, redirects)

        return {
            "event_name": event_name,
            "wikipedia_title": wiki_title,
            "summary": summary,
            "page_views": page_views,
            "forbidden_keywords": forbidden_keywords,
            "acceptable_answers": acceptable_answers,
        }

    def _fetch_wikipedia_summary(self, title: str) -> str | None:
        encoded = requests.utils.quote(title.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        headers = {"User-Agent": "chronicle-game-instancegenerator/1.0"}
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=15, headers=headers)
                if resp.status_code == 200:
                    return resp.json().get("extract", "").strip() or None
                elif resp.status_code == 429:
                    wait = 2 ** attempt * 2
                    logger.warning(f"Wikipedia rate-limited for {title}, waiting {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"Wikipedia summary {title}: HTTP {resp.status_code}")
                    return None
            except Exception as e:
                logger.warning(f"Wikipedia summary {title} attempt {attempt+1}: {e}")
                time.sleep(1)
        return None

    def _fetch_redirects(self, title: str) -> list[str]:
        """Other titles that redirect to this Wikipedia article — common alternate
        names (e.g. "Hague Opium Convention" -> "International Opium Convention")
        that the detective is likely to guess instead of the canonical title."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query", "titles": title, "prop": "redirects",
            "rdlimit": "50", "format": "json",
        }
        headers = {"User-Agent": "chronicle-game-instancegenerator/1.0"}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                return []
            pages = resp.json().get("query", {}).get("pages", {})
            redirects = []
            for page in pages.values():
                for r in page.get("redirects", []):
                    redirects.append(r["title"])
            return redirects
        except Exception as e:
            logger.warning(f"Wikipedia redirects {title}: {e}")
            return []

    def _fetch_page_views(self, title: str) -> int:
        encoded = requests.utils.quote(title.replace(" ", "_"))
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/all-agents/{encoded}/monthly/20230101/20231231"
        )
        headers = {"User-Agent": "chronicle-game-instancegenerator/1.0"}
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=15, headers=headers)
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    return sum(item.get("views", 0) for item in items)
                elif resp.status_code == 429:
                    time.sleep(2 ** attempt * 2)
                    continue
                else:
                    return 0
            except Exception:
                time.sleep(1)
        return 0

    # ------------------------------------------------------------------
    # Forbidden keyword extraction
    # ------------------------------------------------------------------

    def _extract_forbidden_keywords(self, summary: str, event_name: str) -> list[str]:
        keywords = set()

        # Always add content words from the event name itself
        for word in re.findall(r'\b[A-Za-z]{3,}\b', event_name):
            w = word.lower()
            if w not in {"the", "of", "and", "in", "at", "on", "by", "to", "a", "an",
                         "for", "its", "was", "is", "are", "were", "been",
                         "battle", "war", "revolution", "discovery", "invention",
                         "development", "theory", "siege", "fall", "rise", "signing",
                         "formation", "movement", "first", "second", "third"}:
                keywords.add(w)

        # Use spaCy NER to extract named entities from the summary
        ner_entities = self._run_ner(summary)
        for ent in ner_entities:
            keywords.add(ent.lower())

        return sorted(keywords)

    def _run_ner(self, text: str) -> list[str]:
        """Extract named entities using spaCy NER, with a regex fallback."""
        truncated = text[:3000]

        try:
            if self.nlp is None:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("\nDownloading spaCy model en_core_web_sm...")
                    from spacy.cli import download
                    download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")

            doc = self.nlp(truncated)
            # Only keep entities that appear more than once — these are the key identifiers
            word_count = Counter(ent.text.lower() for ent in doc.ents
                                 if ent.label_ in ("PERSON", "GPE", "ORG", "NORP", "LOC"))
            return [w for w, count in word_count.items()
                    if count >= 2 and len(w) >= 3 and w not in GEO_STOPLIST]

        except Exception as e:
            logger.warning(f"spaCy NER failed ({e}), using regex fallback")
            # Regex fallback: capitalized words appearing > 1 time (not sentence-initial)
            # Find all capitalised proper-noun candidates (not at sentence start)
            candidates = re.findall(r'(?<=[.!?]\s)(?![A-Z])|(?<=[^.!?\n]\s)([A-Z][a-z]{2,})', truncated)
            word_count = Counter(w for w in candidates if w)
            return [w.lower() for w, c in word_count.items()
                    if c > 1 and w.lower() not in GEO_STOPLIST]

    # ------------------------------------------------------------------
    # Acceptable answers
    # ------------------------------------------------------------------

    def _build_acceptable_answers(self, event_name: str, redirects: list[str] | None = None) -> list[str]:
        answers = set()
        name_lower = event_name.lower()
        answers.add(name_lower)

        # Without leading "the"
        if name_lower.startswith("the "):
            answers.add(name_lower[4:])

        # With "the" prefix
        answers.add(f"the {name_lower}")

        # Drop common leading words for a shorter variant
        for prefix in ("battle of ", "siege of ", "discovery of ", "invention of ",
                       "development of the ", "development of ", "fall of the ",
                       "fall of ", "signing of the ", "signing of ", "formation of the ",
                       "formation of "):
            if name_lower.startswith(prefix):
                answers.add(name_lower[len(prefix):])
                break

        # Without a leading/trailing/parenthesised year — titles like "1973 oil
        # crisis" or "Battle of X (1944)" are hard to guess exactly since the
        # detective has no way to know the precise year from the description.
        no_year = re.sub(r'^\d{3,4}\s+', '', name_lower)
        no_year = re.sub(r'\s*\(\d{3,4}\)\s*$', '', no_year)
        no_year = re.sub(r'\s+\d{3,4}$', '', no_year)
        if no_year != name_lower and no_year.strip():
            answers.add(no_year.strip())

        # Wikipedia redirects are real-world alternate names for the same event
        # (e.g. "Hague Opium Convention" -> "International Opium Convention"),
        # so a detective guessing one should count as correct.
        for redirect in redirects or []:
            answers.add(redirect.lower())

        return sorted(answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chronicle game instances.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-fetch from Wikipedia (ignore local cache).")
    args = parser.parse_args()

    ChronicleInstanceGenerator().generate(seed=args.seed, no_cache=args.no_cache)
