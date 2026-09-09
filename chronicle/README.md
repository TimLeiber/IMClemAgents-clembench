# Chronicle

A two-player historical deduction game for the clembench benchmark suite.

## Game Overview

One model plays the **Narrator** and one plays the **Detective**.

- The Narrator knows a secret historical event (a battle, scientific discovery, or political milestone) and must describe it across up to 10 rounds — one sentence at a time — without ever naming it or using any forbidden proper nouns.
- The Detective reads each new sentence and tries to identify the event after each clue.

The game ends as soon as the Detective guesses correctly, or after 10 rounds.

## Roles

| Role | Task |
|------|------|
| **Narrator** | Write exactly one new descriptive sentence each round. Must avoid all `forbidden_keywords`. |
| **Detective** | After each sentence, write a short `ANALYSIS` and a specific `GUESS`. |

## Response Format

**Narrator** (each round):
```
PARAGRAPH: <exactly one sentence describing the event without forbidden keywords>
```

**Detective** (each round, under 100 words):
```
ANALYSIS: <2–3 sentences of reasoning>
GUESS: <specific event name, e.g. "The Battle of Waterloo">
```

## Scoring

| Outcome | BENCH_SCORE |
|---------|-------------|
| Abort (format violation) | NaN |
| Lose (10 rounds, no correct guess) | 0 |
| Success on round T, violation rate V | `max(10, 110 − T×10) × (1 − V)` |

Maximum score: **100** (correct on round 1, zero forbidden-keyword violations).

Round 1 = 100, Round 2 = 90, ..., Round 10 = 10.

## Experiments

6 experiments × 10 instances = **60 episodes**.

| Experiment | Category | Difficulty |
|------------|----------|------------|
| `battles_easy/hard` | Military battles | By Wikipedia page views |
| `discoveries_easy/hard` | Science & invention | By Wikipedia page views |
| `political_easy/hard` | Political events | By Wikipedia page views |

Difficulty is assigned by page views: per category the events are sorted by views and split
into thirds — the top third becomes `easy`, the bottom third `hard`. The middle third is
intentionally excluded to keep a clear difficulty gap between the two tiers.

## Generating Instances

Instance generation is a two-step pipeline:

```bash
# 1. Build the candidate event pool from Wikidata (run rarely — a few minutes, ~38k events)
python3 chronicle/fetch_events.py

# 2. Sample from the pool, enrich with Wikipedia data, write instances.json
python3 chronicle/instancegenerator.py --seed 42
```

**Step 1** (`fetch_events.py`) queries Wikidata across 18 event-type categories (battles, wars, treaties, elections, etc.), requires each result to have a real English Wikipedia article, drops numbered-series entries (e.g. "Expedition 7"/"Expedition 31" — indistinguishable from each other once stripped of identifying detail), and writes `resources/events/events.json`. Each event keeps its **`sitelinks`** count — the number of Wikipedia language editions with an article on it — as a stored field. This is a proxy for global obscurity that's independent of English-language attention bias (a topic can be very well known in, say, German or Japanese Wikipedia while having low English page views, or vice versa). The current game mode does not filter on `sitelinks` — difficulty is assigned by English Wikipedia page views instead (see Experiments above) — but the field is preserved specifically so a **future game mode could add a `sitelinks`-based obscurity filter** (e.g. only sampling events below some sitelinks threshold for a "hard" variant) without touching the Wikidata sourcing step at all.

Two categories (`protest`, `expedition`) are queried with direct `P31` only instead of the usual `P31`/`P279*` transitive closure — both have a broken subclass chain on Wikidata that pulls in unrelated content (Holocaust-related articles under "protest", WWII Wehrmacht army formations under "expedition") via bad intermediate classification edges. See `NON_TRANSITIVE` in `fetch_events.py`.

**Step 2** (`instancegenerator.py`) randomly samples `CANDIDATES_PER_CATEGORY` events per game category from `events.json`, fetches each one's Wikipedia summary, page-view count, and redirect list, runs spaCy NER to extract forbidden keywords, and writes `in/instances.json`. Results are cached to `resources/events/events_cache.json`; use `--no-cache` to force a fresh sample and re-fetch.

Forbidden-keyword extraction excludes broad geographic terms (continents, hemispheres, cardinal directions — see `GEO_STOPLIST`) since these were found to make obscure events nearly unguessable without meaningfully testing circumlocution skill (e.g. forbidding "South America" for a Paraguay expedition). `acceptable_answers` now also includes the event's Wikipedia redirect titles (real-world alternate names) and a year-stripped variant of the title, so a Detective's correct-but-differently-phrased guess (e.g. omitting a disambiguating year) still counts as a win.

## Running

```bash
# Mock run (no API calls)
clem run -g chronicle -m mock
clem score -g chronicle

# Live run
clem run -g chronicle -m <model-name>
clem score -g chronicle
```

## File Structure

```
chronicle/
├── clemgame.json
├── master.py               # Game logic, scorer, benchmark
├── fetch_events.py         # Wikidata sourcing -> resources/events/events.json
├── instancegenerator.py    # Sampling + enrichment -> in/instances.json
├── in/
│   └── instances.json
└── resources/
    ├── events/
    │   ├── events.json          # candidate pool (output of fetch_events.py)
    │   └── events_cache.json    # enriched cache (output of instancegenerator.py)
    └── initial_prompts/
        ├── narrator_initial.template
        └── detective_initial.template
```

For full design rationale and technical details see [`.claude/chronicle_design.md`](../.claude/chronicle_design.md).
