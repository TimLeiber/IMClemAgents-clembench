# Geolocate

A single-player multimodal game: identify a location from four consecutive
KartaView images. The player has up to three guesses, each containing exactly
four lines:

```text
EXPLANATION: <one-line explanation>
LATITUDE: <decimal latitude>
LONGITUDE: <decimal longitude>
COUNTRY: <country name>
```

After each non-final guess, the game reports a distance band, whether the
country is correct, and whether the guess moved closer or farther away.
Exact distances are not revealed. A correct-country guess within 1 km ends
the game early; otherwise, the third guess is final. Malformed responses abort
the episode.

## Experiments and dataset

There are **three experiments**, defined by the type of location:

- `civic_public`: public and civic places
- `roadside_commerce`: shops and roadside businesses
- `transport_nodes`: transport facilities and junctions

The reviewed dataset contains 30 locations, ten per category. The official
benchmark uses five per category: **15 instances in total**, each with four images.

The dataset files have different purposes; they are not separate experiments:

| Path | Purpose |
|---|---|
| `resources/kartaview_candidates_final.json` | Candidate locations from dataset preparation |
| `resources/dataset_manifest_final.json` | Selected locations, coordinates, countries and image paths |
| `resources/attribution_final.json` | Image sources, contributors and licensing information |
| `resources/images_final/` | Downloaded images |

Dataset preparation finds locations near OpenStreetMap features, retrieves
nearby KartaView images, and subjects the selections to manual review.
`resources/review_final.json` records the decisions.
See `scripts/prepare_kartaview_evidence_dataset.py --help` for preparation options.

The playable instance files are `in/instances.json` (30 instances) and
`in/instances_geolocate_official.json` (the fixed 15 used in the evaluation).
To regenerate the 30-instance file from the prepared manifest, run from the
repository root:

```bash
python geolocate/instancegenerator.py
```

## Scoring

For the final guess, quality is `100 / (1 + distance_km / 100)`, using haversine
distance. Aggregate clemscore is mean quality multiplied by the fraction of
episodes played without aborting.

Logs also include country accuracy, per-guess progress, and exponential scores
with half-score distances of 100 km and 5 km. Scoring uses the final guess,
not the best earlier guess.

## Running

For a vanilla model, run from the repository root:

```bash
clem run --game geolocate --models glm-5.3-flash \
  --instances_filename instances_geolocate_official \
  --results_dir test_results --temperature 1 -l 50000
```

Add `--experiment_name civic_public` to select one category. For harness runs,
scoring and transcripts, see the [project README](../README.md).

## Counterfactual recovery

`scripts/run_convergence_replay.py` probes whether an agent's accumulated
evidence can produce a useful prediction after a timeout. It provides the
original images and visible history, plus the recorded investigation, without
auxiliary tools.

`--guess-mode one-shot` allows one prediction; `--guess-mode remaining-guesses`
allows the unused guesses with normal game feedback. Other original outcomes
remain unchanged in the comparison. Recovery adds inference after the timeout
and is reported separately from the benchmark.

Use `--help` for run options and `--dry-run` to inspect selected episodes without
calling a model. `scripts/export_convergence_results.py <run-directory>` produces
comparison tables including vanilla scores and alternative distance metrics.
