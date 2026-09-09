# Geolocate

`geolocate` is a single-player, three-turn multimodal geolocation game for
clembench. The player receives four consecutive views from one short KartaView
sequence and iteratively predicts the reference frame's latitude, longitude,
and country together with a short explanation.

## Response

The response starts with a one-line explanation, followed by decimal latitude
and longitude coordinates and a country name. It is parsed as exactly four
labelled lines so format adherence remains measurable.

After each non-final valid prediction, the game returns a
programmatic distance band, country correctness, and whether the prediction
moved closer to or farther from the target. Exact distances are retained in
the interaction log but are not exposed as feedback. The third prediction is
the scored final prediction; a correct-country prediction within one kilometre
ends the episode early.

## Metrics

- Haversine distance in kilometres
- Inverse-distance quality on a 0--100 scale
- Exponential-distance quality as a diagnostic alternative
- Country accuracy
- Per-round distance and quality progression
- Initial, best, and final-improvement quality
- Standard clembench played, aborted, and request-adherence metrics

The inverse-distance score is the episode's quality score. Both distance
curves currently use 100 km as their half-score distance. Aborted episodes are
excluded from mean quality and reduce the aggregate played proportion through
the normal clembench evaluation path.

## Counterfactual recovery

`scripts/run_convergence_replay.py` is separate from the live harness pipeline.
The default `--guess-mode one-shot` retains the original one-prediction probe.
`--guess-mode remaining-guesses` instead permits `max_rounds - predictions
already submitted` further predictions, one per completion. It returns the
normal distance band, country correctness, relative progress, and remaining
guess count after each non-final prediction. Exact target coordinates and
distances remain evaluation-only. No auxiliary tools are provided.

Recovery stops on a correct-country prediction within the success radius,
exhausted guesses, or a malformed response. The final legal prediction is
scored, not the best historical prediction; malformed output aborts the
recovery. All non-selected original episode outcomes remain unchanged in the
full-cohort comparison. Recovery adds inference after timeout and is not an
equal-compute harness comparison.

Use a distinct `--run-name` for each experiment. Requests, provider responses,
scores, and feedback are saved under each episode's `attempt_001/round_NNN`.
`--resume` reuses saved exchanges and continues with the first missing request.
The token limit is per prediction. `--dry-run` reconstructs inputs without
inference. Source results and prior replay experiments are never overwritten.

Run the remaining-guesses experiment with the same reduced-reasoning closer
settings as the final one-shot experiment:

```bash
python geolocate/scripts/run_convergence_replay.py \
  --results-dir results_geolocate \
  --guess-mode remaining-guesses \
  --run-name timeout_recovery_remaining_guesses_01 \
  --temperature 1 \
  --max-tokens 50000 \
  --evidence-char-limit 250000 \
  --event-char-limit 3000 \
  --model-extra-body 'glm-5.3-flash={"reasoning":{"enabled":true,"effort":"low"}}' \
  --model-extra-body 'qwen3.8-27b-reasoning={"reasoning":{"enabled":false}}'
```

After completion, export the comparison and all metrics, including vanilla:

```bash
python geolocate/scripts/export_convergence_results.py \
  results_geolocate_convergence_replay/timeout_recovery_remaining_guesses_01
```

## Preparing the images

KartaView imagery is licensed under CC BY-SA 4.0. The unsuffixed manifest and
`resources/images/` are the original city-centre smoke-control set. They remain
untouched so existing smoke instances cannot silently acquire different images
while retaining their old target coordinates.

The hard-set preparation uses the entries in `location_seeds.json` only as
coverage anchors. It samples deterministic, area-uniform probe points in three
annuli 15--40 km, 40--100 km, and 100--200 km from those anchors. KartaView is
queried around the probes, not at the anchor coordinates. The resulting
candidate pool is deduplicated by sequence. Targets more than 1.25 km from their
sampled probe are rejected. The final 30-location set uses 12 peripheral, 9
secondary-town, and 9 rural/interurban locations, with no repeated country.

Audit a candidate pool, download 30 balanced locations into versioned paths,
and then generate the hard instances:

```bash
python geolocate/scripts/prepare_kartaview_dataset.py --audit --pool-size 300
python geolocate/scripts/prepare_kartaview_dataset.py --download --limit 30
python geolocate/instancegenerator.py
```

The default random seed is fixed, so repeating discovery with the same source
API state produces the same probe plan and selection. The hard files are:

- `resources/kartaview_candidates_hard_v1.json`
- `resources/dataset_manifest_hard_v1.json`
- `resources/attribution_hard_v1.json`
- `resources/images_hard_v1/`

If KartaView advertises a frame whose underlying storage blob is unavailable,
the downloader records that candidate as an exclusion, deterministically
reselects a valid country-compatible candidate, and only publishes the image
directory and manifest after all 120 images have been materialized.

The generated hard experiment is named `kartaview_hard_4view`. Exact KartaView
source metadata and sampling diagnostics remain in the preparation files but
are not copied into model-facing game instances.

The generated attribution record retains source URLs, photo IDs, contributor
IDs, licensing, and the applied projection. Source metadata is not included in
the model-facing game instances.

## Layout

- `master.py`: player, game state, game master, scorer, and benchmark classes
- `instancegenerator.py`: manifest-backed fixed-image instance generation
- `utils.py`: coordinate parsing and geospatial scoring helpers
- `resources/experiment_config.json`: language-independent game constants
- `resources/langconfig.json`: language-specific response labels
- `resources/initial_prompts/`: prompt templates
- `resources/images/`: original easy-control images
- `resources/images_hard_v1/`: derived hard-set images after preparation
- `resources/attribution*.json`: source and licence metadata for each set
- `in/instances.json`: generated clembench instances

## Counterfactual convergence replay

`scripts/run_convergence_replay.py` tests whether the evidence accumulated by
an external agent before an episode timeout was already sufficient for a useful
answer. It is an offline proof of concept, not part of the regular benchmark
pipeline: source interactions are never changed, and replay scores are written
to a separate results directory.

The replay input contains only the original images, player-visible game
messages, and a bounded rendering of the standardized agent trace. Exact target
coordinates and internal game-master evaluations are used only after inference
to score the counterfactual prediction.

First inspect which episodes would be selected without calling a model:

```bash
python geolocate/scripts/run_convergence_replay.py \
  --results-dir results_geolocate \
  --source-agent openclaw-with-qwen3.8-27b-reasoning \
  --limit 5 \
  --dry-run
```

Then run one constrained multimodal completion per selected timeout:

```bash
python geolocate/scripts/run_convergence_replay.py \
  --results-dir results_geolocate \
  --source-agent openclaw-with-qwen3.8-27b-reasoning \
  --model qwen3.8-27b-no-reasoning \
  --limit 5 \
  --temperature 1 \
  --max-tokens 2000
```

Each replay records the bounded evidence, activity counters, raw model
exchange, strict four-line parse result, normal Geolocate distance metrics,
and a counterfactual clemscore. Invalid one-shot responses score zero.
