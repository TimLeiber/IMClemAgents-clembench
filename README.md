# IMClemAgents-clembench

Research snapshot containing Chronicle, Wordle, Geolocate, and SAT_MENU.
SAT_MENU is included as a game and proof of concept, without results.

## Layout

- `chronicle/`, `wordle/`, `geolocate/`, `sat_menu/`: games, templates, utilities, generators and saved instances
- `project_interpretation/`: report analysis functions and plots
- `model_registry.json`: seven models used in the retained experiments, including the Chronicle narrator
- `agent_registry.json`: 24 retained model–harness configurations
- `agent_registry.template.json`: one example for each of Codex, Claude Code, Hermes and OpenClaw
- `model_registry.json.template`: minimal example model registration
- `counterfactual_model_overrides.json`: recorded recovery overrides (GLM low reasoning; Qwen reasoning disabled)
- `transcribe_reasoning.py`: vanilla reasoning transcript exporter
- `results_chronicle/`, `results_wordle/`, `results_geolocate/`: official recorded results
- `results_geolocate_convergence_replay/`: final one-shot and remaining-guesses recovery experiments

Results are present in the local extraction but deliberately ignored by Git:
they contain approximately 40 GB of artifacts. They must be distributed as a
separate versioned archive or through a large-file storage policy before this
repository is published. A fresh Git clone alone will not contain results.

## Current dependency boundary

Games depend on `clemcore`; no engine or harness implementation is vendored here.
Run commands from this repository's root so clemcore discovers its game folders.

This is an extraction, not yet a certification against unmodified PyPI clemcore.
The verified environment uses the project's modified local clemcore 3.7.2 checkout.
Do not assume installing the same version number from PyPI supplies those changes.
The core dependency audit and separate harness package extraction remain future work.

Agent configurations live here, but their implementation belongs in the external
runner. The recovery script consumes saved uniform agent events; its optional
native-trace fallback currently imports the agent adapters from local clemcore.
Exporting saved recovery tables does not call a model or launch a harness.

## Model and agent configuration

Register a provider model in `model_registry.json`, following
`model_registry.json.template`. An agent's `agent_config.clem_model` refers to
that entry's `model_name`, not its provider-side `model_id`.

Copy the matching example from `agent_registry.template.json` to add a model
under an existing harness. `backend` selects the adapter. The examples show
Codex's sandbox setting, Claude Code's permission mode, Hermes's provider and
turn limit, and OpenClaw's logging options. Reasoning levels are model- and
harness-dependent; the example `high` value is not universally supported.
These benchmark configurations permit broad tool access; use only in the
intended isolated environment.

The registries contain only names present in the official results or recovery
records. They retain the current project configuration, not a claim that every
historical episode used identical settings. Recorded requests and per-episode
metadata remain the authority for what was actually sent. Recovery overrides
are separate from the main model settings; the JSON override file documents
them and is not automatically loaded by the replay script.

## Credentials

Only `key.json.template` is distributed, with empty API keys and a loopback
example endpoint. Create your local credential file from the repository root:

```bash
cp key.json.template key.json
```

Then edit `key.json` to supply an OpenRouter API key or an OpenAI-compatible
endpoint. The local file is untracked and excluded by `.gitignore`, so normal
Git staging does not include it. Do not force-add it with `git add -f` or put
credentials in the tracked template. If `key.json` already exists, edit it
directly instead of overwriting it with the copy command.

## Offline scoring and tables

With the project's existing environment activated:

```bash
clem score -r results_chronicle
clem score -r results_wordle
clem score -r results_geolocate
clem eval -r results_chronicle --std --sort clemscore
clem eval -r results_wordle --std --sort clemscore
clem eval -r results_geolocate --std --sort clemscore
python geolocate/scripts/export_convergence_results.py results_geolocate_convergence_replay/timeout_recovery_remaining_guesses_01
```

These commands do not run model inference. See individual game READMEs for game
rules and dataset preparation. Historical paths in recorded artifacts are
preserved rather than rewritten. Saved inputs should be used to reproduce the
evaluated dataset; querying a live imagery service again may produce other data.

## Publication precautions

No API credential file or legacy results directory was copied. Review recorded
traces for sensitive content before distribution. Retain upstream code licensing
and Geolocate attribution records; independently verify image redistribution
rights before publishing the image data. Existing game READMEs may retain
historical workflow instructions and are not all migrated run instructions.
