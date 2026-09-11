# IMClemAgents-clembench

Research snapshot containing Chronicle, Wordle, Geolocate, and SAT Menu.
SAT_MENU is included as a game and proof of concept, without results.

## Layout

- `chronicle/`, `wordle/`, `geolocate/`, `sat_menu/`: games, templates, utilities, generators and saved instances (`sat_menu/` was created but never used for agent benchmarking)
- `project_interpretation/`: report analysis functions and plots
- `model_registry.json`: seven models used in the retained experiments, including the Chronicle narrator
- `agent_registry.json`: 24 model–harness configurations
- `agent_registry.template.json`: one example for each of Codex, Claude Code, Hermes and OpenClaw
- `model_registry.json.template`: minimal example model registration
- `counterfactual_model_overrides.json`: recorded recovery overrides (GLM low reasoning; Qwen reasoning disabled)
- `transcribe_reasoning.py`: vanilla reasoning transcript exporter
- `results_chronicle/`, `results_wordle/`, `results_geolocate/`: official recorded results
- `results_geolocate_convergence_replay/`: final one-shot and remaining-guesses recovery experiments

Results are present in the local extraction but deliberately ignored by Git:
they contain approximately 40 GB of artifacts... More on this in the section on Agent-loop transcripts.

### Experiment size

Each main game uses 15 instances per configuration, evaluated vanilla and through
Codex, Claude Code, Hermes and OpenClaw. Chronicle and Wordle use Gemma 4 E4B,
Nemotron 3.5 Lightning, GPT-OSS 120B, Qwen3.8 27B, GLM-5.3-Flash and
Qwen3.8 2.4T-A95B. Geolocate uses Qwen3.8 27B and GLM-5.3-Flash.

Reasoning settings are recorded in the model and agent registries; explicit
harness-specific settings were used, so they should not all be described as
unchanged provider defaults. Chronicle additionally uses Qwen3.6-35B-A3B as its
fixed narrator, not as an evaluated model.

| Game | Evaluated models | Vanilla + harness configurations | Planned episodes |
|---|---:|---:|-----------------:|
| Chronicle | 6 | 6 + 24 |              450 |
| Wordle | 6 | 6 + 24 |              450 |
| Geolocate | 2 | 2 + 8 |              150 |
| **Original experiments** | | |         **1050** |


For `Geolocate` I ran completions on 65 contexts in two settings. This is independent of the original benchmarking runs:
Instances of the same models used during benchmarked, but with lowered reasoning were given context from timed out episodes of an agent to see if they could recover the agents reasoning and probe whether the agent was completely lost or just slowly converging toward an answer.

- **One-shot recovery:** 65 episode attempts, each with one final prediction
- **Remaining-guesses recovery:** 65 episode attempts, allowing the unused game
  guesses with normal feedback; 150 recovery predictions were recorded in total

Thus the retained project contains **1180 episode attempts: 1050 original
episodes + 65 one-shot recoveries + 65 remaining-guesses recoveries**. These are
not 1180 distinct locations/tasks or successful games. `SAT Menu` contributes game code and
instances, but no result episodes to these totals.

## Installation (Python 3.10+)

Install the required dependencies to run all games:

`pip install -r IMClemAgents-clembench/requirements.txt`

This will also install the `clem` CLI tool.

The `clem` CLI command operates relative to the current working directory, that is, the directory it is called from.

## Model and agent configuration

To add new custom models, populate the `model_registry.json` file with the required fields  (template is provided as *model_registry.json.template*).

The model registry entry must at least specify a name and a backend:

```json
{
  "model_name":"mymodel",
  "backend":"mybackend"
}
```

To add new custom agents, populate the `agent_registry.json` file with the required fields (template is provided as *agent_registry.template.json*).

The agent template provides an example for each supported harness. `agent_config.clem_model` refers to the `model_name` in the model registry.

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

## Running a game with a harness

The standalone runner command is `agentclem`, provided by
[IMClemAgents](https://github.com/TimLeiber/IMClemAgents). With that repository
checked out alongside this one, install it into your active environment:

```bash
python -m pip install -e ../IMClemAgents
```

Run from this repository's root with the agents package installed,
Docker running, and the `clemagents-sandbox:dev` image already built.
Unlike scoring or transcription, a run calls the configured model API and may
incur costs. Replace the placeholders below with your configuration values
before executing the command:

```bash
agentclem \
  --game geolocate \
  --agent <your-harness-with-your-model> \
  --instances_filename <your_instances> \
  --experiment_name <your_experiment> \
  --max-instances <n_instances> \
  --results_dir <your_results_dir> \
  --temperature <n_temp> \
  --episode-timeout <n_timeout>
```

`--agent` selects an entry in `agent_registry.json`, which determines the harness
and references its model through `clem_model` in `model_registry.json`.
`--instances_filename` selects a file under `geolocate/in/` without the `.json`
suffix; `--experiment_name` filters its experiment, and `--max-instances`
limits the number of selected episodes to run. The timeout is the wall-clock limit
in seconds for that episode, not a reasoning budget.

The pipeline starts an isolated container for each episode, connects the harness
to the game's MCP interface, and records game interactions and agent traces under
`--results_dir`. Use `test_results` for tests to keep them separate from official
results. Repeating a
run for the same configuration and instance can replace its test artifacts.
These outputs can then be scored and transcribed with the commands below by
substituting `test_results` for the results directory.

For single-player games, the harness controls `player_0` by default; no
`--models` argument is needed. In Chronicle, use `--agent-player player_1` for
the detective and `--models Qwen3.6-35B-A3B` for the native narrator. `--models`
selects non-harness players, not the model inside the harness.

## Offline scoring, tables and transcripts

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

### Game transcripts

Create the standard game transcripts (HTML and LaTeX):

```bash
clem transcribe -r results_chronicle
clem transcribe -r results_wordle
clem transcribe -r results_geolocate
```

Include provider-exposed reasoning in the vanilla transcript view:

```bash
python transcribe_reasoning.py -r results_chronicle -g chronicle
python transcribe_reasoning.py -r results_wordle -g wordle
python transcribe_reasoning.py -r results_geolocate -g geolocate
```

### Agent-loop transcripts

*Warning*: Full transcription of agent loops in this repository will consume a lot of storage!

Most storage is consumed by `agent_loop.json` files and
`agent_trace.log` files, i.e. the raw log produced by harnesses not the standard game transcripts. The extracted
snapshot contains approximately 19 GB and 16 GB of those two raw trace types,
respectively, and 2 GB of rendered `agent_loop.html` files. Standard game and
reasoning HTML transcripts are comparatively small. Keep the source records
for reproducibility; HTML views can be regenerated from them.

Render recorded harness messages, reasoning, tool calls and tool results as
`agent_loop.html` using the separately installed agents package:

```bash
agentclem-transcribe -r results_chronicle
agentclem-transcribe -r results_wordle
agentclem-transcribe -r results_geolocate
```
