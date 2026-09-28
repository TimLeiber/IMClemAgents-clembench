# RQ3: task–harness compatibility

Run from the repository root:

```bash
python project_interpretation/Q_3/functions/plot_harness_scores.py
```

Outputs in `plots/`: `harness_scores_by_game.pdf`, a PNG preview, aggregate means in CSV, and `configuration_scores.csv` for auditing the included model scores. Re-running replaces these analysis outputs only. No inference or scoring is invoked and source results remain unchanged.

Each bar is the arithmetic mean of a harness's per-model Clemscores for one game. Vanilla is excluded. Wordle and Chronicle include all six evaluated models, with 15 episodes per configuration. Their existing `raw.csv` metrics supply unrounded scores. Aborted episodes contribute zero, equivalent to completion fraction times mean quality among completed episodes.

Geolocate includes Qwen3.8 27B and GLM-5.3-Flash. It uses `Recovered clemscore` from `results_geolocate_convergence_replay/timeout_recovery_remaining_guesses_01/results.csv`. These are full-cohort scores combining remaining-guesses timeout recovery with unchanged episodes, not averages over recovered episodes alone. Unrecovered failures remain failures.

For the report caption, state that Geolocate uses recovery and only two models, whereas the text games use six models. Consequently, cross-game differences also reflect model composition and execution conditions, not task effects alone. No explanatory annotations are embedded in the plot.

The text-game results include the audited Eidos Gemma rerun installed on 28 September 2026. Its 150 episodes replace the historical Gemma cohort, including vanilla, using the neutral analysis identifier `gemma4-e4b`. Original run metadata retains the Ollama runtime identity and precision. The other models and Geolocate results are unchanged.

An alternate recovery export or output directory can be selected with `--recovery-results PATH` or `--output-dir PATH`. Every expected configuration must exist and contain 15 episodes. Missing configurations are not silently counted as zero.

## Task-related tool use

For the Chronicle-only comparison, run `python project_interpretation/Q_3/functions/plot_chronicle_tools.py` after generating the tool counts below. This saves `chronicle_web_tools_vs_clemscore.pdf`, a PNG, and the plotted CSV in `plots/`. Four bars show episode incidence of web search, retrieval, or browsing on the left axis. A line shows mean Clemscore across the six models on the right axis. Both axes use 0–100. All 90 episodes per harness are included, and aborted episodes contribute zero to Clemscore. There is no Geolocate aggregation in this plot. The caption, rather than annotations inside the plot, should explain the measures and that the association is observational.

```bash
python project_interpretation/Q_3/functions/count_task_tools.py
```

This reads existing traces for all 840 harness episodes, including aborted and timed-out runs. It exports `task_tool_summary.csv`, `task_tool_by_model.csv`, `task_tool_episodes.csv`, and `task_tool_calls.csv` in `plots/`. Source results are not modified and no inference or scoring runs.

The primary measure is episodes with at least one observed call in the selected categories, not the number of calls. Wordle counts code/shell execution. Chronicle counts web search, retrieval, and browsing. Geolocate counts the union of execution, web, and image-analysis interfaces. Explicit native web-tool invocations inside execution arguments also count as web use. These are interface-based proxies for potentially relevant investigation, not verified successful or helpful actions. Arbitrary network requests inside shell code are not classified as web calls. Category totals may overlap, but each call contributes at most once to the combined relevant-call count.

Call IDs deduplicate native events and repeated conversation histories. For Codex episodes without captured events in `agent_loop.json`, the existing host `agent_trace.log` is parsed in memory. The audit export includes source paths, call IDs, categories, and argument hashes rather than argument contents. The `captured` field identifies episodes without observed model activity. One OpenClaw Wordle episode lacks capture, so its all-90-episode incidence is a lower bound, not proof of no tool use in that episode.

Geolocate counts investigation in the original episodes. Recovery answers are not additional harness tool calls. Text games have 90 episodes per harness and Geolocate has 30. Differences across games also reflect the different model cohorts. These observational counts do not isolate why a harness scores better.
