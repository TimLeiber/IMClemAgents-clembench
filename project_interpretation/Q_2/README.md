# RQ2: code feedback and process management

Run from the repository root:

```bash
python project_interpretation/Q_2/functions/plot_code_feedback.py
```

Defaults: Qwen3.8 2.4T-A95B, Hermes and OpenClaw, official Wordle and Chronicle traces. No inference, scoring, or result replacement occurs.

Outputs go to `plots/`: `code_feedback_by_harness.pdf`, a PNG preview, and CSV files containing episode counts, evidence excerpts, aggregate rates, and per-game rates. Re-running replaces these analysis outputs only.

Use `--games wordle`, `--model MODEL`, `--harnesses hermes openclaw`, or `--output-dir PATH` to select another comparison. A different output directory preserves the default figures. Extraction supports native tool-call/result events and OpenAI-style request histories. Additional harnesses require usable captured events and may need their tool names or result formats added to the detectors. This is not a claim of universal trace-format support.

## Definitions

- **Write-time diagnostic:** a structured tool result with `lint.status=error`.
- **Passing lint after repair:** a later write to the same resolved file path returns `lint.status=ok` after an observed error. This verifies only the lint transition, not program correctness. Repeated errors before a pass count as one repair sequence.
- **Background continuation:** an execution tool explicitly returns a still-running process or script. Later polls are not counted again as launches.
- **Process polling:** calls to `process` with action `poll` or `wait`, or empty `write_stdin` polling calls. Counts include repeated polls, not just unique processes.
- **Execution interruption:** an execution result explicitly reports SIGTERM, SIGKILL, or a command/execution timeout. A killed process does not establish why it was killed. These are not necessarily episode timeouts.

Only returned tool feedback is searched, not source code, reasoning, or tool documentation. Calls are deduplicated by identity across repeated request histories. Hermes CLI reconstructions with synthetic identities are skipped in favour of wire capture. The diagnostic detector intentionally does not count ordinary runtime errors as write-time lint feedback. Interruptions inside nested programmatic tool calls may not be observable unless their results are printed.

Bars show the percentage of eligible episodes containing at least one event, averaged equally across games. CSV files additionally contain raw event counts. Categories overlap. These are observational measurements, not a causal ablation or a measure of tool quality.

Episodes without any captured tool calls or assistant text are marked as missing capture, not assumed to have zero activity. The corresponding instance is excluded for every compared harness. Calls without results are reported separately. Observed counts are lower bounds when capture is incomplete.

## Current default cohort

The 30 selected episodes per harness yield 29 eligible matched pairs: 14 Wordle and 15 Chronicle. OpenClaw's Wordle `medium_frequency_words_no_clue_no_critic/instance_00003` lacks usable response/session capture, so that pair is excluded. No identified calls in the remaining extraction lack results.

Hermes has one episode with a write-time error followed by a passing lint result. OpenClaw has three episodes with background continuation and polling, comprising five background returns and eight polls. Each harness has one episode with an execution interruption. All these events occur in Wordle. This supports an illustrative interface difference, not a broad explanation of Hermes's advantage across both games.
