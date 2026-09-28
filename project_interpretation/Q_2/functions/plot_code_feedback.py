"""Measure observed code feedback and process management in recorded agent loops."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


REPOSITORY = Path(__file__).resolve().parents[3]
QUESTION = Path(__file__).resolve().parents[1]
LABELS = {"hermes": "Hermes", "openclaw": "OpenClaw", "codex": "Codex", "claude-code": "Claude Code"}
COLORS = {"hermes": "#0072B2", "openclaw": "#E69F00", "codex": "#009E73", "claude-code": "#CC6677"}
METRICS = {"diagnostics": "Write-time\ndiagnostic", "repairs": "Passing lint\nafter repair",
           "background": "Background\ncontinuation", "polls": "Process\npolling", "interruptions": "Execution\ninterruption"}
EXECUTION_TOOLS = {"terminal", "exec", "execute_code", "process", "exec_command", "write_stdin", "Bash"}
INTERRUPTION = re.compile(r"(?:process|command) (?:exited|terminated) (?:with|by) signal (?:SIGTERM|SIGKILL)"
                          r"|(?:command|execution|process|script) timed out|timeout exceeded", re.I)
BACKGROUND = re.compile(r"(?:command|process) still running|script running with cell ID", re.I)


def content_text(content: object) -> str:
    """Extract textual tool content without request metadata.

    Args:
        content (object): A string, content-block list, or structured result.

    Returns:
        str: Text available to the model.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return json.dumps(content, ensure_ascii=False)


def tool_records(trace: dict) -> list[dict]:
    """Merge native events and replayed request histories by tool-call identity.

    Args:
        trace (dict): Recorded agent-loop JSON.

    Returns:
        list[dict]: Unique calls with any captured results, in first-observed order.
    """
    records = {}
    for event in trace["events"]:
        # cli reconstructions use synthetic ids and may duplicate the wire history
        if event.get("source") == "hermes_cli":
            continue
        kind = event["type"]
        if kind == "tool_call":
            call_id = event.get("call_id")
            if not call_id:
                raise ValueError("Tool call lacks an identity and cannot be deduplicated safely")
            records.setdefault(call_id, {"call_id": call_id, "name": event["name"],
                                         "arguments": event.get("arguments", {})})
        elif kind == "tool_result":
            call_id = event.get("call_id")
            if call_id in records:
                records[call_id]["result"] = content_text(event.get("content", ""))
        elif kind == "model_request":
            if not event.get("payload") and not event.get("raw"):
                continue
            request = event.get("payload") or json.loads(event["raw"])
            for message in request.get("messages", []):
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    records.setdefault(call["id"], {"call_id": call["id"], "name": function["name"],
                                                    "arguments": arguments})
                if message.get("role") == "tool" and message.get("tool_call_id") in records:
                    records[message["tool_call_id"]]["result"] = content_text(message.get("content", ""))
    return list(records.values())


def analyze_records(records: list[dict]) -> tuple[dict, list[dict]]:
    """Count explicit diagnostics, subsequent lint passes, and process events.

    Args:
        records (list[dict]): Unique tool calls and captured results.

    Returns:
        tuple[dict, list[dict]]: Metric counts and auditable evidence rows.
    """
    counts = dict.fromkeys(METRICS, 0)
    evidence = []
    pending = {}
    for record in records:
        name = record["name"].split(".")[-1]
        arguments = record["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result = record.get("result", "")
        observed = []

        # only structured write-time lint errors count as diagnostics
        structured = json.loads(result, strict=False) if result.lstrip().startswith("{") and '"lint"' in result else {}
        lint = structured.get("lint", {}) if isinstance(structured, dict) else {}
        path = structured.get("resolved_path") if isinstance(structured, dict) else None
        path = path or arguments.get("path") or arguments.get("file_path")
        if lint.get("status") == "error":
            observed.append(("diagnostics", str(lint.get("output", ""))))
            if path:
                pending[path] = record["call_id"]
        elif lint.get("status") == "ok" and path in pending:
            observed.append(("repairs", f"{path}: passing lint after {pending.pop(path)}"))

        # process waiting is counted separately from the execution result itself
        if name == "process" and arguments.get("action") in {"poll", "wait"}:
            observed.append(("polls", json.dumps(arguments)))
        if name == "write_stdin" and not arguments.get("chars"):
            observed.append(("polls", json.dumps(arguments)))
        if name in EXECUTION_TOOLS:
            match = BACKGROUND.search(result)
            if match and name not in {"process", "write_stdin"}:
                observed.append(("background", result[:500]))
            match = INTERRUPTION.search(result)
            if match:
                observed.append(("interruptions", result[max(0, match.start() - 100):match.end() + 200]))
        for metric, excerpt in observed:
            counts[metric] += 1
            evidence.append({"metric": metric, "call_id": record["call_id"], "tool": name,
                             "path": path or "", "excerpt": excerpt})
    return counts, evidence


def collect_episodes(root: Path, model: str, harnesses: list[str], games: list[str]) -> tuple[list, list]:
    """Analyze official, instance-matched configurations without modifying results.

    Args:
        root (Path): Repository containing per-game result directories.
        model (str): Registry model name.
        harnesses (list[str]): Harness identifiers to compare.
        games (list[str]): Games to include.

    Returns:
        tuple[list, list]: Episode summaries and evidence records.
    """
    episodes, evidence = [], []
    for game in games:
        identities = {}
        for harness in harnesses:
            configuration = f"{harness}-with-{model}"
            folders = [p for p in (root / f"results_{game}").iterdir()
                       if p.is_dir() and p.name.split("--")[-1] == configuration]
            paths = sorted(p for folder in folders for p in (folder / game).glob("*/instance_*/agent_loop.json"))
            identities[harness] = {(p.parent.parent.name, p.parent.name) for p in paths}
            if not paths or len(paths) != len(identities[harness]):
                raise ValueError(f"Missing or duplicate episode traces for {configuration}/{game}")
            for path in paths:
                trace = json.loads(path.read_text())
                records = tool_records(trace)
                captured = bool(records) or any(e["type"] == "assistant_text" for e in trace["events"])
                counts, rows = analyze_records(records)
                identity = {"model": model, "harness": harness, "game": game,
                            "experiment": path.parent.parent.name, "episode": path.parent.name}
                episodes.append({**identity, **counts, "capture_observed": captured, "paired_eligible": False,
                                 "tool_calls": len(records),
                                 "calls_without_results": sum("result" not in r for r in records),
                                 "empty_request_records": sum(e["type"] == "model_request" and not e.get("payload")
                                                              and not e.get("raw") for e in trace["events"]),
                                 "trace": str(path.relative_to(root))})
                evidence.extend({**identity, **row, "trace": str(path.relative_to(root))} for row in rows)
        if any(ids != identities[harnesses[0]] for ids in identities.values()):
            raise ValueError(f"Harness episodes do not match on {game}")
        missing = {(e["experiment"], e["episode"]) for e in episodes
                   if e["game"] == game and not e["capture_observed"]}
        for episode in episodes:
            if episode["game"] == game:
                episode["paired_eligible"] = (episode["experiment"], episode["episode"]) not in missing
    return episodes, evidence


def summarize(episodes: list[dict], harnesses: list[str], games: list[str]) -> list[dict]:
    """Average episode incidence equally across the selected games.

    Args:
        episodes (list[dict]): Per-episode metric counts.
        harnesses (list[str]): Harness order.
        games (list[str]): Equally weighted games.

    Returns:
        list[dict]: Counts and percentages for each harness and metric.
    """
    summary = []
    for harness in harnesses:
        selected = [e for e in episodes if e["harness"] == harness and e["paired_eligible"]]
        for metric in METRICS:
            rates = []
            for game in games:
                group = [e for e in selected if e["game"] == game]
                if not group:
                    raise ValueError(f"No paired captured episodes for {harness}/{game}")
                rates.append(100 * sum(e[metric] > 0 for e in group) / len(group))
            summary.append({"harness": harness, "metric": metric, "episodes": len(selected),
                            "episodes_with_event": sum(e[metric] > 0 for e in selected),
                            "event_count": sum(e[metric] for e in selected),
                            "mean_episode_percent": sum(rates) / len(rates)})
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    """Save audit records as UTF-8 CSV.

    Args:
        path (Path): Output filename.
        rows (list[dict]): Nonempty records with uniform fields.

    Returns:
        None: Writes the CSV file.
    """
    if not rows:
        raise ValueError(f"No records for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict], harnesses: list[str], output: Path) -> None:
    """Draw labelled bars without titles or explanatory annotations.

    Args:
        summary (list[dict]): Harness-level percentages.
        harnesses (list[str]): Bar and legend order.
        output (Path): Output stem for PDF and PNG.

    Returns:
        None: Saves both plot formats.
    """
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7, 3.1), layout="constrained")
    width = 0.75 / len(harnesses)
    for index, harness in enumerate(harnesses):
        values = [next(r["mean_episode_percent"] for r in summary
                       if r["harness"] == harness and r["metric"] == metric) for metric in METRICS]
        positions = [x + (index - (len(harnesses) - 1) / 2) * width for x in range(len(METRICS))]
        bars = ax.bar(positions, values, width, color=COLORS.get(harness, "#777777"), label=LABELS.get(harness, harness))
        ax.bar_label(bars, labels=[f"{value:.1f}".removesuffix(".0") for value in values], padding=3, fontsize=8)
    ax.set_xticks(range(len(METRICS)), METRICS.values())
    ax.set_ylabel("Episodes (%)")
    maximum = max(row["mean_episode_percent"] for row in summary)
    step = 5 if maximum < 25 else 20
    upper = max(step, math.ceil((maximum + 2) / step) * step)
    ax.set_ylim(0, upper)
    ax.set_yticks(range(0, min(upper, 100) + 1, step))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=len(harnesses))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the read-only trace analysis and export plots and audit data.

    Args:
        None.

    Returns:
        None: Writes analysis outputs under Q_2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPOSITORY)
    parser.add_argument("--model", default="qwen3.8-2.4t-a95b")
    parser.add_argument("--harnesses", nargs="+", default=["hermes", "openclaw"])
    parser.add_argument("--games", nargs="+", default=["wordle", "chronicle"])
    parser.add_argument("--output-dir", type=Path, default=QUESTION / "plots")
    args = parser.parse_args()
    episodes, evidence = collect_episodes(args.results_root.resolve(), args.model, args.harnesses, args.games)
    summary = summarize(episodes, args.harnesses, args.games)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "code_feedback_episodes.csv", episodes)
    write_csv(args.output_dir / "code_feedback_evidence.csv", evidence)
    write_csv(args.output_dir / "code_feedback_summary.csv", summary)
    by_game = [{"game": game, **row} for game in args.games
               for row in summarize([e for e in episodes if e["game"] == game], args.harnesses, [game])]
    write_csv(args.output_dir / "code_feedback_by_game.csv", by_game)
    plot_summary(summary, args.harnesses, args.output_dir / "code_feedback_by_harness")
    for row in summary:
        print(f"{row['harness']:12} {row['metric']:14} {row['episodes_with_event']:2}/{row['episodes']} episodes, "
              f"{row['event_count']:3} events, {row['mean_episode_percent']:.2f}%")
    print(f"Calls without captured results: {sum(e['calls_without_results'] for e in episodes)}")
    for episode in episodes:
        if not episode["capture_observed"]:
            print(f"Excluded paired instance due to missing capture: {episode['trace']}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
