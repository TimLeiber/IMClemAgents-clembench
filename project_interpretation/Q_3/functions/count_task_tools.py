"""Count observed task-relevant tool use without changing episode results."""

import hashlib
import json
import re

import pandas as pd

from clemagents.adapters.codex.parse import parse_codex_agent_trace
from plot_harness_scores import GAMES, HARNESSES, MODELS, QUESTION, ROOT, VISUAL_MODELS


EXECUTION = {"bash", "exec", "terminal", "execute_code", "exec_command", "shell", "shell_command"}
WEB = {"web_search", "web_fetch", "web_extract", "websearch", "webfetch", "browser",
       "browser_navigate", "browser_snapshot", "browser_click", "browser_scroll",
       "browser_type", "browser_back", "browser_press", "browser_get_images", "browser_console", "search_query"}
VISION = {"view_image", "vision_analyze", "image", "image_analyze", "image_query"}
RELEVANT = {"wordle": {"execution"}, "chronicle": {"web"}, "geolocate": {"execution", "web", "vision"}}


def extract_calls(trace: dict) -> list[dict]:
    """Deduplicate observed tool calls across native events and request histories.

    Args:
        trace (dict): Normalized agent-loop trace.

    Returns:
        list[dict]: Unique call identities, names, and arguments.
    """
    calls = {}
    for event in trace["events"]:
        if event.get("source") == "hermes_cli":
            continue
        if event["type"] == "tool_call":
            identity = event.get("call_id")
            if not identity:
                raise ValueError("Cannot deduplicate a tool call without its identity")
            calls.setdefault(identity, {"call_id": identity, "tool": event["name"],
                                        "arguments": event.get("arguments", {})})
        elif event["type"] == "model_request":
            request = event.get("payload")
            if not request and event.get("raw"):
                request = json.loads(event["raw"])
            for message in (request or {}).get("messages", []):
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    calls.setdefault(call["id"], {"call_id": call["id"], "tool": function["name"],
                                                  "arguments": function.get("arguments", {})})
    return list(calls.values())


def categories(tool: str, arguments: object) -> set[str]:
    """Assign tool-interface categories without claiming that a call was useful.

    Args:
        tool (str): Captured tool name.
        arguments (object): Submitted arguments for identifying nested web calls.

    Returns:
        set[str]: Execution, web, or vision interface categories.
    """
    name = tool.rsplit(".", 1)[-1].lower()
    found = set()
    if name in EXECUTION:
        found.add("execution")
    if name in WEB or tool.lower() in {"web.run", "web__run"}:
        found.add("web")
    if name in VISION:
        found.add("vision")
    if name in EXECUTION and re.search(r"\b(?:web_search|web_fetch|web_extract|browser_navigate)\s*\(", str(arguments)):
        found.add("web")
    return found


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect episode incidence and individual calls from the evaluated cohort.

    Args:
        None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Episode summaries and call-level evidence.
    """
    episodes, evidence = [], []
    for game in GAMES:
        models = VISUAL_MODELS if game == "geolocate" else MODELS
        for harness in HARNESSES:
            for model in models:
                config = f"{harness}-with-{model}"
                folders = [p for p in (ROOT / f"results_{game}").iterdir()
                           if p.is_dir() and p.name.split("--")[-1] == config]
                assert len(folders) == 1, config
                paths = sorted((folders[0] / game).glob("*/instance_*/interactions.json"))
                assert len(paths) == 15, (game, config, len(paths))
                for interaction in paths:
                    path = interaction.with_name("agent_loop.json")
                    trace = json.loads(path.read_text()) if path.exists() else {"events": []}
                    calls = extract_calls(trace)
                    captured = bool(calls) or any(e["type"] in {"assistant_text", "reasoning", "model_response"}
                                                 for e in trace["events"])
                    source = path
                    if not captured and harness == "codex" and path.with_name("agent_trace.log").exists():
                        trace = parse_codex_agent_trace(path.parent)
                        calls = extract_calls(trace)
                        captured = bool(calls) or any(e["type"] in {"assistant_text", "reasoning", "model_response"}
                                                     for e in trace["events"])
                        source = path.with_name("agent_trace.log")
                    identity = {"game": game, "harness": harness, "model": model,
                                "experiment": path.parent.parent.name, "episode": path.parent.name}
                    counts = dict.fromkeys(["execution", "web", "vision", "auxiliary"], 0)
                    relevant = 0
                    for call in calls:
                        kinds = categories(call["tool"], call["arguments"])
                        relevant += bool(kinds & RELEVANT[game])
                        auxiliary = not any(name in call["tool"] for name in ("start_game", "submit_response"))
                        counts["auxiliary"] += auxiliary
                        for kind in kinds:
                            counts[kind] += 1
                        evidence.append({**identity, "call_id": call["call_id"], "tool": call["tool"],
                                         "categories": ",".join(sorted(kinds)), "auxiliary": auxiliary,
                                         "arguments_sha256": hashlib.sha256(str(call["arguments"]).encode()).hexdigest(),
                                         "trace": str(source.relative_to(ROOT))})
                    episodes.append({**identity, **counts, "captured": captured,
                                     "relevant_calls": relevant, "relevant_observed": relevant > 0,
                                     "trace": str(source.relative_to(ROOT))})
                    del trace
                print(f"Read {game}: {config}", flush=True)
    return pd.DataFrame(episodes), pd.DataFrame(evidence)


def main() -> None:
    """Export tool-use incidence and audit evidence for the RQ3 cohort.

    Args:
        None.

    Returns:
        None: Writes CSV tables and prints observed episode counts.
    """
    episodes, calls = collect()
    output = QUESTION / "plots"
    output.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(output / "task_tool_episodes.csv", index=False)
    calls.to_csv(output / "task_tool_calls.csv", index=False)
    keys = ["game", "harness"]
    summary = episodes.groupby(keys, sort=False).agg(episodes=("episode", "size"),
                                                    captured=("captured", "sum"),
                                                    episodes_with_relevant_tools=("relevant_observed", "sum"),
                                                    relevant_calls=("relevant_calls", "sum"))
    summary["observed_percent"] = 100 * summary["episodes_with_relevant_tools"] / summary["episodes"]
    summary.to_csv(output / "task_tool_summary.csv")
    per_model = episodes.groupby(keys + ["model"], sort=False)["relevant_observed"].mean() * 100
    per_model.rename("observed_percent").to_csv(output / "task_tool_by_model.csv")
    print(summary.to_string())
    print(calls.groupby(["harness", "tool"]).size().to_string())


if __name__ == "__main__":
    main()
