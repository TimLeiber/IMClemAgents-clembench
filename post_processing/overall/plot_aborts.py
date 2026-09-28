"""Compare vanilla abort rates with the average harness failure breakdown."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_overall import CONDITIONS, MODELS
from summarize import NARRATOR_PREFIX


GAMES = ("wordle", "chronicle")
CATEGORIES = {"game_abort": ("Game-level abort", "#0072B2"),
              "agent_abort": ("Agent-control abort", "#E69F00"),
              "timeout": ("Timeout", "#CC6677")}


def classify_abort(episode: Path) -> str:
    """Classify a harness abort with timeout precedence over format errors.

    Args:
        episode (Path): Episode directory containing failure metadata.

    Returns:
        str: Mutually exclusive failure category.
    """
    metadata = json.loads((episode / "agent_trace_meta.json").read_text())
    interactions = json.loads((episode / "interactions.json").read_text())
    responses = [event.get("action", {}).get("content", "")
                 for turn in interactions.get("turns", []) for event in turn
                 if event.get("action", {}).get("type") == "get message"]
    reason = metadata.get("episode_terminal_reason")
    if metadata.get("episode_timed_out") or reason == "episode_timeout":
        return "timeout"
    if any(response.startswith("AGENT_EPISODE_TIMEOUT:") for response in responses):
        return "timeout"
    if reason == "agent_exit_before_game_done":
        return "agent_abort"
    if any(response.startswith(("AGENT_CONTROL_ERROR:", "CLEM_AGENT_CONTROL_ERROR:")) for response in responses):
        return "agent_abort"
    if reason == "game_done":
        return "game_abort"
    raise ValueError(f"Unclassified abort at {episode}: {reason}")


def load_episodes(results_dir: Path) -> pd.DataFrame:
    """Read official completion metrics and classify their aborted episodes.

    Args:
        results_dir (Path): Parent of the per-game result directories.

    Returns:
        pd.DataFrame: Episode identities, execution conditions, and outcomes.
    """
    configurations = {f"{harness}-with-{model}" if harness else model: (model, harness)
                      for model in MODELS for harness in CONDITIONS}
    records = []
    for game in GAMES:
        root = results_dir / f"results_{game}"
        raw = pd.read_csv(root / "raw.csv")
        played = raw.loc[raw["metric"] == "Played"]
        if played.duplicated(["model", "game", "experiment", "episode"]).any():
            raise ValueError(f"Duplicate completion metrics in {root}")
        for row in played.to_dict("records"):
            config = row["model"].removeprefix(NARRATOR_PREFIX)
            model, harness = configurations[config]
            episode = root / row["model"] / game / row["experiment"] / row["episode"]
            if row["value"] not in (0, 1):
                raise ValueError(f"Invalid completion metric at {episode}")
            category = "completed"
            if row["value"] == 0:
                category = classify_abort(episode) if harness else "game_abort"
            records.append({"model": model, "harness": harness, "game": game,
                            "experiment": row["experiment"], "episode": row["episode"], "category": category})
    return pd.DataFrame(records)


def aggregate_rates(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average failure rates equally across games and harnesses.

    Args:
        episodes (pd.DataFrame): Classified episodes including completed games.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Per-configuration rates and plotted averages.
    """
    counts = pd.crosstab([episodes["model"], episodes["harness"], episodes["game"]], episodes["category"])
    rates = counts.reindex(columns=list(CATEGORIES), fill_value=0).div(counts.sum(axis=1), axis=0) * 100
    rates = rates.reset_index()
    for model in MODELS:
        observed = set(rates.loc[rates["model"] == model, ["harness", "game"]].itertuples(index=False, name=None))
        if observed != {(harness, game) for harness in CONDITIONS for game in GAMES}:
            raise ValueError(f"Incomplete configuration coverage for {model}")

    # each model contributes two vanilla rates and eight harness rates
    vanilla = rates.loc[rates["harness"] == ""].groupby("model")["game_abort"].mean()
    harness = rates.loc[rates["harness"] != ""].groupby("model")[list(CATEGORIES)].mean()
    table = harness.reindex(MODELS).copy()
    table.insert(0, "vanilla_abort", vanilla)
    table["harness_total_abort"] = table[list(CATEGORIES)].sum(axis=1)
    return rates, table


def plot_aborts(table: pd.DataFrame, output_dir: Path) -> None:
    """Save a compact paired bar chart without embedded explanatory notes.

    Args:
        table (pd.DataFrame): Average vanilla and harness failure rates.
        output_dir (Path): Plot destination.

    Returns:
        None: Writes PDF and PNG figures.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 3), layout="constrained")
    positions = np.arange(len(MODELS))
    width = 0.3
    ax.bar(positions - width / 2, table["vanilla_abort"], width, color="#777777", label="Vanilla")
    bottom = np.zeros(len(MODELS))
    for category, (label, color) in CATEGORIES.items():
        values = table[category].to_numpy()
        ax.bar(positions + width / 2, values, width, bottom=bottom, color=color, label=label)
        bottom += values
    ax.set_xticks(positions, MODELS.values())
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Abort rate (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False)
    stem = output_dir / "overall_wordle_chronicle_aborts"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Export the abort chart and auditable episode classifications.

    Args:
        None.

    Returns:
        None: Writes figures and CSV breakdowns without changing results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_overall"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_overall/plots"))
    args = parser.parse_args()
    episodes = load_episodes(args.results_dir)
    rates, table = aggregate_rates(episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output_dir / "abort_classifications.csv", index=False)
    rates.to_csv(args.output_dir / "abort_rates_by_configuration.csv", index=False, float_format="%.6f")
    table.to_csv(args.output_dir / "overall_wordle_chronicle_aborts.csv", float_format="%.6f")
    plot_aborts(table, args.output_dir)
    print(table.round(2).to_string())
    print(f"Saved abort figures and CSV breakdowns to {args.output_dir}")


if __name__ == "__main__":
    main()
