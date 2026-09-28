"""Plot Wordle and Chronicle scores, completion, quality, and matched quality gains."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize import load_scores, summarize


MODELS = {"gemma4-e4b": "Gemma E4B",
          "nemotron-3.5-30B-A3-reasoning": "Nemotron 3.5\nLightning",
          "gpt-oss-120b": "GPT-OSS\n120B",
          "qwen3.8-27b-reasoning": "Qwen3.8\n27B",
          "glm-5.3-flash": "GLM-5.3\nFlash",
          "qwen3.8-2.4t-a95b": "Qwen3.8\n2.4T-A95B"}
CONDITIONS = {"": ("Vanilla", "#777777"),
              "codex": ("Codex", "#0072B2"),
              "claude-code": ("Claude Code", "#E69F00"),
              "hermes": ("Hermes", "#009E73"),
              "openclaw": ("OpenClaw", "#CC79A7")}


def matched_quality(scores: pd.DataFrame, games: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare harness and vanilla quality on jointly completed instances.

    Args:
        scores (pd.DataFrame): Episode metrics with normalized model names.
        games (list[str]): Games to compare with equal weight.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Per-game statistics and paired episode scores.
    """
    episodes = scores.pivot(index=["model", "game", "experiment", "episode"], columns="metric", values="value")
    episodes = episodes.apply(pd.to_numeric)
    summaries, pairs = [], []
    for model in MODELS:
        vanilla = episodes.loc[model]
        for harness in CONDITIONS:
            if not harness:
                continue
            config = f"{harness}-with-{model}"
            agent = episodes.loc[config]
            for game in games:
                played = agent.loc[game].query("Played == 1")
                comparison = played.join(vanilla.loc[game], lsuffix="_harness", rsuffix="_vanilla", how="left")
                matched = comparison.loc[comparison["Played_vanilla"] == 1].copy()
                if matched[["Main Score_harness", "Main Score_vanilla"]].isna().any().any():
                    raise ValueError(f"Missing quality for a completed matched episode: {config}/{game}")
                matched["quality_delta"] = matched["Main Score_harness"] - matched["Main Score_vanilla"]
                matched["configuration"] = config
                matched["game"] = game
                pairs.append(matched.reset_index())
                summaries.append({"configuration": config, "game": game, "harness_completed": len(played),
                                  "matched_instances": len(matched), "excluded_instances": len(played) - len(matched),
                                  "harness_quality": matched["Main Score_harness"].mean(),
                                  "vanilla_quality": matched["Main Score_vanilla"].mean(),
                                  "quality_delta": matched["quality_delta"].mean()})
    return pd.DataFrame(summaries), pd.concat(pairs, ignore_index=True)


def plot_scores(table: pd.DataFrame, output_dir: Path, metric: str, ylabel: str, suffix: str) -> None:
    """Save a grouped bar chart and the exact values plotted.

    Args:
        table (pd.DataFrame): Summary of existing Wordle and Chronicle scores.
        output_dir (Path): Destination for PDF, PNG, and CSV files.
        metric (str): Summary column to plot.
        ylabel (str): Metric label for the vertical axis.
        suffix (str): Filename suffix identifying the metric.

    Returns:
        None: Writes the figure and its underlying values.
    """
    if not table["games_included"].eq(2).all():
        raise ValueError("Every configuration must have results for both games")

    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                               "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    positions = np.arange(len(MODELS))
    width = 0.15
    plotted = pd.DataFrame(index=list(MODELS))

    conditions = {key: value for key, value in CONDITIONS.items() if key or metric != "quality_delta"}
    for index, (harness, (label, color)) in enumerate(conditions.items()):
        configs = [f"{harness}-with-{model}" if harness else model for model in MODELS]
        values = table.loc[configs, metric].to_numpy()
        bar_positions = positions + (index - (len(conditions) - 1) / 2) * width
        bars = ax.bar(bar_positions, values, width,
                      label=label, color=color, edgecolor="white", linewidth=0.4)
        labels = [f"{value:.1f}".removesuffix(".0") if pd.notna(value) else "" for value in values]
        ax.bar_label(bars, labels=labels, fontsize=7, padding=3, rotation=90)
        plotted[label] = values

    ax.set_xticks(positions, MODELS.values())
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    if metric == "quality_delta":
        ax.set_ylim(-100, 100)
        ax.set_yticks(np.arange(-100, 101, 20))
        ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(conditions), frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"overall_wordle_chronicle{suffix}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plotted.to_csv(stem.with_suffix(".csv"), index_label="model", float_format="%.6f")
    plt.close(fig)
    print(f"Saved {stem}.pdf, .png, and .csv")


def main() -> None:
    """Plot the two-game comparison without rescoring episodes.

    Args:
        None.

    Returns:
        None: Reads existing metrics and writes four grouped charts and matched comparisons.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_overall"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_overall/plots"))
    args = parser.parse_args()
    games = ["wordle", "chronicle"]
    scores = load_scores(args.results_dir, games)
    table = summarize(scores, games)
    matched, pairs = matched_quality(scores, games)
    table["quality_delta"] = matched.groupby("configuration")["quality_delta"].mean()
    metrics = [("mean_clemscore", "Mean Clemscore (Wordle and Chronicle)", ""),
               ("pct_played", "Mean % played (Wordle and Chronicle)", "_played"),
               ("quality", "Mean quality score (Wordle and Chronicle)", "_quality"),
               ("quality_delta", "Matched quality difference (harness − vanilla)", "_quality_delta")]
    for metric, ylabel, suffix in metrics:
        plot_scores(table, args.output_dir, metric, ylabel, suffix)
    matched.to_csv(args.output_dir / "matched_quality_by_game.csv", index=False, float_format="%.6f")
    pairs.to_csv(args.output_dir / "matched_quality_instances.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
