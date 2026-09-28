"""Compare mean harness Clemscores by game using existing score exports."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
QUESTION = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "results_geolocate_convergence_replay/timeout_recovery_remaining_guesses_01/results.csv"
MODELS = ("gemma4-e4b", "nemotron-3.5-30B-A3-reasoning", "gpt-oss-120b",
          "qwen3.8-27b-reasoning", "glm-5.3-flash", "qwen3.8-2.4t-a95b")
VISUAL_MODELS = ("qwen3.8-27b-reasoning", "glm-5.3-flash")
HARNESSES = {"codex": ("Codex CLI", "#0072B2"),
             "claude-code": ("Claude Code SDK", "#E69F00"),
             "hermes": ("Hermes CLI", "#009E73"),
             "openclaw": ("OpenClaw", "#CC79A7")}
GAMES = ("wordle", "chronicle", "geolocate")


def load_game(game: str) -> pd.DataFrame:
    """Aggregate existing text-game episode scores without invoking scorers.

    Args:
        game (str): Text game to load.

    Returns:
        pd.DataFrame: Clemscores and episode counts indexed by configuration.
    """
    frame = pd.read_csv(ROOT / f"results_{game}/raw.csv")
    frame = frame[frame["metric"].isin(["Played", "Main Score"])].copy()
    frame["model"] = frame["model"].str.removeprefix("Qwen3.6-35B-A3B--")
    episodes = frame.pivot(index=["model", "experiment", "episode"], columns="metric", values="value")
    episodes = episodes.apply(pd.to_numeric)
    if not episodes["Played"].isin([0, 1]).all():
        raise ValueError(f"Invalid completion flags in {game}")
    if episodes.loc[episodes["Played"] == 1, "Main Score"].isna().any():
        raise ValueError(f"Missing quality scores in {game}")

    # zero for aborted episodes yields completion fraction times completed quality
    episodes["clemscore"] = episodes["Main Score"].where(episodes["Played"] == 1, 0)
    return episodes.groupby(level="model").agg(clemscore=("clemscore", "mean"), episodes=("Played", "size"))


def load_comparison(recovery: Path) -> pd.DataFrame:
    """Load the full configuration cohort with Geolocate recovery substituted.

    Args:
        recovery (Path): Remaining-guesses recovery summary CSV.

    Returns:
        pd.DataFrame: Auditable per-model, per-harness game scores.
    """
    tables = {game: load_game(game) for game in GAMES[:2]}
    recovered = pd.read_csv(recovery).set_index("Configuration", verify_integrity=True)
    tables["geolocate"] = recovered.rename(columns={"Recovered clemscore": "clemscore", "Episodes": "episodes"})
    rows = []
    for game, table in tables.items():
        models = VISUAL_MODELS if game == "geolocate" else MODELS
        for harness in HARNESSES:
            for model in models:
                configuration = f"{harness}-with-{model}"
                score = table.loc[configuration, "clemscore"]
                episodes = table.loc[configuration, "episodes"]
                if not np.isfinite(score) or not 0 <= score <= 100 or episodes != 15:
                    raise ValueError(f"Unexpected score or episode count: {game}, {configuration}")
                rows.append({"game": game, "harness": harness, "model": model,
                             "clemscore": score, "episodes": int(episodes)})
    return pd.DataFrame(rows)


def plot_scores(means: pd.DataFrame, output: Path) -> None:
    """Save a compact grouped bar chart without explanatory annotations.

    Args:
        means (pd.DataFrame): Harness means indexed by game and harness.
        output (Path): Output filename stem.

    Returns:
        None: Saves PDF and PNG figures.
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    positions = np.arange(len(GAMES))
    width = 0.19
    for index, (harness, (label, color)) in enumerate(HARNESSES.items()):
        values = [means.loc[(game, harness), "mean_clemscore"] for game in GAMES]
        bars = ax.bar(positions + (index - 1.5) * width, values, width=width, label=label,
                      color=color, edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, labels=[f"{value:.1f}".removesuffix(".0") for value in values],
                     padding=3, fontsize=8)
    ax.set_xticks(positions, ["Wordle", "Chronicle", "Geolocate"])
    ax.set_ylabel("Mean Clemscore")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Export the RQ3 comparison and its underlying configuration scores.

    Args:
        None.

    Returns:
        None: Writes analysis outputs and prints mean scores.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-results", type=Path, default=RECOVERY)
    parser.add_argument("--output-dir", type=Path, default=QUESTION / "plots")
    args = parser.parse_args()
    scores = load_comparison(args.recovery_results)
    means = scores.groupby(["game", "harness"], sort=False).agg(mean_clemscore=("clemscore", "mean"),
                                                             models=("model", "nunique"),
                                                             episodes=("episodes", "sum"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "harness_scores_by_game"
    scores.to_csv(args.output_dir / "configuration_scores.csv", index=False)
    means.to_csv(output.with_suffix(".csv"))
    plot_scores(means, output)
    print(means["mean_clemscore"].unstack("game").reindex(index=HARNESSES, columns=GAMES).round(2))
    print(f"Saved PDF, PNG and CSV outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
