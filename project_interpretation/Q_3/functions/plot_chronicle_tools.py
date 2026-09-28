"""Plot Chronicle web-tool incidence alongside mean harness Clemscore."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_harness_scores import HARNESSES, MODELS, QUESTION, load_game


def load_data() -> pd.DataFrame:
    """Join observed web-tool incidence with existing Chronicle scores.

    Args:
        None.

    Returns:
        pd.DataFrame: Four harness rows with incidence and mean Clemscore.
    """
    tools = pd.read_csv(QUESTION / "plots/task_tool_summary.csv")
    tools = tools.loc[tools["game"] == "chronicle"].set_index("harness", verify_integrity=True)
    scores = load_game("chronicle")
    rows = []
    for harness in HARNESSES:
        configurations = scores.loc[[f"{harness}-with-{model}" for model in MODELS]]
        counts = tools.loc[harness]
        if not configurations["episodes"].eq(15).all() or counts["episodes"] != 90 or counts["captured"] != 90:
            raise ValueError(f"Unexpected Chronicle cohort or missing capture: {harness}")
        rows.append({"harness": harness, "episodes": int(counts["episodes"]),
                     "episodes_with_web_tools": int(counts["episodes_with_relevant_tools"]),
                     "web_tool_percent": 100 * counts["episodes_with_relevant_tools"] / counts["episodes"],
                     "mean_clemscore": configurations["clemscore"].mean()})
    return pd.DataFrame(rows).sort_values("web_tool_percent").reset_index(drop=True)


def plot_data(data: pd.DataFrame, output: Path) -> None:
    """Export a compact dual-axis chart with bars and a score line.

    Args:
        data (pd.DataFrame): Harness web-tool incidence and scores.
        output (Path): Output filename stem.

    Returns:
        None: Saves PDF and PNG figures.
    """
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                                "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, left = plt.subplots(figsize=(5.6, 3.2))
    right = left.twinx()
    positions = np.arange(len(data))
    colors = [HARNESSES[harness][1] for harness in data["harness"]]
    left.bar(positions, data["web_tool_percent"], width=0.58, color=colors,
             edgecolor="white", linewidth=0.4)
    right.plot(positions, data["mean_clemscore"], color="#222222", marker="o",
               linewidth=1.6, markersize=5)
    labels = [HARNESSES[harness][0].replace("Code SDK", "Code\nSDK") for harness in data["harness"]]
    left.set_xticks(positions, labels)
    left.set_ylabel("Episodes with web tools (%)")
    right.set_ylabel("Mean Clemscore")
    for axis in (left, right):
        axis.set_ylim(0, 100)
        axis.set_yticks(range(0, 101, 20))
        axis.spines["top"].set_visible(False)
    left.set_axisbelow(True)
    left.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the Chronicle-only plot and its source table.

    Args:
        None.

    Returns:
        None: Writes analysis artifacts without changing source results.
    """
    data = load_data()
    output = QUESTION / "plots/chronicle_web_tools_vs_clemscore"
    data.to_csv(output.with_suffix(".csv"), index=False)
    plot_data(data, output)
    print(data.to_string(index=False))
    print(f"Saved {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
