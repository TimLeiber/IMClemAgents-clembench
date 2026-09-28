"""Combine existing game scores without running scorers or changing source results."""

import argparse
from pathlib import Path

import pandas as pd


GAMES = ("wordle", "chronicle", "geolocate")
NARRATOR_PREFIX = "Qwen3.6-35B-A3B--"


def load_scores(results_dir: Path, games: list[str]) -> pd.DataFrame:
    """Load existing episode metrics from the selected game exports.

    Args:
        results_dir (Path): Directory containing the three game result folders.
        games (list[str]): Games to include.

    Returns:
        pd.DataFrame: Episode metrics with normalized configuration names.
    """
    frames = []
    for game in games:
        path = results_dir / f"results_{game}" / "raw.csv"
        frame = pd.read_csv(path)
        if set(frame["game"]) != {game}:
            raise ValueError(f"Unexpected game names in {path}")

        frame = frame[frame["metric"].isin(["Played", "Main Score"])].copy()
        if game == "chronicle":
            if not frame["model"].str.startswith(NARRATOR_PREFIX).all():
                raise ValueError(f"Unexpected Chronicle narrator in {path}")
            frame["model"] = frame["model"].str.removeprefix(NARRATOR_PREFIX)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def summarize(scores: pd.DataFrame, games: list[str]) -> pd.DataFrame:
    """Average per-game Clemscores over games with recorded episodes only.

    Args:
        scores (pd.DataFrame): Normalized existing episode metrics.
        games (list[str]): Ordered game columns to include.

    Returns:
        pd.DataFrame: Configuration scores, game coverage, and episode counts.
    """
    keys = ["model", "game", "experiment", "episode"]
    if scores.duplicated(keys + ["metric"]).any():
        raise ValueError("Duplicate episode metrics found")

    episodes = scores.pivot(index=keys, columns="metric", values="value")
    episodes = episodes.reindex(columns=["Played", "Main Score"]).apply(pd.to_numeric)
    if not episodes["Played"].isin([0, 1]).all():
        raise ValueError("Every recorded episode must have Played equal to zero or one")
    if episodes.loc[episodes["Played"] == 1, "Main Score"].isna().any():
        raise ValueError("A played episode is missing its Main Score")

    # aborted episodes affect completion rates but not conditional quality
    episodes.loc[episodes["Played"] == 0, "Main Score"] = float("nan")
    per_game = episodes.groupby(level=["model", "game"]).agg(
        episodes=("Played", "size"), played=("Played", "sum"),
        played_fraction=("Played", "mean"), quality=("Main Score", "mean"))
    per_game["clemscore"] = per_game["played_fraction"] * per_game["quality"]
    per_game.loc[per_game["played"] == 0, "clemscore"] = 0.0

    # missing games remain empty and are excluded from the macro average
    table = per_game["clemscore"].unstack("game").reindex(columns=games)
    table["mean_clemscore"] = table[games].mean(axis=1)
    table["pct_played"] = per_game["played_fraction"].groupby(level="model").mean() * 100
    table["quality"] = per_game["quality"].groupby(level="model").mean()
    table["games_included"] = table[games].notna().sum(axis=1)
    table["episodes_recorded"] = per_game["episodes"].groupby(level="model").sum()
    table.index.name = "configuration"
    table.columns.name = None
    return table.sort_values("mean_clemscore", ascending=False, kind="stable")


def main() -> None:
    """Write a combined table from existing scores.

    Args:
        None.

    Returns:
        None: Writes CSV and HTML tables without modifying source results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results_overall"))
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument("--output-name", default="overall_summary")
    args = parser.parse_args()
    if len(set(args.games)) != len(args.games):
        parser.error("Each game may only be selected once")
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a filename stem, not a path")

    table = summarize(load_scores(args.results_dir, args.games), args.games)
    output = args.results_dir / args.output_name
    table.to_csv(output.with_suffix(".csv"), float_format="%.4f")
    table.to_html(output.with_suffix(".html"), float_format=lambda value: f"{value:.2f}", na_rep="—")
    print(f"Wrote {len(table)} configurations to {output}.csv and {output}.html")
    print("Mean Clemscore gives equal weight to each recorded game, excluding absent games")
    print("All-aborted games count as zero; source scores and results are unchanged")


if __name__ == "__main__":
    main()
