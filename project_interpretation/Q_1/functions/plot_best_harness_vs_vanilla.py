"""Plot the best harness-supported result against vanilla for each model.

The comparison uses the simple arithmetic mean of the clemscores from
Chronicle and Wordle. For the harness-supported bar, the best harness is
selected independently on each game before those two winning scores are
averaged. The two games may therefore contribute results from different
harnesses. Optionally include Geolocate, restricting the comparison to models
evaluated on every selected game. Missing game results never count as zero.

The implementation intentionally uses Pillow rather than matplotlib so that
it runs in the existing clembench environment without another plotting
dependency.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


REPOSITORY_DIR = Path(__file__).resolve().parents[3]
QUESTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERY_RESULTS = (REPOSITORY_DIR / "results_geolocate_convergence_replay" /
                            "timeout_recovery_remaining_guesses_01" / "results.csv")

DEFAULT_MODEL_ORDER = (
    "gemma4-e4b",
    "nemotron-3.5-30B-A3-reasoning",
    "gpt-oss-120b",
    "qwen3.8-27b-reasoning",
    "glm-5.3-flash",
    "qwen3.8-2.4t-a95b",
)

DEFAULT_MODEL_LABELS = {
    "gemma4-e4b": "Gemma E4B",
    "nemotron-3.5-30B-A3-reasoning": "Nemotron 3.5\nLightning",
    "gpt-oss-120b": "GPT-OSS 120B",
    "qwen3.8-27b-reasoning": "Qwen 3.8 27B",
    "glm-5.3-flash": "GLM 5.3 Flash",
    "qwen3.8-2.4t-a95b": "Qwen 3.8\n2.4T-A95B",
}

DEFAULT_HARNESSES = ("codex", "claude-code", "hermes", "openclaw")
HARNESS_LABELS = {
    "codex": "Codex",
    "claude-code": "Claude Code",
    "hermes": "Hermes",
    "openclaw": "OpenClaw",
}


@dataclass(frozen=True)
class ModelComparison:
    """The two plotted values for one model."""

    model: str
    vanilla_mean: float
    best_harness_mean: float
    best_wordle_harness: str
    best_chronicle_harness: str
    best_geolocate_harness: str | None = None


def _float_or_none(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _strip_environment_model(config_name: str) -> str:
    """Remove Chronicle's narrator prefix from an evaluated configuration."""

    if "--" in config_name:
        return config_name.split("--", maxsplit=1)[1]
    return config_name


def _load_clemscores(results_csv: Path) -> dict[str, float]:
    """Read configuration-level clemscores from a ``clem eval`` CSV file.

    ``clem eval`` leaves clemscore blank when a configuration played zero
    percent of its episodes.  Such an observed configuration contributes zero
    here; a completely absent configuration remains missing and is rejected by
    :func:`build_comparison_data`.
    """

    with results_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {results_csv}")

        # recovery tables include the full cohort and its unchanged vanilla reference
        if "Recovered clemscore" in reader.fieldnames:
            scores = {}
            for row in reader:
                config = row["Configuration"]
                model = config.split("-with-", 1)[1]
                vanilla = float(row["Vanilla clemscore"])
                if model in scores and not math.isclose(scores[model], vanilla):
                    raise ValueError(f"Inconsistent vanilla reference for {model}")
                scores[model] = vanilla
                scores[config] = float(row["Recovered clemscore"])
            return scores

        name_column = reader.fieldnames[0]
        clemscore_column = next(
            (column for column in reader.fieldnames if column.lower().endswith("clemscore")),
            None,
        )
        played_column = next(
            (column for column in reader.fieldnames if column.endswith("Average % Played")),
            None,
        )
        if clemscore_column is None or played_column is None:
            raise ValueError(
                f"Could not identify clemscore and played columns in {results_csv}"
            )

        scores: dict[str, float] = {}
        for row in reader:
            config_name = _strip_environment_model(row[name_column].strip())
            score = _float_or_none(row[clemscore_column])
            played = _float_or_none(row[played_column])
            if score is None and played == 0.0:
                score = 0.0
            if score is not None:
                scores[config_name] = score
        return scores


def build_comparison_data(
    wordle_results_csv: Path,
    chronicle_results_csv: Path,
    *,
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    harnesses: Sequence[str] = DEFAULT_HARNESSES,
    geolocate_results_csv: Path | None = None,
    geolocate_only: bool = False,
) -> list[ModelComparison]:
    """Average selected games equally, using the common vanilla model cohort."""

    if geolocate_only and geolocate_results_csv is None:
        raise ValueError("Geolocate-only comparisons require a Geolocate results table")
    game_scores = {} if geolocate_only else {
        "wordle": _load_clemscores(wordle_results_csv),
        "chronicle": _load_clemscores(chronicle_results_csv),
    }
    if geolocate_results_csv is not None:
        game_scores["geolocate"] = _load_clemscores(Path(geolocate_results_csv))
        model_order = [model for model in model_order if model in game_scores["geolocate"]]
    if not model_order:
        raise ValueError("No models are shared by the selected game tables")
    comparisons: list[ModelComparison] = []

    for model in model_order:
        missing_vanilla = [
            game for game, scores in game_scores.items() if model not in scores
        ]
        if missing_vanilla:
            raise ValueError(
                f"Missing vanilla score for {model}: {', '.join(missing_vanilla)}"
            )

        vanilla_mean = sum(scores[model] for scores in game_scores.values()) / len(game_scores)
        best_by_game: dict[str, tuple[str, float]] = {}
        for game, scores in game_scores.items():
            available = {
                harness: scores[f"{harness}-with-{model}"]
                for harness in harnesses
                if f"{harness}-with-{model}" in scores
            }
            missing = [harness for harness in harnesses if harness not in available]
            if missing:
                raise ValueError(f"Missing harness scores for {model} on {game}: {', '.join(missing)}")
            best_harness = max(
                available,
                key=lambda harness: (available[harness], -harnesses.index(harness)),
            )
            best_by_game[game] = (best_harness, available[best_harness])

        best_harness_mean = sum(score for _, score in best_by_game.values()) / len(game_scores)
        comparisons.append(
            ModelComparison(
                model=model,
                vanilla_mean=vanilla_mean,
                best_harness_mean=best_harness_mean,
                best_wordle_harness=best_by_game.get("wordle", ("", 0))[0],
                best_chronicle_harness=best_by_game.get("chronicle", ("", 0))[0],
                best_geolocate_harness=best_by_game["geolocate"][0] if "geolocate" in best_by_game else None,
            )
        )

    return comparisons


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf") if bold else
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if bold else
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _centered_multiline_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    top_y: float,
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
    spacing: int = 5,
) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    width = box[2] - box[0]
    draw.multiline_text(
        (center_x - width / 2, top_y),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def plot_best_harness_vs_vanilla(
    *,
    wordle_results_csv: Path | str = REPOSITORY_DIR / "results_wordle" / "results.csv",
    chronicle_results_csv: Path | str = REPOSITORY_DIR / "results_chronicle" / "results.csv",
    output_path: Path | str = QUESTION_DIR / "plots" / "best_harness_vs_vanilla.png",
    model_order: Sequence[str] = DEFAULT_MODEL_ORDER,
    model_labels: Mapping[str, str] = DEFAULT_MODEL_LABELS,
    harnesses: Sequence[str] = DEFAULT_HARNESSES,
    y_max: float = 100.0,
    geolocate_results_csv: Path | str | None = None,
    geolocate_only: bool = False,
) -> Path:
    """Write the grouped capability-threshold chart and return its path.

    The x axis is deliberately categorical. ``model_order`` should therefore
    encode the ordering justified in the report (for example, approximate
    deployment scale or expected capability); horizontal spacing must not be
    interpreted as a numerical difference in parameter count.
    """

    wordle_results_csv = Path(wordle_results_csv)
    chronicle_results_csv = Path(chronicle_results_csv)
    output_path = Path(output_path)
    if y_max <= 0:
        raise ValueError("y_max must be positive")

    comparisons = build_comparison_data(
        wordle_results_csv,
        chronicle_results_csv,
        model_order=model_order,
        harnesses=harnesses,
        geolocate_results_csv=geolocate_results_csv,
        geolocate_only=geolocate_only,
    )

    width, height = (1100, 800) if geolocate_only else (1900, 1080)
    left, right, top, bottom = 170, 70, 70, 170
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    colors = {
        "background": "#FFFFFF",
        "text": "#172033",
        "muted": "#5C667A",
        "grid": "#D9DEE8",
        "axis": "#677287",
        "harness": "#2C6EBA",
        "vanilla": "#E28E2C",
    }
    image = Image.new("RGB", (width, height), colors["background"])
    draw = ImageDraw.Draw(image)

    axis_font = _load_font(26)
    tick_font = _load_font(22)
    label_font = _load_font(24, bold=True)
    value_font = _load_font(22, bold=True)

    tick_step = 20 if y_max >= 80 else max(5, int(math.ceil(y_max / 5 / 5) * 5))
    for tick in range(0, int(y_max) + 1, tick_step):
        y = plot_bottom - (tick / y_max) * plot_height
        draw.line((plot_left, y, plot_right, y), fill=colors["grid"], width=2)
        tick_text = str(tick)
        tick_box = draw.textbbox((0, 0), tick_text, font=tick_font)
        draw.text(
            (plot_left - 22 - (tick_box[2] - tick_box[0]), y - 13),
            tick_text,
            font=tick_font,
            fill=colors["muted"],
        )

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=colors["axis"], width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=colors["axis"], width=3)

    group_width = plot_width / len(comparisons)
    bar_width = min(86, group_width * 0.27)
    gap = max(10, group_width * 0.035)

    for index, comparison in enumerate(comparisons):
        center_x = plot_left + group_width * (index + 0.5)
        bars = (
            (comparison.vanilla_mean, center_x - bar_width - gap / 2, colors["vanilla"]),
            (comparison.best_harness_mean, center_x + gap / 2, colors["harness"]),
        )
        for value, x0, color in bars:
            bounded_value = min(max(value, 0), y_max)
            y0 = plot_bottom - (bounded_value / y_max) * plot_height
            draw.rectangle((x0, y0, x0 + bar_width, plot_bottom), fill=color)
            value_text = f"{value:.1f}"
            value_box = draw.textbbox((0, 0), value_text, font=value_font)
            draw.text(
                (x0 + (bar_width - (value_box[2] - value_box[0])) / 2, y0 - 34),
                value_text,
                font=value_font,
                fill=colors["text"],
            )

        _centered_multiline_text(
            draw,
            center_x,
            plot_bottom + 24,
            model_labels.get(comparison.model, comparison.model),
            font=label_font,
            fill=colors["text"],
        )

    # center the vertical axis label within the plotting area
    y_title = Image.new("RGBA", (plot_height, 50), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_title)
    y_text = "Clemscore" if geolocate_only else "Average clemscore"
    y_text_width = y_draw.textlength(y_text, font=axis_font)
    y_draw.text(
        ((plot_height - y_text_width) / 2, 4),
        y_text,
        font=axis_font,
        fill=colors["text"],
    )
    y_title = y_title.rotate(90, expand=True)
    image.paste(y_title, (28, int(plot_top + (plot_height - y_title.height) / 2)), y_title)

    legend_y = height - 66
    legend_items = (
        (colors["vanilla"], "Vanilla model"),
        (colors["harness"], "Best harness" if geolocate_only else "Best available harness per game"),
    )
    legend_widths = []
    for _, text in legend_items:
        box = draw.textbbox((0, 0), text, font=axis_font)
        legend_widths.append(34 + 12 + box[2] - box[0])
    total_legend_width = sum(legend_widths) + 70
    legend_x = (width - total_legend_width) / 2
    for (color, text), item_width in zip(legend_items, legend_widths):
        draw.rectangle((legend_x, legend_y, legend_x + 34, legend_y + 24), fill=color)
        draw.text(
            (legend_x + 46, legend_y - 4),
            text,
            font=axis_font,
            fill=colors["text"],
        )
        legend_x += item_width + 70

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("output_path must end in .png or .pdf")
    image.save(output_path, dpi=(180, 180))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geolocate-results", type=Path, nargs="?", const=DEFAULT_RECOVERY_RESULTS,
                        help="include Geolocate; defaults to remaining-guesses recovery when no path is supplied")
    parser.add_argument("--geolocate-only", action="store_true",
                        help="plot only Geolocate, using remaining-guesses recovery unless another table is supplied")
    parser.add_argument(
        "--wordle-results",
        type=Path,
        default=REPOSITORY_DIR / "results_wordle" / "results.csv",
    )
    parser.add_argument(
        "--chronicle-results",
        type=Path,
        default=REPOSITORY_DIR / "results_chronicle" / "results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=QUESTION_DIR / "plots" / "best_harness_vs_vanilla.png",
    )
    args = parser.parse_args()
    if args.geolocate_only and args.geolocate_results is None:
        args.geolocate_results = DEFAULT_RECOVERY_RESULTS
    output = plot_best_harness_vs_vanilla(
        wordle_results_csv=args.wordle_results,
        chronicle_results_csv=args.chronicle_results,
        output_path=args.output,
        geolocate_results_csv=args.geolocate_results,
        geolocate_only=args.geolocate_only,
    )
    print(output)


if __name__ == "__main__":
    main()
