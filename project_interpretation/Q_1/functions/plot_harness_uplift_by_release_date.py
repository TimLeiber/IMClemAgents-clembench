"""Plot harness uplift against model release date for research question 1.

Harness uplift is the difference between the best harness-supported score and
the vanilla score. The best harness is selected independently for Chronicle
and Wordle before the two game scores are averaged.

Release dates use one continuous, proportional calendar axis. The regression
therefore represents elapsed time without categorical spacing or axis breaks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw

from plot_best_harness_vs_vanilla import (
    DEFAULT_MODEL_LABELS,
    REPOSITORY_DIR,
    QUESTION_DIR,
    _load_font,
    build_comparison_data,
)


# OpenRouter release dates, except that the Gemma date is the shared Gemma 4
# family release date used by the Google/Hugging Face release materials.
# Dates are deliberately stored as ISO YYYY-MM-DD strings for easy auditing.
MODEL_RELEASE_DATES = {
    "gemma4-e4b-mlx": "2026-04-03",
    "nemotron-3.5-30B-A3-reasoning": "2026-08-11",
    "gpt-oss-120b": "2025-08-05",
    "qwen3.8-27b-reasoning": "2026-08-14",
    "glm-5.3-flash": "2026-08-26",
    "qwen3.8-2.4t-a95b": "2026-08-12",
}

RELEASE_DATE_SOURCES = {
    "gemma4-e4b-mlx": "https://huggingface.co/blog/gemma4",
    "nemotron-3.5-30B-A3-reasoning": "https://openrouter.ai/nvidia/nemotron-3.5-lightning",
    "gpt-oss-120b": "https://openrouter.ai/openai/gpt-oss-120b",
    "qwen3.8-27b-reasoning": "https://openrouter.ai/qwen/qwen3.8-27b",
    "glm-5.3-flash": "https://openrouter.ai/z-ai/glm-5.3-flash",
    "qwen3.8-2.4t-a95b": "https://openrouter.ai/qwen/qwen3.8-2.4t-a95b-20260812",
}


@dataclass(frozen=True)
class UpliftObservation:
    model: str
    release_date: date
    uplift: float


def build_uplift_data(
    wordle_results_csv: Path,
    chronicle_results_csv: Path,
    *,
    release_dates: Mapping[str, str] = MODEL_RELEASE_DATES,
) -> list[UpliftObservation]:
    """Build chronologically ordered observations from the two eval tables."""

    comparisons = build_comparison_data(
        wordle_results_csv,
        chronicle_results_csv,
        model_order=tuple(release_dates),
    )
    observations = [
        UpliftObservation(
            model=comparison.model,
            release_date=date.fromisoformat(release_dates[comparison.model]),
            uplift=comparison.best_harness_mean - comparison.vanilla_mean,
        )
        for comparison in comparisons
    ]
    return sorted(observations, key=lambda observation: observation.release_date)


def _persistent_positive_crossover(
    observations: list[UpliftObservation],
) -> tuple[int, int] | None:
    """Return indices bracketing the first persistent negative-to-positive shift."""

    for right_index in range(1, len(observations)):
        before = observations[:right_index]
        after = observations[right_index:]
        if (
            any(observation.uplift < 0 for observation in before)
            and observations[right_index - 1].uplift < 0
            and all(observation.uplift >= 0 for observation in after)
        ):
            return right_index - 1, right_index
    return None


def _least_squares_release_date(
    observations: list[UpliftObservation],
) -> tuple[float, float, float]:
    """Return intercept, daily slope, and R-squared for the minimum-MSE line.

    The independent variable is elapsed calendar days since the earliest model
    release, so the fit matches the temporal meaning of the x-axis.
    """

    first_ordinal = min(observation.release_date.toordinal() for observation in observations)
    x_values = [
        observation.release_date.toordinal() - first_ordinal
        for observation in observations
    ]
    y_values = [observation.uplift for observation in observations]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    if denominator == 0:
        raise ValueError("At least two distinct release dates are required")
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * x_value for x_value in x_values]
    residual_sum = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(y_values, predictions)
    )
    total_sum = sum((actual - y_mean) ** 2 for actual in y_values)
    r_squared = 1 - residual_sum / total_sum if total_sum else 1.0
    return intercept, slope, r_squared


def plot_harness_uplift_by_release_date(
    *,
    wordle_results_csv: Path | str = REPOSITORY_DIR / "results_wordle" / "results.csv",
    chronicle_results_csv: Path | str = REPOSITORY_DIR / "results_chronicle" / "results.csv",
    output_path: Path | str = QUESTION_DIR / "plots" / "harness_uplift_by_release_date.png",
    release_dates: Mapping[str, str] = MODEL_RELEASE_DATES,
) -> Path:
    """Write the RQ1 release-date/uplift plot and return its output path."""

    output_path = Path(output_path)
    observations = build_uplift_data(
        Path(wordle_results_csv),
        Path(chronicle_results_csv),
        release_dates=release_dates,
    )
    if len(observations) < 2:
        raise ValueError("At least two model observations are required")

    width, height = 2100, 1120
    left, right, top, bottom = 190, 180, 100, 250
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
        "positive": "#2C6EBA",
        "negative": "#E28E2C",
        "cutoff": "#8C3E73",
        "regression": "#268477",
    }
    image = Image.new("RGB", (width, height), colors["background"])
    draw = ImageDraw.Draw(image)

    axis_font = _load_font(26)
    tick_font = _load_font(22)
    label_font = _load_font(23, bold=True)
    date_font = _load_font(20)
    value_font = _load_font(22, bold=True)
    annotation_font = _load_font(20, bold=True)

    max_absolute = max(abs(observation.uplift) for observation in observations)
    y_limit = max(30, int((max_absolute + 9.999) // 10) * 10)
    y_min, y_max = -float(y_limit), float(y_limit)

    def y_position(value: float) -> float:
        return plot_bottom - ((value - y_min) / (y_max - y_min)) * plot_height

    ordinals = [observation.release_date.toordinal() for observation in observations]
    first_ordinal, last_ordinal = ordinals[0], ordinals[-1]

    def x_position(ordinal: float) -> float:
        fraction = (ordinal - first_ordinal) / (last_ordinal - first_ordinal)
        return plot_left + fraction * plot_width

    for tick in range(-y_limit, y_limit + 1, 10):
        y = y_position(tick)
        is_zero = tick == 0
        draw.line(
            (plot_left, y, plot_right, y),
            fill=colors["axis"] if is_zero else colors["grid"],
            width=4 if is_zero else 2,
        )
        tick_text = str(tick)
        box = draw.textbbox((0, 0), tick_text, font=tick_font)
        draw.text(
            (plot_left - 22 - (box[2] - box[0]), y - 13),
            tick_text,
            font=tick_font,
            fill=colors["muted"],
        )

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=colors["axis"], width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=colors["axis"], width=3)

    coordinates = [
        (
            x_position(observation.release_date.toordinal()),
            y_position(observation.uplift),
        )
        for observation in observations
    ]
    intercept, slope, r_squared = _least_squares_release_date(observations)
    def regression_y(ordinal: int) -> float:
        return y_position(intercept + slope * (ordinal - first_ordinal))

    draw.line(
        (
            plot_left,
            regression_y(first_ordinal),
            plot_right,
            regression_y(last_ordinal),
        ),
        fill=colors["regression"],
        width=6,
    )
    regression_label = f"Least-squares trend over calendar date  (R² = {r_squared:.2f})"
    regression_box = draw.textbbox((0, 0), regression_label, font=annotation_font)
    draw.rectangle(
        (
            plot_left + 14,
            plot_top + 8,
            plot_left + 30 + regression_box[2] - regression_box[0],
            plot_top + 40,
        ),
        fill=colors["background"],
    )
    draw.text(
        (plot_left + 22, plot_top + 10),
        regression_label,
        font=annotation_font,
        fill=colors["regression"],
    )

    crossover = _persistent_positive_crossover(observations)
    if crossover is not None:
        left_index, right_index = crossover
        cutoff_ordinal = (
            observations[left_index].release_date.toordinal()
            + observations[right_index].release_date.toordinal()
        ) / 2
        cutoff_x = x_position(cutoff_ordinal)
        dash_length, dash_gap = 18, 12
        y = plot_top
        while y < plot_bottom:
            draw.line(
                (cutoff_x, y, cutoff_x, min(y + dash_length, plot_bottom)),
                fill=colors["cutoff"],
                width=4,
            )
            y += dash_length + dash_gap
        label = "Observed harness-leverage cutoff"
        box = draw.textbbox((0, 0), label, font=annotation_font)
        label_width = box[2] - box[0]
        label_x = cutoff_x - label_width - 28
        draw.rectangle(
            (label_x - 8, plot_top + 8, cutoff_x - 12, plot_top + 40),
            fill=colors["background"],
        )
        draw.text(
            (label_x, plot_top + 10),
            label,
            font=annotation_font,
            fill=colors["cutoff"],
        )

    label_offsets = {
        "gpt-oss-120b": (16, -58),
        "gemma4-e4b-mlx": (20, -10),
        "nemotron-3.5-30B-A3-reasoning": (-315, -58),
        "qwen3.8-2.4t-a95b": (-325, -68),
        "qwen3.8-27b-reasoning": (20, -54),
        "glm-5.3-flash": (-170, -58),
    }

    for index, (observation, (x, y)) in enumerate(zip(observations, coordinates)):
        color = colors["positive"] if observation.uplift >= 0 else colors["negative"]
        radius = 13
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        value_text = f"{observation.uplift:+.1f}"
        value_box = draw.textbbox((0, 0), value_text, font=value_font)
        value_y = y - 44 if observation.uplift >= 0 else y + 20
        draw.text(
            (x - (value_box[2] - value_box[0]) / 2, value_y),
            value_text,
            font=value_font,
            fill=color,
        )

        offset_x, offset_y = label_offsets.get(observation.model, (12, -54))
        model_label = DEFAULT_MODEL_LABELS.get(observation.model, observation.model).replace("\n", " ")
        draw.text(
            (x + offset_x, y + offset_y),
            model_label,
            font=label_font,
            fill=colors["text"],
        )

    # Calendar ticks on the continuous date axis.
    calendar_ticks = [
        date(2025, 8, 5),
        date(2025, 10, 1),
        date(2025, 12, 1),
        date(2026, 2, 1),
        date(2026, 4, 3),
        date(2026, 6, 1),
        date(2026, 8, 26),
    ]
    for tick_date in calendar_ticks:
        ordinal = tick_date.toordinal()
        if not first_ordinal <= ordinal <= last_ordinal:
            continue
        x = x_position(ordinal)
        label = (
            tick_date.strftime("%Y-%m-%d")
            if tick_date in {calendar_ticks[0], calendar_ticks[-1]}
            else tick_date.strftime("%b %Y")
        )
        box = draw.textbbox((0, 0), label, font=date_font)
        draw.line((x, plot_bottom, x, plot_bottom + 10), fill=colors["axis"], width=2)
        draw.text(
            (x - (box[2] - box[0]) / 2, plot_bottom + 24),
            label,
            font=date_font,
            fill=colors["muted"],
        )

    # Vertical y-axis title.
    y_title = Image.new("RGBA", (plot_height, 56), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_title)
    y_draw.text(
        (0, 5),
        "Harness uplift in average clemscore",
        font=axis_font,
        fill=colors["text"],
    )
    y_title = y_title.rotate(90, expand=True)
    image.paste(y_title, (30, int(plot_top + (plot_height - y_title.height) / 2)), y_title)

    x_label = "Model release date"
    x_box = draw.textbbox((0, 0), x_label, font=axis_font)
    draw.text(
        ((width - (x_box[2] - x_box[0])) / 2, height - 74),
        x_label,
        font=axis_font,
        fill=colors["text"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("output_path must end in .png or .pdf")
    image.save(output_path, dpi=(180, 180))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=QUESTION_DIR / "plots" / "harness_uplift_by_release_date.png",
    )
    args = parser.parse_args()
    output = plot_harness_uplift_by_release_date(
        wordle_results_csv=args.wordle_results,
        chronicle_results_csv=args.chronicle_results,
        output_path=args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
