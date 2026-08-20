#!/usr/bin/env python3
"""Generate the manuscript CD diagram from the rebuilt stats bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import studentized_range


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_MEANS = (
    PROJECT_ROOT / "results" / "reviewer_revision" / "claim_verification" / "five_method" / "dataset_means_f1.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "sreport" / "cd_diagram_f1.png"

METHOD_ORDER = ["qualsynth", "smote", "ctgan", "tabfairgdt", "tabddpm"]
DISPLAY_NAMES = {
    "qualsynth": "QualSynth",
    "smote": "SMOTE",
    "ctgan": "CTGAN",
    "tabfairgdt": "TabFairGDT",
    "tabddpm": "TabDDPM",
}
COLORS = {
    "qualsynth": "#1f77b4",
    "smote": "#2ca02c",
    "ctgan": "#ff7f0e",
    "tabfairgdt": "#9467bd",
    "tabddpm": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the F1 CD diagram from rebuilt stats.")
    parser.add_argument(
        "--dataset-means-csv",
        default=str(DEFAULT_DATASET_MEANS),
        help="CSV produced by scripts/verify_result_claims.py for the target scope.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT),
        help="PNG path for the rendered diagram.",
    )
    return parser.parse_args()


def load_dataset_means(path: Path) -> pd.DataFrame:
    dataset_means = pd.read_csv(path).set_index("dataset")[METHOD_ORDER]
    if dataset_means.isna().any().any():
        raise RuntimeError(f"Dataset means contain missing values: {path}")
    return dataset_means


def compute_mean_ranks(dataset_means: pd.DataFrame) -> tuple[pd.Series, float]:
    ranks = dataset_means.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0).sort_values()

    n_datasets = len(dataset_means)
    n_methods = len(METHOD_ORDER)
    q_alpha = studentized_range.ppf(0.95, n_methods, float("inf")) / (2**0.5)
    cd = q_alpha * ((n_methods * (n_methods + 1) / (6 * n_datasets)) ** 0.5)
    return mean_ranks, cd


def draw_cd_diagram(mean_ranks: pd.Series, cd: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=250)

    axis_y = 0.80
    ax.set_xlim(0.95, 5.05)
    ax.set_ylim(0.0, 1.42)
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Average rank (lower is better)", fontsize=11, labelpad=10)

    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_position(("data", axis_y))
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="x", labelsize=10, width=1.0)

    items = list(mean_ranks.items())
    left_items = items[:3]
    right_items = items[3:]

    left_label_x = 1.08
    left_line_x = 1.72
    right_label_x = 4.92
    right_line_x = 4.28
    left_ys = [0.56, 0.41, 0.26]
    right_ys = [0.56, 0.41]

    for (method, rank), label_y in zip(left_items, left_ys):
        color = COLORS[method]
        ax.plot(rank, axis_y, "o", color=color, markersize=7.5, zorder=3)
        ax.plot([rank, rank], [axis_y, label_y], color=color, linewidth=1.2)
        ax.plot([left_line_x, rank], [label_y, label_y], color=color, linewidth=1.2)
        ax.text(
            left_label_x,
            label_y,
            f"{DISPLAY_NAMES[method]} ({rank:.2f})",
            ha="left",
            va="center",
            fontsize=10.5,
        )

    for (method, rank), label_y in zip(right_items, right_ys):
        color = COLORS[method]
        ax.plot(rank, axis_y, "o", color=color, markersize=7.5, zorder=3)
        ax.plot([rank, rank], [axis_y, label_y], color=color, linewidth=1.2)
        ax.plot([rank, right_line_x], [label_y, label_y], color=color, linewidth=1.2)
        ax.text(
            right_label_x,
            label_y,
            f"{DISPLAY_NAMES[method]} ({rank:.2f})",
            ha="right",
            va="center",
            fontsize=10.5,
        )

    # Non-significant clique bar.
    ax.plot(
        [mean_ranks.iloc[0], mean_ranks.iloc[-1]],
        [1.02, 1.02],
        color="black",
        linewidth=2.2,
        solid_capstyle="butt",
    )

    # Critical-difference ruler.
    cd_y = 1.21
    cd_start = 1.18
    cd_end = cd_start + cd
    ax.plot([cd_start, cd_end], [cd_y, cd_y], color="black", linewidth=2.0)
    ax.plot([cd_start, cd_start], [cd_y - 0.035, cd_y + 0.035], color="black", linewidth=2.0)
    ax.plot([cd_end, cd_end], [cd_y - 0.035, cd_y + 0.035], color="black", linewidth=2.0)
    ax.text(
        (cd_start + cd_end) / 2,
        cd_y + 0.055,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")


def main() -> None:
    args = parse_args()
    dataset_means = load_dataset_means(Path(args.dataset_means_csv))
    mean_ranks, cd = compute_mean_ranks(dataset_means)
    output_path = Path(args.output_path)
    draw_cd_diagram(mean_ranks, cd, output_path)
    print(f"Saved diagram to {output_path}")
    for method, rank in mean_ranks.items():
        print(f"{DISPLAY_NAMES[method]}: {rank:.3f}")
    print(f"CD: {cd:.3f}")


if __name__ == "__main__":
    main()
