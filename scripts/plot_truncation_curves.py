#!/usr/bin/env python3
"""Plot ActionPiece and RPG truncation results for DY, QB, and Beauty."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


DIMENSIONS = [128, 768, 1536, 4096]

# Metrics are named explicitly because the supplied RPG-Beauty row used a
# different column order from the DY/QB rows.
RESULTS = {
    "DY": {
        "ActionPiece": {
            "R@10": [0.1508, 0.1492, 0.1356, 0.1401],
            "N@10": [0.0959, 0.0949, 0.0832, 0.0851],
        },
        "RPG": {
            "R@10": [0.1378, 0.1103, 0.1102, 0.0835],
            "N@10": [0.0877, 0.0717, 0.0733, 0.0598],
        },
    },
    "QB": {
        "ActionPiece": {
            "R@10": [0.2821, 0.2969, 0.2972, 0.2925],
            "N@10": [0.2352, 0.2376, 0.2419, 0.2401],
        },
        "RPG": {
            "R@10": [0.3110, 0.3202, 0.2943, 0.2528],
            "N@10": [0.2293, 0.2431, 0.2039, 0.1726],
        },
    },
    "Beauty": {
        "ActionPiece": {
            "R@10": [0.0777, 0.0779, 0.0786, 0.0756],
            "N@10": [0.0422, 0.0434, 0.0441, 0.0410],
        },
        "RPG": {
            "R@10": [0.0767, 0.0757, 0.0697, 0.0541],
            "N@10": [0.0449, 0.0446, 0.0402, 0.0319],
        },
    },
}

MODEL_STYLES = {
    "ActionPiece": {"color": "#0072B2", "marker": "o"},
    "RPG": {"color": "#D55E00", "marker": "s"},
}

METRIC_STYLES = {
    "R@10": {"linestyle": "-"},
    "N@10": {"linestyle": "--"},
}


def plot_dataset(dataset: str, output_dir: Path, formats: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x_positions = range(len(DIMENSIONS))

    for model, metrics in RESULTS[dataset].items():
        for metric, values in metrics.items():
            ax.plot(
                x_positions,
                values,
                color=MODEL_STYLES[model]["color"],
                marker=MODEL_STYLES[model]["marker"],
                linestyle=METRIC_STYLES[metric]["linestyle"],
                linewidth=2.2,
                markersize=6,
                label=f"{model} · {metric}",
            )

    ax.set_title(f"{dataset}: embedding-dimension truncation", pad=10)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Metric value")
    ax.set_xticks(list(x_positions), [str(d) for d in DIMENSIONS])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.06, y=0.12)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    for output_format in formats:
        output_path = output_dir / f"truncation_{dataset.lower()}.{output_format}"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        print(f"saved: {output_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/truncation"),
        help="Directory for generated figures (default: plots/truncation)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
        help="Output formats (default: png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in RESULTS:
        plot_dataset(dataset, args.output_dir, args.formats)


if __name__ == "__main__":
    main()
