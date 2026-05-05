"""
Stage 8: Visualization.

Generates the primary thesis charts:
  - Page 1: Temporal Acceleration (H2) — bar chart of delta_t by phase
  - Page 2: Linguistic Shift (H1) — grouped bars for util/cong hits by phase
  - Page 3: Aesthetic Score Shift — bar chart of mean aesthetic_score by phase

Output is saved as a multi-page PDF.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diderot_thesis_results.pdf")

# Hard-coded final results (from thesis table) — update if re-running the pipeline
RESULTS = {
    "Phase":           ["Utilitarian", "Congruence"],
    "Delta_T":         [144.140, 82.172],
    "Util_Hits":       [0.319, 0.361],
    "Cong_Hits":       [0.147, 0.201],
    "Aesthetic_Score": [0.521, 0.566],
}

COLORS = ["#34495e", "#e74c3c"]
plt.style.use("seaborn-v0_8-whitegrid")


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}", suffix: str = ""):
    """Annotate each bar with its value."""
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + ax.get_ylim()[1] * 0.02,
            fmt.format(yval) + suffix,
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )


def page_temporal_acceleration(df: pd.DataFrame) -> plt.Figure:
    """H2: Bar chart of mean inter-purchase interval by phase."""
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(df["Phase"], df["Delta_T"], color=COLORS, width=0.6)
    ax.set_ylabel(r"Days between Purchases ($\Delta T$)", fontsize=12)
    ax.set_title("H2 Validation: Temporal Acceleration in the Beauty Spiral", fontsize=14, pad=20)
    ax.set_ylim(0, 180)
    add_bar_labels(ax, bars, suffix=" days")
    plt.tight_layout()
    return fig


def page_linguistic_shift(df: pd.DataFrame) -> plt.Figure:
    """H1: Grouped bar chart comparing utilitarian vs. congruence markers by phase."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df["Phase"]))
    width = 0.35

    bars_util = ax.bar(x - width / 2, df["Util_Hits"], width, label="Utilitarian Markers", color=COLORS[0])
    bars_cong = ax.bar(x + width / 2, df["Cong_Hits"], width, label="Congruence Markers",  color=COLORS[1])

    ax.set_ylabel("Mean Keyword Hits per Review", fontsize=12)
    ax.set_title("H1 Validation: Linguistic Shift Across Phases", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Phase"])
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=11)
    add_bar_labels(ax, bars_util)
    add_bar_labels(ax, bars_cong)
    plt.tight_layout()
    return fig


def page_aesthetic_score(df: pd.DataFrame) -> plt.Figure:
    """Validation: Bar chart of mean aesthetic embedding score by phase."""
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(df["Phase"], df["Aesthetic_Score"], color=COLORS, width=0.6)
    ax.set_ylabel("Mean Aesthetic Score (S)", fontsize=12)
    ax.set_title("Embedding Validation: Aesthetic Score Shift Across Phases", fontsize=14, pad=20)
    ax.set_ylim(0.48, 0.60)
    add_bar_labels(ax, bars, fmt="{:.3f}")
    plt.tight_layout()
    return fig


def main():
    df = pd.DataFrame(RESULTS)

    logger.info("Generating thesis visualizations...")
    with PdfPages(OUTPUT_PDF) as pdf:
        for page_fn in [page_temporal_acceleration, page_linguistic_shift, page_aesthetic_score]:
            fig = page_fn(df)
            pdf.savefig(fig)
            plt.close(fig)

    logger.info(f"Saved thesis charts to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
