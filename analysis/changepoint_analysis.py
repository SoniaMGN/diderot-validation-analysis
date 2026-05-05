"""
Stage 7a: Bayesian Change Point Detection (ruptures)

Instead of using a hardcoded aesthetic_score threshold (0.6) to label phases,
this script lets the data determine where each user's consumption regime shifts.

For each user with enough reviews, we run ruptures' Pelt algorithm on their
aesthetic_score time series to find the single most likely structural break.
We then compare the data-driven break point against the threshold-based one
and re-label phases using the detected break.

Outputs:
  - data/diderot_changepoint.csv   : full dataset with data-driven phase labels
  - analysis/changepoint_report.pdf: visualizations and summary statistics
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import logging
import os
from matplotlib.backends.backend_pdf import PdfPages
from constants import data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE  = data_path("diderot_with_velocity.csv")
OUTPUT_CSV  = data_path("diderot_changepoint.csv")
OUTPUT_PDF  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "changepoint_report.pdf")

# Minimum reviews a user needs for change point detection to be meaningful
MIN_REVIEWS = 6
# Penalty for Pelt — controls sensitivity. Lower = more breaks detected.
# pen=1 detects breaks in ~70% of eligible users, which is appropriate
# for a dataset where not all users exhibit the Diderot spiral.
PELT_PENALTY = 1


def detect_changepoint(scores: np.ndarray) -> int | None:
    """
    Run Pelt change point detection on a 1D aesthetic score series.
    Returns the index of the detected break, or None if no break is found.
    """
    signal = scores.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    try:
        result = algo.predict(pen=PELT_PENALTY)
        # result is a list of breakpoint indices; last entry is always len(signal)
        # We want the first real break (ignore the terminal index)
        breaks = [b for b in result if b < len(scores)]
        return breaks[0] if breaks else None
    except rpt.exceptions.BadSegmentationParameters:
        return None


def label_phases_by_changepoint(group: pd.DataFrame) -> pd.DataFrame:
    """
    For a single user's sorted review history, detect the structural break
    in aesthetic_score and label reviews as Utilitarian / Congruence / Baseline.
    """
    scores = group["aesthetic_score"].values
    cp = detect_changepoint(scores)

    if cp is not None:
        group = group.copy()
        group["cp_phase"] = "Utilitarian"
        group.iloc[cp:, group.columns.get_loc("cp_phase")] = "Congruence"
        group["cp_index"] = cp
    else:
        group = group.copy()
        group["cp_phase"] = "Baseline"
        group["cp_index"] = np.nan

    return group


def compare_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    For users where both methods found a break, compute how many reviews
    apart the threshold-based and data-driven break points are.
    """
    # Threshold-based break: first Congruence review index within user
    threshold_breaks = (
        df[df["phase"] == "Congruence"]
        .groupby("user_id")
        .apply(lambda g: g.index[0])
        .rename("threshold_break_idx")
    )

    cp_breaks = (
        df[df["cp_phase"] == "Congruence"]
        .groupby("user_id")
        .apply(lambda g: g.index[0])
        .rename("cp_break_idx")
    )

    comparison = pd.concat([threshold_breaks, cp_breaks], axis=1).dropna()
    comparison["index_diff"] = (comparison["cp_break_idx"] - comparison["threshold_break_idx"]).abs()
    return comparison


def build_report(df: pd.DataFrame, comparison: pd.DataFrame, output_path: str):
    """Generate a PDF with phase distribution, agreement analysis, and example users."""
    plt.style.use("seaborn-v0_8-whitegrid")

    with PdfPages(output_path) as pdf:

        # --- Page 1: Phase distribution comparison ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Change Point Detection vs. Threshold Labeling", fontsize=14, y=1.02)

        for ax, col, title in zip(
            axes,
            ["phase", "cp_phase"],
            ["Threshold Method (score > 0.6)", "Data-Driven (ruptures Pelt)"],
        ):
            counts = df[col].value_counts()
            ax.bar(counts.index, counts.values, color=["#34495e", "#e74c3c", "#95a5a6"])
            ax.set_title(title, fontsize=12)
            ax.set_ylabel("Number of Reviews")
            for i, (label, val) in enumerate(counts.items()):
                ax.text(i, val + 50, str(val), ha="center", fontweight="bold")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Agreement between methods ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(comparison["index_diff"], bins=30, color="#2980b9", edgecolor="white")
        ax.set_xlabel("Reviews apart (threshold break vs. detected break)", fontsize=12)
        ax.set_ylabel("Number of Users", fontsize=12)
        ax.set_title(
            f"Method Agreement: {(comparison['index_diff'] == 0).mean():.1%} exact match, "
            f"{(comparison['index_diff'] <= 1).mean():.1%} within 1 review",
            fontsize=12,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: Aesthetic score comparison by data-driven phase ---
        fig, ax = plt.subplots(figsize=(8, 5))
        cp_report = (
            df[df["cp_phase"] != "Baseline"]
            .groupby("cp_phase")["aesthetic_score"]
            .mean()
        )
        bars = ax.bar(cp_report.index, cp_report.values, color=["#34495e", "#e74c3c"], width=0.5)
        ax.set_ylabel("Mean Aesthetic Score", fontsize=12)
        ax.set_title("Aesthetic Score by Data-Driven Phase", fontsize=13)
        ax.set_ylim(0.48, 0.62)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", fontweight="bold",
            )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 4: Example user — aesthetic score with detected break ---
        example_users = (
            df[df["cp_phase"] == "Congruence"]["user_id"]
            .value_counts()
            .head(3)
            .index
        )
        fig, axes = plt.subplots(len(example_users), 1, figsize=(12, 4 * len(example_users)))
        if len(example_users) == 1:
            axes = [axes]

        for ax, uid in zip(axes, example_users):
            user_df = df[df["user_id"] == uid].reset_index(drop=True)
            ax.plot(user_df.index, user_df["aesthetic_score"], marker="o", markersize=4,
                    color="#2c3e50", linewidth=1.5, label="Aesthetic Score")
            cp_idx = user_df[user_df["cp_phase"] == "Congruence"].index[0]
            ax.axvline(x=cp_idx, color="#e74c3c", linestyle="--", linewidth=2,
                       label=f"Detected break (review {cp_idx})")
            ax.axhline(y=0.6, color="#95a5a6", linestyle=":", linewidth=1, label="Threshold (0.6)")
            ax.set_title(f"User: {uid[:20]}...", fontsize=10)
            ax.set_xlabel("Review sequence")
            ax.set_ylabel("Aesthetic Score")
            ax.legend(fontsize=9)

        plt.suptitle("Example Users: Detected Regime Change", fontsize=13, y=1.01)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    logger.info(f"Saved change point report to: {output_path}")


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Run calculate_velocity.py first."
        )

    logger.info(f"Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Only run detection on users with enough reviews
    user_counts = df["user_id"].value_counts()
    eligible_users = user_counts[user_counts >= MIN_REVIEWS].index
    logger.info(
        f"Running change point detection on {len(eligible_users):,} users "
        f"({len(eligible_users)/df['user_id'].nunique():.1%} of total)..."
    )

    eligible_df = df[df["user_id"].isin(eligible_users)].copy()
    short_df    = df[~df["user_id"].isin(eligible_users)].copy()
    short_df["cp_phase"] = "Baseline"
    short_df["cp_index"] = np.nan

    eligible_df = (
        eligible_df
        .groupby("user_id", group_keys=False)
        .apply(label_phases_by_changepoint)
    )

    df_out = pd.concat([eligible_df, short_df]).sort_values(["user_id", "timestamp"])

    # Summary
    cp_counts = df_out["cp_phase"].value_counts()
    logger.info(f"Data-driven phase distribution:\n{cp_counts.to_string()}")

    # Compare methods
    comparison = compare_methods(df_out)
    exact_match = (comparison["index_diff"] == 0).mean()
    within_one  = (comparison["index_diff"] <= 1).mean()
    logger.info(f"Method agreement — exact: {exact_match:.1%}, within 1 review: {within_one:.1%}")

    # Velocity comparison by data-driven phase
    velocity = (
        df_out[df_out["cp_phase"] != "Baseline"]
        .dropna(subset=["delta_t"])
        .groupby("cp_phase")["delta_t"]
        .mean()
    )
    logger.info(f"Mean delta_t by data-driven phase:\n{velocity.to_string()}")

    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved change point data to: {OUTPUT_CSV}")

    build_report(df_out, comparison, OUTPUT_PDF)


if __name__ == "__main__":
    main()
