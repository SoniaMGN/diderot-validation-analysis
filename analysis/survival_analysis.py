"""
Stage 7b: Survival Analysis — Cox Proportional Hazards Model (lifelines)

Replaces the OLS regression with a model that is actually appropriate for
time-between-events data. Instead of predicting delta_t directly (which is
right-skewed and violates OLS normality assumptions), we model the *hazard*
of making the next purchase.

A higher hazard = shorter time to next purchase = faster consumption.

We use the data-driven phase labels from changepoint_analysis.py if available,
falling back to the threshold-based labels otherwise.

The model tests:
  - Does lagged congruence language increase purchase hazard?
  - Does lagged aesthetic score increase purchase hazard?
  - Does being in the Congruence phase independently increase hazard?

Outputs:
  - analysis/survival_report.pdf: Kaplan-Meier curves + Cox model summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import re
from matplotlib.backends.backend_pdf import PdfPages
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from constants import data_path, CONGRUENCE_REGEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Prefer data-driven labels; fall back to threshold labels
CP_FILE       = data_path("diderot_changepoint.csv")
FALLBACK_FILE = data_path("diderot_with_velocity.csv")
OUTPUT_PDF    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survival_report.pdf")


def load_data() -> tuple[pd.DataFrame, str]:
    """Load the best available dataset and return it with the phase column name."""
    if os.path.exists(CP_FILE):
        logger.info(f"Using data-driven phase labels from: {CP_FILE}")
        df = pd.read_csv(CP_FILE)
        phase_col = "cp_phase"
    else:
        logger.warning(f"Change point file not found — falling back to threshold labels.")
        df = pd.read_csv(FALLBACK_FILE)
        phase_col = "phase"
    return df, phase_col


def prepare_survival_data(df: pd.DataFrame, phase_col: str) -> pd.DataFrame:
    """
    Build the survival analysis dataset.

    Each row represents a purchase interval. The 'event' is always 1 (the next
    purchase was observed). We add lagged linguistic features as covariates.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id", "timestamp"])

    # Compute congruence hits if missing
    if "cong_hits" not in df.columns:
        df["text"] = df["text"].str.lower().fillna("")
        df["cong_hits"] = df["text"].apply(lambda x: len(re.findall(CONGRUENCE_REGEX, x)))

    # Lagged features: language/score at T-1 predicts interval at T
    df["lagged_cong_hits"]       = df.groupby("user_id")["cong_hits"].shift(1)
    df["lagged_aesthetic_score"] = df.groupby("user_id")["aesthetic_score"].shift(1)

    # Binary phase indicator (1 = Congruence, 0 = Utilitarian)
    df["in_congruence"] = (df[phase_col] == "Congruence").astype(int)

    # Drop rows without a complete interval (first purchase per user, Baseline)
    survival_df = df[df[phase_col] != "Baseline"].dropna(
        subset=["delta_t", "lagged_cong_hits", "lagged_aesthetic_score"]
    )

    # Remove zero-duration intervals (same-day purchases cause issues in survival models)
    survival_df = survival_df[survival_df["delta_t"] > 0].copy()

    # Survival analysis requires: duration (T) and event observed (E=1 for all here)
    survival_df["T"] = survival_df["delta_t"]
    survival_df["E"] = 1  # all intervals are fully observed (no censoring)

    return survival_df


def run_kaplan_meier(survival_df: pd.DataFrame) -> tuple:
    """Fit KM curves for Utilitarian vs Congruence and run log-rank test."""
    util_T = survival_df[survival_df["in_congruence"] == 0]["T"]
    cong_T = survival_df[survival_df["in_congruence"] == 1]["T"]
    util_E = survival_df[survival_df["in_congruence"] == 0]["E"]
    cong_E = survival_df[survival_df["in_congruence"] == 1]["E"]

    kmf_util = KaplanMeierFitter()
    kmf_cong = KaplanMeierFitter()
    kmf_util.fit(util_T, util_E, label="Utilitarian Phase")
    kmf_cong.fit(cong_T, cong_E, label="Congruence Phase")

    lr_result = logrank_test(util_T, cong_T, util_E, cong_E)
    logger.info(f"Log-rank test p-value: {lr_result.p_value:.4f}")

    return kmf_util, kmf_cong, lr_result


def run_cox(survival_df: pd.DataFrame) -> CoxPHFitter:
    """Fit Cox Proportional Hazards model."""
    cox_df = survival_df[["T", "E", "lagged_cong_hits", "lagged_aesthetic_score", "in_congruence"]].copy()

    # Standardize covariates for interpretable coefficients
    for col in ["lagged_cong_hits", "lagged_aesthetic_score"]:
        mean, std = cox_df[col].mean(), cox_df[col].std()
        if std > 0:
            cox_df[col] = (cox_df[col] - mean) / std

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="T", event_col="E")
    logger.info(f"\n{cph.summary.to_string()}")
    return cph


def build_report(
    kmf_util, kmf_cong, lr_result, cph: CoxPHFitter, survival_df: pd.DataFrame, output_path: str
):
    """Generate a multi-page PDF with KM curves, hazard ratios, and model summary."""
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#34495e", "#e74c3c"]

    with PdfPages(output_path) as pdf:

        # --- Page 1: Kaplan-Meier survival curves ---
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf_util.plot_survival_function(ax=ax, color=colors[0], ci_show=True)
        kmf_cong.plot_survival_function(ax=ax, color=colors[1], ci_show=True)
        ax.set_xlabel("Days since last purchase", fontsize=12)
        ax.set_ylabel("Probability of NOT yet purchasing", fontsize=12)
        ax.set_title(
            f"Kaplan-Meier: Time to Next Purchase by Phase\n"
            f"Log-rank p = {lr_result.p_value:.4f}  "
            f"({'significant' if lr_result.p_value < 0.05 else 'not significant'} at α=0.05)",
            fontsize=13,
        )
        ax.set_xlim(0, 500)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Cox hazard ratios (forest plot) ---
        fig, ax = plt.subplots(figsize=(10, 5))
        summary = cph.summary.copy()
        summary = summary.sort_values("exp(coef)")

        y_pos = range(len(summary))
        ax.barh(
            y_pos,
            summary["exp(coef)"] - 1,
            left=1,
            color=[colors[1] if v > 1 else colors[0] for v in summary["exp(coef)"]],
            height=0.5,
            alpha=0.8,
        )
        ax.errorbar(
            summary["exp(coef)"],
            y_pos,
            xerr=[
                summary["exp(coef)"] - summary["exp(coef) lower 95%"],
                summary["exp(coef) upper 95%"] - summary["exp(coef)"],
            ],
            fmt="none",
            color="black",
            capsize=4,
        )
        ax.axvline(x=1, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(summary.index, fontsize=11)
        ax.set_xlabel("Hazard Ratio (HR > 1 = faster purchasing)", fontsize=12)
        ax.set_title(
            "Cox PH Model: Hazard Ratios with 95% CI\n"
            "(covariates standardized; HR > 1 means higher purchase rate)",
            fontsize=13,
        )

        # Annotate with HR and p-value
        for i, (idx, row) in enumerate(summary.iterrows()):
            ax.text(
                row["exp(coef)"] + 0.01,
                i,
                f"HR={row['exp(coef)']:.3f}  p={row['p']:.3f}",
                va="center", fontsize=9,
            )

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: Cox model summary table ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis("off")
        summary_text = cph.summary.round(4).to_string()
        plt.text(
            0.01, 0.95,
            "Cox Proportional Hazards Model Summary\n\n" + summary_text,
            {"fontsize": 10, "family": "monospace"},
            va="top", transform=ax.transAxes,
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 4: Median survival by phase ---
        fig, ax = plt.subplots(figsize=(8, 5))
        medians = {
            "Utilitarian": kmf_util.median_survival_time_,
            "Congruence":  kmf_cong.median_survival_time_,
        }
        bars = ax.bar(medians.keys(), medians.values(), color=colors, width=0.5)
        ax.set_ylabel("Median days to next purchase", fontsize=12)
        ax.set_title("Median Survival Time by Phase", fontsize=13)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f} days",
                ha="center", fontweight="bold", fontsize=11,
            )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    logger.info(f"Saved survival report to: {output_path}")


def main():
    df, phase_col = load_data()

    logger.info("Preparing survival dataset...")
    survival_df = prepare_survival_data(df, phase_col)
    logger.info(
        f"Survival dataset: {len(survival_df):,} intervals, "
        f"{survival_df['user_id'].nunique():,} users"
    )

    logger.info("Fitting Kaplan-Meier curves...")
    kmf_util, kmf_cong, lr_result = run_kaplan_meier(survival_df)

    logger.info("Fitting Cox Proportional Hazards model...")
    cph = run_cox(survival_df)

    build_report(kmf_util, kmf_cong, lr_result, cph, survival_df, OUTPUT_PDF)


if __name__ == "__main__":
    main()
