"""
Stage 6: Automated Content Analysis.

Counts utilitarian vs. congruence keyword hits per review using the shared
regex dictionaries, then aggregates results into the final thesis summary table.
"""

import pandas as pd
import re
import logging
import os
from constants import data_path, UTILITARIAN_REGEX, CONGRUENCE_REGEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE  = data_path("diderot_with_velocity.csv")
OUTPUT_FILE = data_path("final_diderot_report.csv")


def apply_dictionaries(df: pd.DataFrame) -> pd.DataFrame:
    """Count regex keyword hits for each review."""
    text = df["text"].str.lower().fillna("")
    df["util_hits"] = text.apply(lambda x: len(re.findall(UTILITARIAN_REGEX, x)))
    df["cong_hits"] = text.apply(lambda x: len(re.findall(CONGRUENCE_REGEX, x)))
    return df


def build_thesis_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean metrics by phase, excluding Baseline users and the first
    purchase per user (NaN delta_t) for a clean Utilitarian vs. Congruence comparison.
    """
    filtered = df[df["phase"] != "Baseline"].dropna(subset=["delta_t"])

    report = filtered.groupby("phase").agg(
        delta_t=("delta_t", "mean"),
        util_hits=("util_hits", "mean"),
        cong_hits=("cong_hits", "mean"),
        aesthetic_score=("aesthetic_score", "mean"),
    ).round(3)

    return report


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Run calculate_velocity.py first."
        )

    logger.info(f"Loading velocity data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    logger.info("Applying keyword dictionaries...")
    df = apply_dictionaries(df)

    report = build_thesis_table(df)

    separator = "=" * 60
    logger.info(f"\n{separator}")
    logger.info("FINAL THESIS RESULTS: THE DIDEROT EFFECT IN BEAUTY")
    logger.info(separator)
    logger.info(f"\n{report.to_string()}")
    logger.info(separator)

    report.to_csv(OUTPUT_FILE)
    logger.info(f"Saved thesis table to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
