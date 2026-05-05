"""
Stage 4: Label each review with a phase — Utilitarian (pre-trigger),
Congruence (post-trigger), or Baseline (no trigger found).

The trigger is defined as the first review where aesthetic_score > AESTHETIC_THRESHOLD.
"""

import pandas as pd
import numpy as np
import logging
import os
from constants import data_path, AESTHETIC_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE  = data_path("diderot_final_scored.jsonl")
OUTPUT_FILE = data_path("diderot_with_triggers.csv")


def find_trigger(group: pd.DataFrame) -> pd.DataFrame:
    """
    For a single user's review history (sorted by timestamp), find the first
    review that crosses the aesthetic threshold and label all reviews accordingly.
    """
    trigger_hits = group[group["aesthetic_score"] > AESTHETIC_THRESHOLD].sort_values("timestamp")

    if not trigger_hits.empty:
        t0 = trigger_hits["timestamp"].iloc[0]
        group["phase"] = np.where(group["timestamp"] < t0, "Utilitarian", "Congruence")
    else:
        group["phase"] = "Baseline"

    return group


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Run embeddings.py first."
        )

    logger.info(f"Loading scored data from: {INPUT_FILE}")
    df = pd.read_json(INPUT_FILE, lines=True)

    logger.info("Standardizing timestamps...")
    # Handle both millisecond epoch integers and ISO strings
    if pd.api.types.is_integer_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    dropped = df["timestamp"].isna().sum()
    if dropped:
        logger.warning(f"Dropped {dropped} rows with unparseable timestamps.")
    df = df.dropna(subset=["timestamp"])

    logger.info(f"Labeling phases for {df['user_id'].nunique():,} users...")
    df = df.sort_values(["user_id", "timestamp"])
    df = df.groupby("user_id", group_keys=False).apply(find_trigger)

    phase_counts = df["phase"].value_counts()
    logger.info(f"Phase distribution:\n{phase_counts.to_string()}")

    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved phase-labeled data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
