"""
Stage 1: Load raw Amazon reviews CSV, apply 5-core user filter,
convert timestamps, and save the working set as JSONL.
"""

import pandas as pd
import logging
import os
from constants import data_path, MIN_REVIEWS_PER_USER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_CSV  = data_path("Amazon_reviews_2023.csv")
OUTPUT_FILE = data_path("diderot_working_set.jsonl")


def load_and_filter(input_path: str, min_reviews: int = MIN_REVIEWS_PER_USER) -> pd.DataFrame:
    """Load the raw CSV, parse timestamps, and keep only 5-core users."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    logger.info("Converting timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    dropped = df["timestamp"].isna().sum()
    if dropped:
        logger.warning(f"Dropped {dropped} rows with unparseable timestamps.")
    df = df.dropna(subset=["timestamp"])

    # 5-core filter: keep users with at least min_reviews reviews
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_reviews].index
    df_final = df[df["user_id"].isin(valid_users)].copy()
    df_final = df_final.sort_values(["user_id", "timestamp"])

    logger.info(f"Retained {len(df_final):,} rows across {df_final['user_id'].nunique():,} users.")
    return df_final


def main():
    df = load_and_filter(INPUT_CSV)
    df.to_json(OUTPUT_FILE, orient="records", lines=True)
    logger.info(f"Saved working set to: {OUTPUT_FILE}")
    logger.info(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
