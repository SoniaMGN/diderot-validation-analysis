"""
Stage 5: Calculate the inter-purchase interval (delta_t) for each review —
the number of days since the user's previous purchase.
"""

import pandas as pd
import logging
import os
from constants import data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE  = data_path("diderot_with_triggers.csv")
OUTPUT_FILE = data_path("diderot_with_velocity.csv")


def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by user and timestamp, then compute the day-gap between consecutive
    purchases for each user. The first purchase per user will have NaN delta_t.
    """
    df = df.sort_values(["user_id", "timestamp"])
    df["delta_t"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        / 86400  # convert seconds → days
    )
    return df


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Run identify_triggers.py first."
        )

    logger.info(f"Loading phase-labeled data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # Ensure correct dtype after CSV round-trip
    df.columns = df.columns.str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    dropped = df["timestamp"].isna().sum()
    if dropped:
        logger.warning(f"Dropped {dropped} rows with unparseable timestamps.")
    df = df.dropna(subset=["timestamp"])

    logger.info("Calculating inter-purchase intervals...")
    df = calculate_velocity(df)

    logger.info(
        f"delta_t stats (days):\n"
        f"  mean={df['delta_t'].mean():.2f}  "
        f"median={df['delta_t'].median():.2f}  "
        f"max={df['delta_t'].max():.2f}"
    )

    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved velocity data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
