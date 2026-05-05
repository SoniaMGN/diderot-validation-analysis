"""
Stage 2: Map ASINs to product categories and produce the final
time-sorted sequence used for embedding.
"""

import pandas as pd
import json
import logging
import os
from constants import data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE   = data_path("diderot_working_set.jsonl")
MAPPING_FILE = data_path("asin2category.json")   # 1.25 GB file — loaded selectively
OUTPUT_FILE  = data_path("diderot_final_sequence.jsonl")


def load_category_map(mapping_path: str, target_asins: set) -> dict:
    """
    Load only the ASIN→category entries we actually need from the large mapping file.
    Supports both a single JSON dict and a JSONL format.
    """
    if not os.path.exists(mapping_path):
        logger.warning(f"Category mapping file not found: {mapping_path}. Skipping category enrichment.")
        return {}

    logger.info("Loading category mapping (this may take a moment)...")
    asin_map: dict = {}

    with open(mapping_path, "r") as f:
        try:
            raw_map = json.load(f)
            for asin, cat in raw_map.items():
                if asin in target_asins:
                    asin_map[asin] = cat
        except json.JSONDecodeError:
            # Fall back to JSONL format
            f.seek(0)
            for line in f:
                try:
                    data = json.loads(line)
                    asin = data.get("asin")
                    cat  = data.get("category")
                    if asin and asin in target_asins:
                        asin_map[asin] = cat
                except json.JSONDecodeError:
                    continue

    logger.info(f"Mapped {len(asin_map):,} of {len(target_asins):,} unique ASINs.")
    return asin_map


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}. Run initial_data_analysis.py first.")

    logger.info("Loading sequenced reviews...")
    df = pd.read_json(INPUT_FILE, lines=True)

    target_asins = set(df["asin"].unique())
    asin_map = load_category_map(MAPPING_FILE, target_asins)

    df["category"] = df["asin"].map(asin_map)
    df = df.sort_values(["user_id", "timestamp"])

    df.to_json(OUTPUT_FILE, orient="records", lines=True)
    logger.info(f"Saved sorted sequence to: {OUTPUT_FILE}")
    logger.info(f"Category coverage: {df['category'].notna().mean():.1%} of rows")


if __name__ == "__main__":
    main()
