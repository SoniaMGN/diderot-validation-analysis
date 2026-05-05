"""
Stage 3: Compute aesthetic similarity scores for every review using the
Mixedbread embedding API, then flag reviews that exceed the trigger threshold.
"""

import numpy as np
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from mixedbread import Mixedbread
from constants import (
    data_path,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    AESTHETIC_ANCHOR,
    AESTHETIC_THRESHOLD,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE  = data_path("diderot_final_sequence.jsonl")
OUTPUT_FILE = data_path("diderot_final_scored.jsonl")


def get_anchor_embedding(client: Mixedbread) -> np.ndarray:
    """Encode the aesthetic anchor text and return its embedding vector."""
    res = client.embed(
        model=EMBEDDING_MODEL,
        input=[AESTHETIC_ANCHOR],
        normalized=True,
        encoding_format="float",
    )
    return np.array(res.data[0].embedding)


def score_texts(client: Mixedbread, texts: list[str], anchor_vec: np.ndarray) -> list[float]:
    """
    Batch-encode all review texts and compute cosine similarity against the anchor.
    Returns a list of scores in the same order as `texts`.
    """
    all_scores: list[float] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        res = client.embed(
            model=EMBEDDING_MODEL,
            input=batch,
            normalized=True,
            encoding_format="float",
        )
        batch_vecs = np.array([d.embedding for d in res.data])
        scores = np.dot(batch_vecs, anchor_vec).tolist()
        all_scores.extend(scores)

        if len(all_scores) % 1024 == 0:
            logger.info(f"Progress: {len(all_scores):,} / {len(texts):,} reviews scored")

    return all_scores


def main():
    api_key = os.getenv("MIXEDBREAD_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MIXEDBREAD_API_KEY is not set. Add it to your .env file."
        )

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Run diderot_sequential_sorting.py first."
        )

    client = Mixedbread(api_key=api_key)

    logger.info("Encoding aesthetic anchor...")
    anchor_vec = get_anchor_embedding(client)

    logger.info(f"Loading reviews from: {INPUT_FILE}")
    df = pd.read_json(INPUT_FILE, lines=True)
    texts = df["text"].fillna("").tolist()

    logger.info(f"Scoring {len(texts):,} reviews in batches of {EMBEDDING_BATCH_SIZE}...")
    df["aesthetic_score"] = score_texts(client, texts, anchor_vec)

    df["is_disruptive"] = df["aesthetic_score"] > AESTHETIC_THRESHOLD

    df.to_json(OUTPUT_FILE, orient="records", lines=True)
    logger.info(f"Saved scored data to: {OUTPUT_FILE}")
    logger.info(
        f"Disruptive reviews: {df['is_disruptive'].sum():,} "
        f"({df['is_disruptive'].mean():.1%} of total)"
    )


if __name__ == "__main__":
    main()
