"""
Shared constants for the Diderot Effect analysis pipeline.
Centralizing these avoids duplication and makes tuning easy.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def data_path(filename: str) -> str:
    """Return the absolute path to a file in the data directory."""
    return os.path.join(DATA_DIR, filename)

# --- Embedding Model ---
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_BATCH_SIZE = 128

AESTHETIC_ANCHOR = (
    "I bought this because it matches my current beauty routine and fits my vibe "
    "perfectly; it has that minimal, clean girl look that completes my collection "
    "and looks so good with my other viral products and nice packaging."
)

# --- Phase Labeling ---
AESTHETIC_THRESHOLD = 0.6  # Minimum aesthetic score to qualify as a trigger event
MIN_REVIEWS_PER_USER = 5   # 5-core filter

# --- Regex Dictionaries ---
UTILITARIAN_REGEX = r'\b(functional|replacement|needed|broke|works|utility|basic|simple|necessity|daily|price|cheap)\b'
CONGRUENCE_REGEX  = r'\b(aesthetic|matching|harmony|set|collection|vibe|minimal|complete|routine|looks good|cohesive|display)\b'
