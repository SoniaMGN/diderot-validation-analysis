import pandas as pd
from mixedbread import Mixedbread
import numpy as np
import os

# 1. Initialize
mxbai = Mixedbread(api_key= os.getenv("EMBEDDING_API_KEY"))

# 2. Define our "Aesthetic Anchor"
anchor_text = "I bought this because it matches my current beauty routine and fits my vibe perfectly; it has that minimal, clean girl look that completes my collection and looks so good with my other viral products and nice packaging."

print("Encoding Anchor and Reviews via Mixedbread...")

# 3. Get the Anchor Embedding
anchor_res = mxbai.embed(
    model="mixedbread-ai/mxbai-embed-large-v1",
    input=[anchor_text],
    normalized=True,
    encoding_format="float"
)
anchor_vec = np.array(anchor_res.data[0].embedding)

# 4. Load your sorted data
df = pd.read_json("../data/diderot_final_sequence.jsonl", lines=True)
texts = df['text'].fillna("").tolist()

# 5. Batch Process (Crucial for 14k rows)
all_scores = []
batch_size = 128

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    res = mxbai.embed(
        model="mixedbread-ai/mxbai-embed-large-v1",
        input=batch,
        normalized=True,
        encoding_format="float"
    )
    batch_vecs = np.array([d.embedding for d in res.data])
    scores = np.dot(batch_vecs, anchor_vec)
    all_scores.extend(scores)

    if len(all_scores) % 1024 == 0:
        print(f"Progress: {len(all_scores)} / {len(texts)} analyzed...")

df['aesthetic_score'] = all_scores

# 6. Mark the "Disruptive Item"
df['is_disruptive'] = df['aesthetic_score'] > 0.6
df.to_json("diderot_final_scored.jsonl", orient="records", lines=True)

print("Analysis Complete! Data saved with Semantic Scores.")