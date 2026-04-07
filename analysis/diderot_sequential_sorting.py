import pandas as pd
import json
import os

# 1. Pathing (adjusting for your /analysis and /data folders)
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)
mapping_path = os.path.join(project_root, "data", "asin2category.json") # The 1.25GB file
reviews_path = os.path.join(project_root, "data",  "diderot_working_set.jsonl")

print("Loading sequenced reviews...")
df = pd.read_json(reviews_path, lines=True)

# 2. Efficiently Load the 1.25GB Mapping
# We only care about the ASINs that are actually in our 15k sample
print("Mapping categories (this may take a moment due to file size)...")
asin_map = {}
target_asins = set(df['asin'].unique())

with open(mapping_path, 'r') as f:
    # Most ASIN maps are {asin: category} or a list of dicts
    # We load it line by line or as a single dict depending on format
    try:
        raw_map = json.load(f)
        for asin, cat in raw_map.items():
            if asin in target_asins:
                asin_map[asin] = cat
    except:
        # If it's a JSONL format (one dict per line)
        f.seek(0)
        for line in f:
            data = json.loads(line)
            # Adjust keys based on your specific file structure
            asin = data.get('asin')
            cat = data.get('category')
            if asin in target_asins:
                asin_map[asin] = cat

# 3. Apply the Mapping
df['category'] = df['asin'].map(asin_map)

# 4. Final Sequence Sort
# Now we sort by User -> Time -> Category
df = df.sort_values(by=['user_id', 'timestamp'])

# 5. Save the 'Master Diderot Sequence'
output_path = os.path.join(project_root, "diderot_final_sequence.jsonl")
df.to_json(output_path, orient="records", lines=True)

print("-" * 30)
print("SUCCESS: Category data integrated.")
print(df[['user_id', 'timestamp', 'category', 'text']].head(10))
print("-" * 30)