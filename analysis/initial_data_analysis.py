import pandas as pd
import os

# 1. Fix Pathing
base_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(base_dir)
full_path = os.path.join(project_root, "data", "Amazon_reviews_2023.csv")

print(f"Loading data from: {full_path}")

try:
    df = pd.read_csv(full_path, low_memory=False)
    
    # 2. CONVERT TO DATETIME (The Fix)
    # This handles "2020-05-05 14:08:48" correctly
    print("Converting timestamps...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows that failed (just in case there's garbage in the CSV)
    df = df.dropna(subset=['timestamp'])

    # 3. Filter for 5-core
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df_final = df[df['user_id'].isin(valid_users)].copy()

    # 4. Sequential Sort
    # Sorting by date works perfectly now that they are datetime objects
    df_final = df_final.sort_values(['user_id', 'timestamp'])

    # 5. Save as JSONL (Pandas will save dates as ISO strings automatically)
    output_path = os.path.join(project_root, "diderot_working_set.jsonl")
    df_final.to_json(output_path, orient="records", lines=True)

    print("-" * 30)
    print(f"Success! {len(df_final)} usable rows saved.")
    print(f"Earliest Review: {df_final['timestamp'].min()}")
    print(f"Latest Review: {df_final['timestamp'].max()}")
    print("-" * 30)

except Exception as e:
    print(f"Error: {e}")