import pandas as pd
import numpy as np

# Load the scored data
df = pd.read_json("../data/diderot_final_scored.jsonl", lines=True)

# Standardize timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

def find_trigger(group):
    # Capture the user_id from the name of the group
    uid = group.name 
    
    threshold = 0.6
    trigger_hits = group[group['aesthetic_score'] > threshold].sort_values('timestamp')
    
    if not trigger_hits.empty:
        t0 = trigger_hits['timestamp'].iloc[0]
        group['phase'] = np.where(group['timestamp'] < t0, 'Utilitarian', 'Congruence')
    else:
        group['phase'] = 'Baseline'
    
    # FORCE user_id to be a column in this group
    group['user_id'] = uid
    return group

print("Identifying triggers and locking user_id column...")
# We group by user_id but pass it as the 'name' to the function
df = df.groupby('user_id', group_keys=False).apply(find_trigger)

# Final save
df.to_csv("../data/diderot_with_triggers.csv", index=False)
print("Step 1 RE-COMPLETE: 'user_id' is now a hard-coded column.")