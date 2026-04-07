import pandas as pd

# Load the new CSV
df = pd.read_csv("../data/diderot_with_triggers.csv")

# Clean up column names just in case of weird whitespace
df.columns = df.columns.str.strip()

print(f"Verified columns: {df.columns.tolist()}")

# Ensure types are correct
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort for temporal calculation
df = df.sort_values(['user_id', 'timestamp'])

# Calculate the gap in days
df['delta_t'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 86400

# Save result
df.to_csv("../data/diderot_with_velocity.csv", index=False)
print("Step 2 Complete: Temporal velocity (Delta T) calculated.")