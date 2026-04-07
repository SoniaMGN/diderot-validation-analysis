import pandas as pd
import re

# 1. Load the velocity data
df = pd.read_csv("../data/diderot_with_velocity.csv")

# 2. Define Custom Dictionaries (Regex)
# Utilitarian: Focus on need, function, and replacement
util_rx = r'\b(functional|replacement|needed|broke|works|utility|basic|simple|necessity|daily|price|cheap)\b'
# Congruence: Focus on matching, aesthetics, and collection building
cong_rx = r'\b(aesthetic|matching|harmony|set|collection|vibe|minimal|complete|routine|looks good|cohesive|display)\b'

# 3. Clean text and count hits
df['text'] = df['text'].str.lower().fillna("")
df['util_hits'] = df['text'].apply(lambda x: len(re.findall(util_rx, x)))
df['cong_hits'] = df['text'].apply(lambda x: len(re.findall(cong_rx, x)))

# 4. FINAL AGGREGATION: The "Thesis Table"
# We exclude 'Baseline' users (who never spiraled) and the first purchase (NaN delta_t)
# to get a clean comparison of the transition.
report = df[df['phase'] != 'Baseline'].dropna(subset=['delta_t']).groupby('phase').agg({
    'delta_t': 'mean',        # H2: Velocity
    'util_hits': 'mean',      # H1: Utilitarian markers
    'cong_hits': 'mean',      # H1: Congruence markers
    'aesthetic_score': 'mean' # Validation: Embedding alignment
}).round(3)

print("\n" + "="*60)
print("FINAL THESIS RESULTS: THE DIDEROT EFFECT IN BEAUTY")
print("="*60)
print(report)
print("="*60)

# 5. Save the final summary for your LaTeX table
report.to_csv("../data/final_diderot_report.csv")
print("\nFinal report saved to data/final_diderot_report.csv")