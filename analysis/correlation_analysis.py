import pandas as pd
import statsmodels.api as sm
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 1. Load the data
file_path = "../data/diderot_with_velocity.csv"

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    df = pd.read_csv(file_path)

    # 2. Re-apply dictionary hits (Safety check)
    if 'cong_hits' not in df.columns:
        cong_rx = r'\b(aesthetic|matching|harmony|set|collection|vibe|minimal|complete|routine|looks good|cohesive|display)\b'
        df['text'] = df['text'].str.lower().fillna("")
        df['cong_hits'] = df['text'].apply(lambda x: len(re.findall(cong_rx, str(x))))

    # 3. Prepare Lagged Variables
    df = df.sort_values(['user_id', 'timestamp'])
    df['lagged_cong_hits'] = df.groupby('user_id')['cong_hits'].shift(1)
    df['lagged_aesthetic_score'] = df.groupby('user_id')['aesthetic_score'].shift(1)

    # 4. Filter for Correlation Analysis
    analysis_df = df[df['phase'] == 'Congruence'].dropna(subset=['delta_t', 'lagged_cong_hits', 'lagged_aesthetic_score'])

    # 5. Run Regression
    X = analysis_df[['lagged_cong_hits', 'lagged_aesthetic_score']]
    X = sm.add_constant(X)
    y = analysis_df['delta_t']
    model = sm.OLS(y, X).fit()

    # 6. Save Table to PDF
    with PdfPages('diderot_correlation_report.pdf') as pdf:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Create the text representation of the model summary
        summary_text = model.summary().as_text()
        
        plt.text(0.01, 0.99, summary_text, {
            'fontsize': 10, 
            'family': 'monospace'
        }, va='top', transform=ax.transAxes)
        
        plt.title("Correlation Analysis: Language as a Predictor of Speed", fontsize=14, pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print("PDF Created: diderot_correlation_report.pdf")
    print(model.summary())