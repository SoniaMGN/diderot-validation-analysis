import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# 1. Setup the Data based on your Final Results
data = {
    'Phase': ['Utilitarian', 'Congruence'],
    'Delta_T': [144.140, 82.172],
    'Util_Hits': [0.319, 0.361],
    'Cong_Hits': [0.147, 0.201],
    'Aesthetic_Score': [0.521, 0.566]
}
df_results = pd.DataFrame(data)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#34495e', '#e74c3c']

# 2. Create the PDF
with PdfPages('diderot_thesis_results.pdf') as pdf:
    
    # --- PAGE 1: Temporal Acceleration (H2) ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars = ax1.bar(df_results['Phase'], df_results['Delta_T'], color=colors, width=0.6)
    ax1.set_ylabel('Days between Purchases ($\Delta T$)', fontsize=12)
    ax1.set_title('H2 Validation: Temporal Acceleration in the Beauty Spiral', fontsize=14, pad=20)
    ax1.set_ylim(0, 180)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval:.2f} days',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    # --- PAGE 2: Linguistic Shift (H1 - Dictionary) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_results['Phase']))
    width = 0.35
    ax2.bar(x - width/2, df_results['Util_Hits'], width, label='Utilitarian Markers', color='#95a5a6')
    ax2.bar(x + width/2, df_results['Cong_Hits'], width, label='Congruence Markers', color='#f1c40f')
    ax2.set_ylabel('Mean Keyword Hits per Review', fontsize=12)
    ax2.set_title('H1 Validation: Shift in Consumer Vocabulary', fontsize=14, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_results['Phase'])
    ax2.legend()
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

    # --- PAGE 3: Aesthetic Alignment (H1 - AI Embeddings) ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(df_results['Phase'], df_results['Aesthetic_Score'], marker='o',
             linestyle='-', color='#2ecc71', linewidth=3, markersize=10)
    ax3.fill_between(df_results['Phase'], df_results['Aesthetic_Score'], alpha=0.2, color='#2ecc71')
    ax3.set_ylabel('Cosine Similarity to Aesthetic Anchor ($S$)', fontsize=12)
    ax3.set_title('Semantic Progression toward Aesthetic Ideal', fontsize=14, pad=20)
    ax3.set_ylim(0.4, 0.65)
    plt.tight_layout()
    pdf.savefig(fig3)
    plt.close(fig3)

    # --- PAGE 4: Summary Table ---
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=df_results.round(3).values, 
                      colLabels=df_results.columns, 
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    ax4.set_title('Summary Table: Diderot Validation Results', fontsize=14, pad=20)
    plt.tight_layout()
    pdf.savefig(fig4)
    plt.close(fig4)

print("PDF created: diderot_thesis_results.pdf")