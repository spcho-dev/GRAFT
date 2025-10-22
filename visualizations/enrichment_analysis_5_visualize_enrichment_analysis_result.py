import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- Configuration ---
enrichment_csv_dir = "./enrichment_analysis_results"
cancer_term_file = './enrichment_analysis_results/cancer_related_terms_with_diseaseids.txt'
output_dir = './enrichment_analysis_results'


def barplot_enrichment_with_highlight(df, title, cancer_term_file, top_n=15, save_name=None, output_dir='./enrichment_final_results'):
    """
    Generates and saves a styled horizontal bar plot for enrichment results,
    highlighting cancer-related terms.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()

    rename_map = {
        'P_value': 'p_value',
        'Term_Name': 'name',
        'Gene_Count': 'intersection_size',
        'Term_ID': 'native'
    }
    df = df.rename(columns=rename_map)

    required_cols = ['p_value', 'name', 'intersection_size', 'native']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: The input DataFrame for '{title}' is missing a required column: '{col}' (or its variants).")
            return
    # -----------------------------------------------------------------

    # --- Data Preparation ---
    df['p_value'] = df['p_value'].clip(lower=1e-300)
    df['log10_pval'] = -np.log10(df['p_value'])
    df['name'] = df['name'].str.replace(r'\s*\(GO:\d+\)', '', regex=True)
    df['Count'] = df['intersection_size']
    df = df.sort_values('log10_pval', ascending=False).head(top_n)
    df = df[::-1]
    df['name'] = pd.Categorical(df['name'], categories=df['name'], ordered=True)

    # Load cancer-related term IDs for highlighting.
    try:
        with open(cancer_term_file, 'r', encoding='utf-8') as f:
            cancer_ids = set()
            for line in f:
                # This regex robustly finds IDs like 'GO:...' or 'KEGG:...'.
                match = re.search(r'ID:\s*([A-Z0-9:]+)', line)
                if match:
                    cancer_ids.add(match.group(1).strip())
    except FileNotFoundError:
        print(f"Warning: Cancer term file not found at '{cancer_term_file}'. No terms will be highlighted.")
        cancer_ids = set()

    df['highlight'] = df['native'].apply(lambda x: x in cancer_ids)

    # --- Visualization ---
    norm = Normalize(vmin=df['Count'].min(), vmax=df['Count'].max())
    cmap = plt.cm.plasma
    colors = cmap(norm(df['Count']))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df['name'], df['log10_pval'], color=colors, edgecolor='black')

    ax.set_yticks([])
    for i, (bar, name, highlight) in enumerate(zip(bars, df['name'], df['highlight'])):
        y = bar.get_y() + bar.get_height() / 2
        ax.text(-0.5, y, name, va='center', ha='right',
                fontsize=10, fontweight='bold' if highlight else 'normal')

    ax.set_xlabel('-log10(Adjusted p-value)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    cbar_ax = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gene Count', fontsize=11)

    if save_name is not None:
        save_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved figure: {save_path}")

    plt.close()


# --- Main Execution ---
try:
    # Load the enrichment data from CSV files.
    go_bp = pd.read_csv(os.path.join(enrichment_csv_dir, 'top_15_GO_BP.csv'))
    # go_cc = pd.read_csv(os.path.join(enrichment_csv_dir, 'top_15_GO_CC.csv'))
    # go_mf = pd.read_csv(os.path.join(enrichment_csv_dir, 'top_15_GO_MF.csv'))
    kegg = pd.read_csv(os.path.join(enrichment_csv_dir, 'top_15_KEGG.csv'))
except FileNotFoundError as e:
    print(f"Error: Could not find an input CSV file in '{enrichment_csv_dir}'.")
    print(f"Please ensure the directory and files exist. Details: {e}")
    exit()

# Generate a plot for each enrichment category.
print("Generating enrichment plots...")
barplot_enrichment_with_highlight(go_bp, 'GO Biological Process Enrichment', cancer_term_file, save_name='GO_BP', output_dir=output_dir)
# barplot_enrichment_with_highlight(go_cc, 'GO Cellular Component Enrichment', cancer_term_file, save_name='GO_CC', output_dir=output_dir)
# barplot_enrichment_with_highlight(go_mf, 'GO Molecular Function Enrichment', cancer_term_file, save_name='GO_MF', output_dir=output_dir)
barplot_enrichment_with_highlight(kegg, 'KEGG Pathway Enrichment', cancer_term_file, save_name='KEGG', output_dir=output_dir)
