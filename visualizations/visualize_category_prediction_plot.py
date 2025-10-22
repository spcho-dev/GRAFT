import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Define file and directory paths ---
output_dir = "./figures"
network_dirs = {
    'STRING': "./node_prediction_pancancer/prediction_genes_STRING",
    'CPDB': "./node_prediction_pancancer/prediction_genes_CPDB"
}
driver_path = "../Data/796true.txt"
nondriver_path = "../Data/2187false.txt"
potential_path = "./node_prediction_pancancer/potential_driver_genes_ncg7.2.txt"

# --- Load gene lists from text files ---
try:
    driver_genes = set(pd.read_csv(driver_path, header=None)[0].str.strip())
    nondriver_genes = set(pd.read_csv(nondriver_path, header=None)[0].str.strip())
    potential_genes = set(pd.read_csv(potential_path, header=None)[0].str.strip())
except FileNotFoundError as e:
    print(f"Error: Data file not found. Please check the path. Details: {e}")
    exit()

# --- Gene classification function ---
def classify_gene(gene):
    """Assigns a gene to a predefined category."""
    if gene in driver_genes:
        return 'Known driver genes'
    elif gene in potential_genes:
        return 'Potential driver genes'
    elif gene in nondriver_genes:
        return 'Non-driver genes'
    else:
        return 'Other genes'

# --- Setup for Plotting ---
# Define a fixed category order and color map for consistent plotting.
categories = ['Known driver genes', 'Potential driver genes', 'Non-driver genes', 'Other genes']
color_map = {
    'Known driver genes': '#A31621',
    'Potential driver genes': '#BF4E8A',
    'Non-driver genes': '#0B4F6C',
    'Other genes': '#556B2F'
}

# --- Process each network's prediction data ---
network_stats = {}
for network_name, pred_dir in network_dirs.items():
    if not os.path.isdir(pred_dir):
        print(f"Warning: Directory not found for {network_name}: {pred_dir}")
        continue

    # Load and merge predictions from all 10 folds.
    all_preds = []
    for i in range(1, 11):
        fold_file = os.path.join(pred_dir, f"predictions_fold_{i}.txt")
        if not os.path.exists(fold_file):
            print(f"Warning: File not found: {fold_file}")
            continue
        df = pd.read_csv(fold_file, sep="\t")
        df = df[['Gene', 'Predicted']]
        df.rename(columns={'Predicted': f'Predicted_{i}'}, inplace=True)
        all_preds.append(df)

    if not all_preds:
        print(f"No prediction data found for {network_name}.")
        continue

    # Merge fold data and calculate the mean prediction score.
    merged_df = all_preds[0]
    for df in all_preds[1:]:
        merged_df = pd.merge(merged_df, df, on='Gene')
    merged_df['mean_pred'] = merged_df[[f'Predicted_{i}' for i in range(1, 11)]].mean(axis=1)

    # Classify genes and calculate statistics for each category.
    merged_df['Category'] = merged_df['Gene'].apply(classify_gene)
    stats = merged_df.groupby('Category').agg(
        Mean_Prediction=('mean_pred', 'mean'),
        Std_Prediction=('mean_pred', 'std')
    ).reindex(categories).reset_index()

    network_stats[network_name] = stats

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.4
bar_spacing = 0.5
group_gap = 0.3

# Calculate bar positions for each network group.
x_pos_string = np.arange(len(categories)) * bar_spacing
cpdb_start_pos = (len(categories) * bar_spacing) + group_gap
x_pos_cpdb = (np.arange(len(categories)) * bar_spacing) + cpdb_start_pos

# Draw bars for STRING and CPDB networks.
if 'STRING' in network_stats:
    stats_string = network_stats['STRING']
    ax.bar(x_pos_string, stats_string['Mean_Prediction'],
           color=[color_map[cat] for cat in stats_string['Category']],
           capsize=5, width=width)

if 'CPDB' in network_stats:
    stats_cpdb = network_stats['CPDB']
    ax.bar(x_pos_cpdb, stats_cpdb['Mean_Prediction'],
           color=[color_map[cat] for cat in stats_cpdb['Category']],
           capsize=5, width=width)

# --- Style Axes and Labels ---
ax.set_xticks(list(x_pos_string) + list(x_pos_cpdb))
ax.set_xticklabels([]) # Hide individual bar ticks.

# Add primary labels for each network group (STRING, CPDB).
ax.text(np.mean(x_pos_string), -0.05, 'STRING', ha='center', va='top',
        fontsize=18, fontweight='bold', transform=ax.get_xaxis_transform())
ax.text(np.mean(x_pos_cpdb), -0.05, 'CPDB', ha='center', va='top',
        fontsize=18, fontweight='bold', transform=ax.get_xaxis_transform())

ax.set_ylabel('Predicted Score', fontsize=16, fontweight='bold')
ax.set_ylim(0, 1.0)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(11)

handles = [plt.Rectangle((0,0), 1, 1, color=color_map[cat]) for cat in categories]
ax.legend(handles, categories, fontsize=15.5)

plt.tight_layout()

# --- Save Figure ---
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, 'predicted_gene_category.jpg')
plt.savefig(
    output_filename,
    format='jpg',
    dpi=800,
    quality=100,
    bbox_inches='tight'
)
print(f"Figure successfully saved to '{output_filename}'")
plt.show()
