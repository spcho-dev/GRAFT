import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import os

# --- Configuration ---
# Input data file containing attention info.
data_file = "./attn_info.pt"
output_dir = "./figures"
output_filename = os.path.join(output_dir, 'clustermap_attention.jpg')

# --- Data Loading ---
try:
    # The .pt file is a list containing a single dictionary with attention weights and test indices.
    attn_info_list = torch.load(data_file)
except FileNotFoundError:
    print(f"Error: Data file not found at '{data_file}'")
    print("Please ensure the data file is in the same directory as the script.")
    exit()

# This script assumes the .pt file contains data in a list.
data = attn_info_list[0]

# --- Data Processing ---
# Extract attention weights and labels for the test set.
attn_weights = data['attn_weight'].squeeze(-1)  # Shape: [N, 3]
test_idx = data['test_idx']
test_attn_weights = attn_weights[test_idx]
test_labels = data['label'][test_idx]

# Convert to DataFrame and standardize the weights for visualization.
attn_df = pd.DataFrame(test_attn_weights.numpy(), columns=["PPI", "Pathway", "GO"])
attn_scaled = StandardScaler().fit_transform(attn_df)
attn_scaled_df = pd.DataFrame(attn_scaled, columns=["PPI", "Pathway", "GO"])

# Clip outliers to a range of [-3, 3] for a more stable and readable clustermap.
attn_clipped = attn_scaled_df.clip(lower=-3, upper=3)

# To ensure the plot is readable and renders quickly, sample up to 10,000 genes.
num_samples = min(10000, len(attn_clipped))
subset_idx = np.random.choice(len(attn_clipped), size=num_samples, replace=False)
attn_subset = attn_clipped.iloc[subset_idx]

# Create row colors for the clustermap sidebar: red for drivers, blue for non-drivers.
labels_subset = test_labels.numpy()[subset_idx]
row_colors = ['#A31621' if label == 1 else '#0B4F6C' for label in labels_subset]


# --- Visualization ---
# Generate the clustermap using seaborn.
g = sns.clustermap(
    attn_subset,
    method='ward',
    metric='euclidean',
    cmap="YlGnBu",
    figsize=(8, 10),
    linewidths=0.5,
    linecolor='white',
    xticklabels=True,
    yticklabels=False,
    vmin=-1.5,
    vmax=1.5,
    row_colors=row_colors
)

# Style the x-axis labels.
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(),
    fontsize=14,
    fontweight='bold'
)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0, ha='center', va='top')

# Style the dendrogram (clustering) lines to be darker and thicker.
for line in g.ax_row_dendrogram.lines:
    line.set_color('black')
    line.set_linewidth(1.0)
for line in g.ax_col_dendrogram.lines:
    line.set_color('black')
    line.set_linewidth(1.0)

# Add a custom legend for the row colors (Driver/Non-driver).
# A new axes is created to position the legend outside the main plot.
legend_patches = [
    mpatches.Patch(color='#A31621', label='Driver'),
    mpatches.Patch(color='#0B4F6C', label='Non-driver')
]
legend_ax = g.fig.add_axes([0.75, 0.92, 0.1, 0.1])
legend_ax.legend(
    handles=legend_patches,
    loc="upper left",
    fontsize=17
)
legend_ax.axis("off")


# --- Save Figure ---
# Create the output directory if it does not exist.
os.makedirs(output_dir, exist_ok=True)

# Save the figure in high quality.
g.savefig(
    output_filename,
    format='jpg',
    dpi=600,
    quality=100,
    bbox_inches='tight'
)
print(f"\n✅ Figure successfully saved to '{output_filename}'")
plt.show()
