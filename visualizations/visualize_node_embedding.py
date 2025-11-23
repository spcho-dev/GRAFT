import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for saving images
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
data_file = "./node_prediction/node_embeddings.csv"
output_dir = "./figures"
output_filename = os.path.join(output_dir, "umap_visualization.jpg")

# --- Load Data ---
try:
    df = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: Data file not found at '{data_file}'")
    print("Please ensure the data file is in the same directory as the script.")
    exit()

# --- Preprocessing ---
embedding_cols = [col for col in df.columns if col not in ['true_label', 'pred_prob', 'pred_label', 'gene']]
embedding_data = StandardScaler().fit_transform(df[embedding_cols].values)

# --- UMAP Dimensionality Reduction ---
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)
umap_result = reducer.fit_transform(embedding_data)
df['umap_x'] = umap_result[:, 0]
df['umap_y'] = umap_result[:, 1]

# --- Visualization ---
plt.figure(figsize=(6, 5))
colors = {0: '#1f77b4', 1: '#d62728'}  # blue=non-driver, red=driver
labels = {0: "Non-driver", 1: "Driver"}

# Plot Non-drivers
subset_neg = df[df['true_label'] == 0]
plt.scatter(subset_neg['umap_x'], subset_neg['umap_y'],
            s=2,
            alpha=0.8,
            color=colors[0],
            label=labels[0],
            zorder=1)

# Plot Drivers
subset_pos = df[df['true_label'] == 1]
plt.scatter(subset_pos['umap_x'], subset_pos['umap_y'],
            s=3,
            alpha=0.8,
            color=colors[1],
            label=labels[1],
            zorder=2)

# --- Style Plot ---
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.grid(False)

# Reorder legend
handles, legend_labels = plt.gca().get_legend_handles_labels()
try:
    order = [legend_labels.index("Driver"), legend_labels.index("Non-driver")]
    plt.legend([handles[i] for i in order], [legend_labels[i] for i in order],
               loc='best',
               markerscale=3,
               frameon=False,
               fontsize=12)
except ValueError:
    plt.legend(title='Gene Type', loc='best', markerscale=3, frameon=False, fontsize=12)


plt.tight_layout()

# --- Save Figure ---
os.makedirs(output_dir, exist_ok=True)
plt.savefig(
    output_filename,
    format='jpg',
    dpi=600,
    bbox_inches='tight'
)
plt.close()

print(f"UMAP visualization successfully saved to '{output_filename}'")