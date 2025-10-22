import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist
import os

# --- Configuration ---
data_file = "./node_prediction_pancancer/node_embeddings.csv"
output_dir = "./figures"
output_filename = os.path.join(output_dir, "kde_distance_plot.jpg")

# --- Load Data ---
try:
    df = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: Data file not found at '{data_file}'")
    print("Please ensure the data file is in the same directory as the script.")
    exit()

# --- Data Processing ---
# Extract embedding columns (all columns except metadata)
embedding_cols = [col for col in df.columns if col not in ['true_label', 'pred_prob', 'pred_label', 'fold', 'gene']]
# Standardize the embedding data
embedding_data = StandardScaler().fit_transform(df[embedding_cols].values)

# Get indices for driver and non-driver genes
idx_driver = df[df['true_label'] == 1].index.to_list()
idx_non_driver = df[df['true_label'] == 0].index.to_list()

# Separate the embeddings
emb_driver = embedding_data[idx_driver]
emb_non_driver = embedding_data[idx_non_driver]

# --- Calculate Pairwise Cosine Distances ---
# (pdist = intra-group, cdist = inter-group)
dist_driver_driver = pdist(emb_driver, metric='cosine')
dist_non_non = pdist(emb_non_driver, metric='cosine')
dist_driver_non = cdist(emb_driver, emb_non_driver, metric='cosine').flatten()

# --- Visualization ---
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))

plot_data = [
    (dist_driver_driver, "Driver–Driver", '#e41a1c'),
    (dist_non_non, "Non-driver–Non-driver", '#377eb8'),
    (dist_driver_non, "Driver–Non-driver", 'gray')
]

for dist, label, color in plot_data:
    if len(dist) > 0:
        kde = sns.kdeplot(dist, label=label, color=color, linewidth=2)
        x, y = kde.get_lines()[-1].get_data()
        plt.fill_between(x, y, alpha=0.3, color=color)

plt.xlabel("Cosine Distance")
plt.ylabel("Density")
# plt.title("Pairwise Embedding Distance (KDE)")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()

# --- Save Figure ---
os.makedirs(output_dir, exist_ok=True)
plt.savefig(
    output_filename,
    format='jpg',
    dpi=300,
    quality=95,
    bbox_inches='tight'
)
plt.close()

print(f"Filled KDE plot successfully saved to '{output_filename}'")