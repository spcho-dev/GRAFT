import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# --- Configuration ---
# Input data file containing attention info.
data_file = "./node_prediction/attn_info.pt"
output_dir = "./figures"
output_filename = os.path.join(output_dir, 'avg_attention_weight.jpg')

# --- Data Loading ---
try:
    # The .pt file is a list containing a single dictionary with attention weights and test indices.
    attn_info_list = torch.load(data_file)
except FileNotFoundError:
    print(f"Error: Data file not found at '{data_file}'")
    print("Please ensure the data file is in the same directory as the script.")
    exit()

# --- Data Processing ---
# Extract attention weights for the test samples from the loaded data.
attn_weights_list = []
for data in attn_info_list:
    attn = data['attn_weight'].squeeze(-1)
    test_idx = data['test_idx']
    attn_weights_list.append(attn[test_idx])

# Concatenate results into a single NumPy array [num_test_samples, 3].
attn_weights_all = torch.cat(attn_weights_list, dim=0).numpy()

# Calculate the mean and standard deviation across all test samples for each network type.
mean_weights = np.mean(attn_weights_all, axis=0)
std_weights = np.std(attn_weights_all, axis=0)

print("Attention Weights (based on a single fold):")
print(f"  - Mean: {mean_weights}")
print(f"  - Std Dev: {std_weights}")

# --- Visualization ---
labels = ["PPI", "Pathway", "GO"]
colors = ["#4c72b0", "#55a868", "#c44e52"]

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(7, 6))

# Create bar plot with error bars.
bars = ax.bar(labels, mean_weights, yerr=std_weights, capsize=6,
              color=colors, edgecolor='black', width=0.6,
              error_kw=dict(lw=1.5, ecolor='black', capthick=1.5))

# Adjust y-axis limit for better spacing.
max_y = max(mean_weights + std_weights)
ax.set_ylim(0, max_y + 0.05)

# Annotate each bar with its mean value.
for bar, val, err in zip(bars, mean_weights, std_weights):
    ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.01,
            f"{val:.2f}", ha='center', va='bottom', fontsize=15, fontweight='bold')

ax.set_ylabel("Average Attention Weight", fontsize=14, fontweight='bold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=15, fontweight='bold')

for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(12)

ax.grid(axis='y', linestyle='--', linewidth=0.7, color='black', alpha=0.5)
ax.set_axisbelow(True)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# --- Save Figure ---
os.makedirs(output_dir, exist_ok=True)

plt.savefig(
    output_filename,
    format='jpg',
    dpi=600,
    quality=100,
    bbox_inches='tight'
)
print(f"\nFigure successfully saved to '{output_filename}'")
plt.show()

