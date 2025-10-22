import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import scipy.sparse as sp
import networkx as nx
import numpy as np
import pandas as pd
import argparse
# Ensure you are importing the modified model that can return embeddings and attention
from model_with_outputs import GRAFT
from utils import preprocessing_incidence_matrix, extract_edge_data_with_score, load_kfold_data, load_label_single, stratified_kfold_split
import random
import gc
import os

# --- Hyperparameter and Setting ---
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset (CPDB, STRING)')
parser.add_argument('cancerType', type=str, help='Types of cancer (pan-cancer, kirc)')
parser.add_argument('--embed_dim', type=int, default=128, help='embedding dimension')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--device', type=int, default=0, help='GPU device ID (if available)')
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Determine the single fold index to execute (adjusting to 0-based index - 0~9)
fold_index_to_run = 9
if not (0 <= fold_index_to_run < 10):
    print(f"Error: fold_to_run must be between 1 and 10 (index 0 and 9)")
    exit()

# --- Loss Function ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        loss = self.alpha * ((1 - pt) ** self.gamma) * BCE
        return loss.mean()


# --- Data Input ---
dataPath = f"../Data/{args.dataset}"
try:
    data_x_df = pd.read_csv(dataPath + f'/multiomics_features_{args.dataset}.tsv', sep='\t', index_col=0)
except FileNotFoundError:
    print(f"Error: Multi-omics features file not found at {dataPath}/multiomics_features_{args.dataset}.tsv")
    exit()
data_x_df = data_x_df.dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_x_df.values)
data_x = torch.tensor(features_scaled, dtype=torch.float32, device=device)
data_x = data_x[:,:48] # Keep only first 48 features initially

cancerType = 'pan-cancer'

# --- Load Hyperparameters and Select Features based on Cancer Type ---
if cancerType=='pan-cancer':
    data_x = data_x[:,:48] # Use all 48 features
    print("--- [INFO] Loading hyperparameters for Pan-Cancer ---")
    learning_rate = 0.001
    epochs = 30
    num_heads = 4
    num_layers = 3
    dropout = 0.1
else:
    cancerType_dict = {
                         'kirc':[0,16,32], 'brca':[1,17,33], 'prad':[3,19,35],
                         'stad':[4,20,36], 'hnsc':[5,21,37], 'luad':[6,22,38],
                         'thca':[7,23,39], 'blca':[8,24,40], 'esca':[9,25,41],
                         'lihc':[10,26,42], 'ucec':[11,27,43], 'coad':[12,28,44],
                         'lusc':[13,29,45], 'cesc':[14,30,46], 'kirp':[15,31,47]
                       }
    if cancerType not in cancerType_dict:
        print(f"Error: Unknown cancer type '{cancerType}'.")
        exit()
    data_x = data_x[:, cancerType_dict[cancerType]] # Select specific features
    print(f"===== [INFO] Loading hyperparameters for Specific Cancer: {cancerType.upper()} =====")
    learning_rate = 1e-4
    epochs = 50
    num_heads = 2
    num_layers = 2
    dropout = 0.2

print(f"Applied Hyperparameters: Learning Rate={learning_rate}, Epochs={epochs}, Heads={num_heads}, Layers={num_layers}, Dropout={dropout}")

# --- Load Network Data and Preprocess ---
node_features = data_x
try:
    ppiAdj = torch.load(dataPath+f'/{args.dataset}_ppi.pkl')
    pathAdj = torch.load(dataPath+'/pathway_SimMatrix_filtered.pkl')
    goAdj = torch.load(dataPath+'/GO_SimMatrix_filtered.pkl')
except FileNotFoundError as e:
    print(f"Error loading network pkl file: {e}. Please ensure data files exist.")
    exit()

ppi_row, ppi_col, ppi_score = extract_edge_data_with_score(ppiAdj)
path_row, path_col, path_score = extract_edge_data_with_score(pathAdj)
go_row, go_col, go_score = extract_edge_data_with_score(goAdj)

edge_indices_with_score = {
    "ppi": (ppi_row, ppi_col, ppi_score),
    "path": (path_row, path_col, path_score),
    "go": (go_row, go_col, go_score)
}
edge_index_dict = {
    'ppi': torch.stack([ppi_row, ppi_col], dim=0).to(device),
    'path': torch.stack([path_row, path_col], dim=0).to(device),
    'go': torch.stack([go_row, go_col], dim=0).to(device),
}

try:
    msigdb_genelist = pd.read_csv('./Data/msigdb/geneList.csv', header=None)
    msigdb_genelist = list(msigdb_genelist[0].values)
    incidence_matrix = preprocessing_incidence_matrix(msigdb_genelist, dataPath)
    gene_set_matrix = torch.tensor(incidence_matrix.values, dtype=torch.float32, device=device)
except FileNotFoundError as e:
    print(f"Error loading gene list or processing incidence matrix: {e}")
    exit()

# --- Calculate PE and Centrality ---
torch_ppi_dense = ppiAdj.to_dense().cpu().numpy()
ppi_sp = sp.csr_matrix(torch_ppi_dense)

G = nx.from_scipy_sparse_matrix(ppi_sp)
pagerank_dict = nx.pagerank(G, alpha=0.85)
pagerank_vec = torch.tensor([pagerank_dict.get(i, 0) for i in range(node_features.shape[0])], dtype=torch.float32, device=device).unsqueeze(1)

deg_row = ppi_sp.sum(axis=1).A1
deg_row[deg_row == 0] = 1.0
rw_mat = ppi_sp.multiply(1.0 / deg_row[:, None])
pca = PCA(n_components=args.embed_dim)
rw_pe_pca = pca.fit_transform(rw_mat.toarray())
rw_pe = torch.tensor(rw_pe_pca, dtype=torch.float32, device=device)


# --- Model Training & Evaluation ---
attn_info = None
node_embeddings = None

if cancerType == 'pan-cancer':
    fold_path = f"{dataPath}/10fold/fold_{fold_index_to_run+1}"
    try:
        train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, Y = load_kfold_data(fold_path, device)
    except FileNotFoundError:
        print(f"Error: Fold data not found at {fold_path}")
        exit()
else: # Specific cancer
    try:
        path = f"{dataPath}/dataset/specific-cancer/"
        label_new, label_pos, label_neg = load_label_single(path, cancerType, device)
        random.shuffle(label_pos)
        random.shuffle(label_neg)
        l = len(label_new)
        l1 = int(len(label_pos)/10)
        l2 = int(len(label_neg)/10)
        folds = stratified_kfold_split(label_pos, label_neg, l, l1, l2)
        train_idx, valid_idx, test_idx, train_mask, val_mask, test_mask = folds[fold_index_to_run]
        Y = label_new
    except FileNotFoundError as e:
         print(f"Error loading specific cancer label data: {e}")
         exit()

model = GRAFT(
    input_dim=node_features.shape[1],
    gene_set_dim=gene_set_matrix.shape[1],
    embed_dim=args.embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_fn = FocalLoss(alpha=1.0, gamma=1.5)

# --- Training Loop ---
print(f'--------- Training ---------')
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    logits, _ = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
    loss = loss_fn(logits[train_idx], Y[train_idx])
    loss.backward()
    optimizer.step()

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        pred_logits, _ = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
        pred_probs = torch.sigmoid(pred_logits)
        val_auc = roc_auc_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
        val_aupr = average_precision_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
        print(f"Epoch {epoch+1}/{epochs} | Val AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")

# --- Final Evaluation and Data Capture ---
print(f'--------- Evaluating and Capturing Data ---------')
model.eval()
with torch.no_grad():
    final_logits, attn_weight = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
    final_probs = torch.sigmoid(final_logits)
    pred_labels = (final_probs > 0.5).float()

    test_auc = roc_auc_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
    test_aupr = average_precision_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
    test_f1 = f1_score(Y[test_idx].cpu(), pred_labels[test_idx].cpu())

    print(f"\n Final Test Results:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  AUPR: {test_aupr:.4f}")
    print(f"  F1: {test_f1:.4f}")

    # Capture attention info
    attn_info = {
        "attn_weight": attn_weight.detach().cpu(),
        "label": Y.detach().cpu(),
        "test_idx": torch.tensor(test_idx)
    }

    # Capture node embeddings
    node_embed = model.get_node_embeddings(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
    df_embed = pd.DataFrame(node_embed.cpu().numpy())
    df_embed['true_label'] = Y.cpu().numpy()
    df_embed['pred_prob'] = final_probs.cpu().numpy()
    df_embed['pred_label'] = pred_labels.cpu().numpy()
    df_embed['gene'] = data_x_df.index
    node_embeddings = df_embed


# --- Save data for visualization scripts ---
output_base_dir = "./node_prediction"

# 1. Save Attention Info
if attn_info:
    output_dir = output_base_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"attn_info.pt")
    print(f"\n--- Saving attention info to '{output_path}' ---")
    # Save as a list containing the single fold's data
    torch.save([attn_info], output_path)
    print("Attention info saved.")

# 2. Save Node Embeddings
if node_embeddings is not None:
    output_dir = output_base_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"node_embeddings.csv")
    print(f"\n--- Saving node embeddings for '{output_path}' ---")
    node_embeddings.to_csv(output_path, index=False)
    print("Node embeddings saved.")

print("\n--- Script finished ---")