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
from model import GRAFT
from utils import preprocessing_incidence_matrix, extract_edge_data_with_score, load_kfold_data, load_label_single, stratified_kfold_split
import random
import gc


# Hyperparameter and setting
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


# Loss Function
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


# Data input
dataPath = f"./Data/{args.dataset}"
# load new multi-omics feature 
data_x_df = pd.read_csv(dataPath + f'/multiomics_features_{args.dataset}.tsv', sep='\t', index_col=0)
data_x_df = data_x_df.dropna()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_x_df.values)
data_x = torch.tensor(features_scaled, dtype=torch.float32, device=device)
data_x = data_x[:,:48]

cancerType = args.cancerType.lower()

if cancerType=='pan-cancer':
    data_x = data_x[:,:48]
    print("--- [INFO] Loading hyperparameters for Pan-Cancer ---")
    learning_rate = 0.001
    epochs = 30
    num_heads = 4
    num_layers = 3
    dropout = 0.1
    
else:
    cancerType_dict = {
                        'kirc':[0,16,32],
                        'brca':[1,17,33],
                        'prad':[3,19,35],
                        'stad':[4,20,36],
                        'hnsc':[5,21,37],
                        'luad':[6,22,38],
                        'thca':[7,23,39],
                        'blca':[8,24,40],
                        'esca':[9,25,41],
                        'lihc':[10,26,42],
                        'ucec':[11,27,43],
                        'coad':[12,28,44],
                        'lusc':[13,29,45],
                        'cesc':[14,30,46],
                        'kirp':[15,31,47]
                                }
    data_x = data_x[:, cancerType_dict[cancerType]]
    print(f"===== [INFO] Loading hyperparameters for Specific Cancer: {cancerType.upper()} =====")
    learning_rate = 1e-4
    epochs = 50
    num_heads = 2
    num_layers = 2
    dropout = 0.2

print(f"Applied Hyperparameters: Learning Rate={learning_rate}, Epochs={epochs}, Heads={num_heads}, Layers={num_layers}, Dropout={dropout}")

node_features = data_x  # torch.Tensor, [N, 48]
ppiAdj = torch.load(dataPath+f'/{args.dataset}_ppi.pkl')
pathAdj = torch.load(dataPath+'/pathway_SimMatrix_filtered.pkl')
goAdj = torch.load(dataPath+'/GO_SimMatrix_filtered.pkl')

# Extract edge indices (row, col) and corresponding edge scores from adjacency matrices
ppi_row, ppi_col, ppi_score = extract_edge_data_with_score(ppiAdj)
path_row, path_col, path_score = extract_edge_data_with_score(pathAdj)
go_row, go_col, go_score = extract_edge_data_with_score(goAdj)

# Dictionary storing edge index and score tuples for each biological network type
# Used later by EdgeImportanceEncoder to incorporate edge confidence into attention bias
edge_indices_with_score = {
    "ppi": (ppi_row, ppi_col, ppi_score),     # PPI with confidence
    "path": (path_row, path_col, path_score),  # Pathway co-occurrence
    "go": (go_row, go_col, go_score)         # gene semantic similarity
}

# Dictionary storing only edge indices [2, num_edges] for each network type
# Used by TripleGNNFeatureExtractor to perform GCN message passing per network
edge_index_dict = {
    'ppi': torch.stack([ppi_row, ppi_col], dim=0).to(device),     # [2, num_edges]
    'path': torch.stack([path_row, path_col], dim=0).to(device),
    'go': torch.stack([go_row, go_col], dim=0).to(device),
}


# Gene set matrix
msigdb_genelist = pd.read_csv('./Data/msigdb/geneList.csv', header=None)
msigdb_genelist = list(msigdb_genelist[0].values)
incidence_matrix = preprocessing_incidence_matrix(msigdb_genelist, dataPath)
gene_set_matrix = torch.tensor(incidence_matrix.values, dtype=torch.float32, device=device)


# Random Walk Positional Encoding & PageRank Centrality
torch_ppi_dense = ppiAdj.to_dense().cpu().numpy()
ppi_sp = sp.csr_matrix(torch_ppi_dense)

# PageRank Centrality
G = nx.from_scipy_sparse_matrix(ppi_sp)
pagerank_dict = nx.pagerank(G, alpha=0.85)
pagerank_vec = torch.tensor([pagerank_dict[i] for i in range(len(pagerank_dict))], dtype=torch.float32, device=device).unsqueeze(1)  # [N, 1]

# Random Walk Positional Encoding
deg_row = ppi_sp.sum(axis=1).A1
deg_row[deg_row == 0] = 1.0
rw_mat = ppi_sp.multiply(1.0 / deg_row[:, None])
pca = PCA(n_components=args.embed_dim)  # Converting to dimensionalized tensors via PCA
rw_pe_pca = pca.fit_transform(rw_mat.toarray()) # [N, N] -> [N, embed_dim]
rw_pe = torch.tensor(rw_pe_pca, dtype=torch.float32, device=device)


# Model Train & Test
cross_val=10    # 10 fold
AUC = np.zeros(shape=(cross_val))
AUPR = np.zeros(shape=(cross_val))
F1_SCORES = np.zeros(cross_val)

if cancerType != 'pan-cancer':
    num_folds = cross_val
    path = f"{dataPath}/dataset/specific-cancer/"
    label_new, label_pos, label_neg = load_label_single(path, cancerType, device)
    random.shuffle(label_pos)
    random.shuffle(label_neg)
    l = len(label_new)
    l1 = int(len(label_pos)/num_folds)
    l2 = int(len(label_neg)/num_folds)
    folds = stratified_kfold_split(label_pos, label_neg, l, l1, l2)
    Y = label_new

print(f"----- {cancerType.upper()} 10-fold validation ------")
for i in range(cross_val):
    print(f'--------- Fold {i+1} Begin ---------')
    
    if cancerType == 'pan-cancer':
        fold_path = f"{dataPath}/10fold/fold_{i+1}"
        train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, Y = load_kfold_data(fold_path, device)
    else:
        train_idx, valid_idx, test_idx, train_mask, val_mask, test_mask = folds[i]
    
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
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)  # shape: [N]
        loss = loss_fn(logits[train_idx], Y[train_idx])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_logits = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
            pred_probs = torch.sigmoid(pred_logits)

            val_auc = roc_auc_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
            val_aupr = average_precision_score(Y[valid_idx].cpu(), pred_probs[valid_idx].cpu())
            print(f"[Fold {i+1}] Epoch {epoch+1} | Val AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")

    model.eval()
    with torch.no_grad():
        final_logits = model(node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, rw_pe, pagerank_vec)
        final_probs = torch.sigmoid(final_logits)
        pred_labels = (final_probs > 0.5).float()

        AUC[i] = roc_auc_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
        AUPR[i] = average_precision_score(Y[test_idx].cpu(), final_probs[test_idx].cpu())
        F1_SCORES[i] = f1_score(Y[test_idx].cpu(), pred_labels[test_idx].cpu())
    
    print(f"Fold {i+1} Results — AUC: {AUC[i]:.3f}, AUPR: {AUPR[i]:.3f}, F1: {F1_SCORES[i]:.3f}")

    with open(f"./final_results_{args.dataset}_{cancerType.upper()}.txt", "a") as result_file:
        result_file.write(f"Fold {i+1}: AUC={AUC[i]:.3f}, AUPR={AUPR[i]:.3f}, F1-score={F1_SCORES[i]:.3f}\n")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
print("========== Final 10-Fold Results ==========")
print(f"Mean AUC: {AUC.mean():.3f} ± {AUC.std():.3f}")
print(f"Mean AUPR: {AUPR.mean():.3f} ± {AUPR.std():.3f}")
print(f"Mean F1: {F1_SCORES.mean():.3f} ± {F1_SCORES.std():.3f}")

with open(f"./final_results_{args.dataset}_{cancerType.upper()}.txt", "a") as result_file:
    result_file.write(f"\nFinal Results:\nMean AUC: {AUC.mean():.3f} ± {AUC.std():.3f}\nMean AUPR: {AUPR.mean():.3f} ± {AUPR.std():.3f}\nMean F1-score: {F1_SCORES.mean():.3f} ± {F1_SCORES.std():.3f}\n\n")

