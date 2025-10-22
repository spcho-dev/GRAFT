import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TripleGNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.ppi_gcn1 = GCNConv(input_dim, hidden_dim)
        self.ppi_gcn2 = GCNConv(hidden_dim, out_dim)
        self.path_gcn1 = GCNConv(input_dim, hidden_dim)
        self.path_gcn2 = GCNConv(hidden_dim, out_dim)
        self.go_gcn1 = GCNConv(input_dim, hidden_dim)
        self.go_gcn2 = GCNConv(hidden_dim, out_dim)
        self.attn_vector = nn.Linear(out_dim, 1)

    def forward(self, x, edge_index_dict):
        ppi_feat = self.ppi_gcn2(F.relu(self.ppi_gcn1(x, edge_index_dict['ppi'])), edge_index_dict['ppi'])
        path_feat = self.path_gcn2(F.relu(self.path_gcn1(x, edge_index_dict['path'])), edge_index_dict['path'])
        go_feat = self.go_gcn2(F.relu(self.go_gcn1(x, edge_index_dict['go'])), edge_index_dict['go'])

        all_feat = torch.stack([ppi_feat, path_feat, go_feat], dim=1)
        attn_weight = torch.softmax(self.attn_vector(all_feat), dim=1)
        weighted_feat = (all_feat * attn_weight).sum(dim=1)
        return weighted_feat, attn_weight

class EdgeImportanceEncoder(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.linear = nn.Linear(3, num_heads)

    def forward(self, edge_indices_with_score, num_heads, N, device):
        edge_bias = torch.zeros((num_heads, N, N), device=device)
        for _, (row, col, score) in edge_indices_with_score.items():
            row, col, score = row.to(device), col.to(device), score.to(device)
            features = torch.stack([row.float(), col.float(), score], dim=1)
            bias = torch.sigmoid(self.linear(features))
            for h in range(num_heads):
                edge_bias[h, row, col] += bias[:, h]
        return edge_bias

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class GRAFT(nn.Module):
    def __init__(self, input_dim, gene_set_dim, embed_dim=128, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.edge_encoder = EdgeImportanceEncoder(num_heads)
        self.gnn_encoder = TripleGNNFeatureExtractor(input_dim, embed_dim, embed_dim)
        self.gene_set_attention = nn.Linear(gene_set_dim, gene_set_dim)
        self.gene_set_embedding = nn.Parameter(torch.randn(gene_set_dim, embed_dim))
        self.input_proj = nn.Linear(input_dim + embed_dim * 3 + 1, embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, 1)

    def _get_embeddings_before_pred(self, node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, pos_enc, centrality_vec):
        """Helper function to get node embeddings and GNN attention weights."""
        N = node_features.size(0)
        device = node_features.device
        attn_bias = self.edge_encoder(edge_indices_with_score, self.num_heads, N, device)
        gnn_feat, attn_weight = self.gnn_encoder(node_features, edge_index_dict)
        attn_weights_gs = torch.softmax(self.gene_set_attention(gene_set_matrix), dim=-1)
        gene_set_embed = torch.matmul(attn_weights_gs, self.gene_set_embedding)
        x_concat = torch.cat([node_features, gnn_feat, gene_set_embed, pos_enc, centrality_vec], dim=-1)
        x = self.input_proj(x_concat)
        x = (x + pos_enc).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, attn_bias)
        return x.squeeze(0), attn_weight

    def forward(self, node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, pos_enc, centrality_vec):
        x, attn_weight = self._get_embeddings_before_pred(
            node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, pos_enc, centrality_vec
        )
        logits = self.out_proj(x).squeeze(-1)
        return logits, attn_weight

    def get_node_embeddings(self, node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, pos_enc, centrality_vec):
        """Returns the final node embeddings before the prediction layer."""
        x, _ = self._get_embeddings_before_pred(
            node_features, gene_set_matrix, edge_indices_with_score, edge_index_dict, pos_enc, centrality_vec
        )
        return x

