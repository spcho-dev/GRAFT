import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATConv
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

def one_hot(labels, n_classes=2):
    return torch.eye(n_classes)[labels, :]

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        u = torch.matmul(q, k.transpose(-2, -1))
        u = u / self.scale
        if mask is not None:
            u = u.masked_fill(mask, -np.inf)
        attn = self.softmax(u)
        output = torch.matmul(attn, v)
        return attn, output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)
        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v
        n_q, d_q_ = q.size()
        n_k, d_k_ = k.size()
        n_v, d_v_ = v.size()
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(n_q, n_head, d_q).permute(1, 0, 2).contiguous().view(-1, n_q, d_q)
        k = k.view(n_k, n_head, d_k).permute(1, 0, 2).contiguous().view(-1, n_k, d_k)
        v = v.view(n_v, n_head, d_v).permute(1, 0, 2).contiguous().view(-1, n_v, d_v)
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, n_q, d_v).permute(0, 1, 2).contiguous().view(n_q, -1)
        output = self.fc_o(output)
        return attn, output

class SelfAttention(nn.Module):
    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))
        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)
        attn, output = self.mha(q, k, v, mask=mask)
        return output

class Gat_En(nn.Module):
    def __init__(self, nfeat, hidden_size, out, dropout):
        super(Gat_En, self).__init__()
        self.gat1 = GATConv(nfeat, hidden_size, heads=3, dropout=dropout)
        self.gat2 = GATConv(3*hidden_size, out, heads=1, concat=True, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        return x

class MODIG(nn.Module):
    def __init__(self, nfeat, hidden_size1, hidden_size2, dropout):
        super(MODIG, self).__init__()
        self.view_GNN = Gat_En(nfeat, hidden_size1, hidden_size2, dropout)
        self.self_attn = SelfAttention(n_head=1, d_k=64, d_v=32, d_x=hidden_size2, d_o=hidden_size2)
        self.attn = Attention(hidden_size2)
        self.MLP = nn.Linear(hidden_size2, 1)
        self.dropout = dropout

    def forward(self, graphs):
        embs = []
        for i in range(len(graphs)):
            emb = self.view_GNN(graphs[i])
            embs.append(emb)
        fused_outs = []
        for emb in embs:
            outs = self.self_attn(emb)
            fused_outs.append(outs)
        alpha = 0.6
        embs2 = []
        for i in range(len(embs)):
            emb2 = alpha * fused_outs[i] + (1 - alpha) * embs[i]
            embs2.append(emb2)
        emb_f, atts = self.attn(torch.stack(embs2, dim=1))
        output = self.MLP(emb_f)
        return output
    
class ModigGraph(object):
    def __init__(self, graph_path, ppi_type, cancer_type):
        self.graph_path = graph_path
        self.ppi_type = ppi_type
        self.cancer_type = cancer_type

    def get_node_genelist(self):
        print('Get gene list')
        # gene info file path check
        gene = pd.read_csv("./Data/simmatrix/gene_info_for_GOSemSim.csv")
        gene_list = list(set(gene['Symbol']))
        ppi = pd.read_csv(os.path.join('./Data/ppi', 'STRING_ppi_edgelist.tsv'), sep='\t',
                          encoding='utf8', usecols=['partner1', 'partner2'])
        ppi.columns = ['source', 'target']
        ppi.dropna(inplace=True)
        final_gene_node = sorted(list(set(gene_list) | set(ppi.source) | set(ppi.target)))
        
        # Build PPI Network
        G = nx.from_pandas_edgelist(ppi)
        # Handle disconnected nodes by creating a full adjacency with all genes
        # Note: This might be memory intensive. If pre-calculated files exist, they should be used.
        # For reproduction, we assume graph files are already generated in main run.
        return final_gene_node, None # Returning None for ppi_final as it's loaded from file

    def get_node_omicfeature(self):
        final_gene_node, _ = self.get_node_genelist()
        omics_file = pd.read_csv('./Data/feature/multiomics_features_SNV_METH_GE_CNA_filtered_STRING.tsv', sep='\t', index_col=0)
        expendgene = sorted(list(set(omics_file.index) | set(final_gene_node)))
        temp = pd.DataFrame(index=expendgene, columns=omics_file.columns)
        omics_adj = temp.combine_first(omics_file)
        omics_adj.fillna(0, inplace=True)
        omics_adj = omics_adj.loc[final_gene_node]
        omics_adj.sort_index(inplace=True)

        if self.cancer_type != 'pancan':
            omics_data = omics_adj[omics_adj.columns[omics_adj.columns.str.contains(self.cancer_type)]]
        elif self.cancer_type == 'pancan':
            chosen_project = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD',
                              'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']
            omics_temp = [omics_adj[omics_adj.columns[omics_adj.columns.str.contains(cancer)]] for cancer in chosen_project]
            omics_data = pd.concat(omics_temp, axis=1)

        omics_feature_vector = sp.csr_matrix(omics_data, dtype=np.float32)
        omics_feature_vector = torch.FloatTensor(np.array(omics_feature_vector.todense()))
        return omics_feature_vector, final_gene_node

    def load_featured_graph(self, network, omicfeature):
        omics_feature_vector = sp.csr_matrix(omicfeature, dtype=np.float32)
        omics_feature_vector = torch.FloatTensor(np.array(omics_feature_vector.todense()))
        
        if isinstance(network, pd.DataFrame):
             # If index is gene names, need to convert to integer index matching omicfeature order
             # Assuming network is already sorted by final_gene_node from main pre-processing
             G = nx.from_pandas_adjacency(network)
        else:
             G = nx.from_pandas_edgelist(network)

        # Convert node labels to integers (0 to N-1)
        G_adj = nx.convert_node_labels_to_integers(G, ordering='sorted', label_attribute='label')
        graph = from_networkx(G_adj)
        
        # Ensure undirected
        if not graph.is_undirected():
             graph.edge_index = torch.cat([graph.edge_index, graph.edge_index.flip(0)], dim=1)
             
        graph.x = omics_feature_vector
        return graph
    
    # def generate_graph(self, thr_go, thr_exp, thr_seq, thr_path):
    #     """
    #     generate tri-graph: PPI GSN GO_network
    #     """
    #     print('Generating graph data from raw similarity matrices...')
    #     final_gene_node, ppi = self.get_node_genelist()

    #     # 경로 확인 및 로드
    #     path_sim_file = './Data/simmatrix/pathsim_matrix.csv'
    #     go_sim_file = './Data/simmatrix/GOSemSim_matrix.csv'
    #     exp_sim_file = './Data/simmatrix/expsim_matrix.csv'
    #     seq_sim_file = './Data/simmatrix/seqsim_rrbs_matrix.csv'

    #     # 파일 존재 여부 체크
    #     if not all(os.path.exists(f) for f in [path_sim_file, go_sim_file, exp_sim_file, seq_sim_file]):
    #          raise FileNotFoundError("Raw similarity matrix files not found in ./Data/simmatrix/")

    #     path = pd.read_csv(path_sim_file, sep='\t', index_col=0)
    #     path_matrix = path.applymap(lambda x: 0 if x < thr_path else 1)
    #     np.fill_diagonal(path_matrix.values, 0)

    #     go = pd.read_csv(go_sim_file, sep='\t', index_col=0)
    #     go_matrix = go.applymap(lambda x: 0 if x < thr_go else 1)
    #     np.fill_diagonal(go_matrix.values, 0)

    #     exp = pd.read_csv(exp_sim_file, sep='\t', index_col=0)
    #     exp_matrix = exp.applymap(lambda x: 0 if x < thr_exp else 1)
    #     np.fill_diagonal(exp_matrix.values, 0)

    #     seq = pd.read_csv(seq_sim_file, sep='\t', index_col=0)
    #     seq_matrix = seq.applymap(lambda x: 0 if x < thr_seq else 1)
    #     np.fill_diagonal(seq_matrix.values, 0)

    #     networklist = []
    #     # 순서: GO, EXP, SEQ, PATH (PPI는 별도 리턴)
    #     for matrix in [go_matrix, exp_matrix, seq_matrix, path_matrix]:
    #         temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)
    #         network = temp.combine_first(matrix)
    #         network.fillna(0, inplace=True)
    #         network_adj = network[final_gene_node].loc[final_gene_node]
    #         networklist.append(network_adj)
    #         print(f'Generated network shape: {network_adj.shape}')

    #     # 생성된 그래프 파일 저장 (다음 번 실행 속도 향상을 위해)
    #     if not os.path.exists(self.graph_path):
    #         os.makedirs(self.graph_path)
            
    #     ppi.to_csv(os.path.join(self.graph_path, self.ppi_type + '_ppi.tsv'), sep='\t')
    #     networklist[0].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_go) + '_go.tsv'), sep='\t')
    #     networklist[1].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_exp) + '_exp.tsv'), sep='\t')
    #     networklist[2].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_seq) + '_seq.tsv'), sep='\t')
    #     networklist[3].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_path) + '_path.tsv'), sep='\t')

    #     return ppi, networklist[0], networklist[1], networklist[2], networklist[3]
    
    
    def generate_graph(self, thr_go, thr_exp, thr_seq, thr_path):
        """
        generate tri-graph: PPI GSN GO_network
        """
        print('Generating graph data from raw similarity matrices...')
        final_gene_node, ppi = self.get_node_genelist()

        # 경로 확인
        path_sim_file = './Data/simmatrix/pathsim_matrix.csv'
        go_sim_file = './Data/simmatrix/GOSemSim_matrix.csv'
        exp_sim_file = './Data/simmatrix/expsim_matrix.csv'
        seq_sim_file = './Data/simmatrix/seqsim_rrbs_matrix.csv'

        if not all(os.path.exists(f) for f in [path_sim_file, go_sim_file, exp_sim_file, seq_sim_file]):
             raise FileNotFoundError("Raw similarity matrix files not found in ./Data/simmatrix/")

        # --- [Memory Optimization] Helper function to read csv in chunks ---
        def load_and_threshold_csv(file_path, threshold):
            print(f"Processing {os.path.basename(file_path)} with threshold {threshold}...")
            processed_chunks = []
            try:
                for chunk in pd.read_csv(file_path, sep='\t', index_col=0, chunksize=1000, dtype=np.float32):
                    chunk_binary = (chunk >= threshold).astype(np.int8)
                    processed_chunks.append(chunk_binary)
                
                full_df = pd.concat(processed_chunks)
                
                np.fill_diagonal(full_df.values, 0)
                return full_df
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                raise e
        # ------------------------------------------------------------------

        

        go_matrix = load_and_threshold_csv(go_sim_file, thr_go)
        print("GO matrix processed.")
        
        exp_matrix = load_and_threshold_csv(exp_sim_file, thr_exp)
        print("EXP matrix processed.")
        
        seq_matrix = load_and_threshold_csv(seq_sim_file, thr_seq)
        print("SEQ matrix processed.")
        
        path_matrix = load_and_threshold_csv(path_sim_file, thr_path)
        print("PATH matrix processed.")

        networklist = []
        for matrix in [go_matrix, exp_matrix, seq_matrix, path_matrix]:
            temp = pd.DataFrame(index=final_gene_node, columns=final_gene_node)
            network = matrix.reindex(index=final_gene_node, columns=final_gene_node, fill_value=0)
            networklist.append(network)
            print(f'Generated network shape: {network.shape}')

        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)
            
        ppi.to_csv(os.path.join(self.graph_path, self.ppi_type + '_ppi.tsv'), sep='\t')
        networklist[0].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_go) + '_go.tsv'), sep='\t')
        networklist[1].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_exp) + '_exp.tsv'), sep='\t')
        networklist[2].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_seq) + '_seq.tsv'), sep='\t')
        networklist[3].to_csv(os.path.join(self.graph_path, self.ppi_type + '_' + str(thr_path) + '_path.tsv'), sep='\t')

        return ppi, networklist[0], networklist[1], networklist[2], networklist[3]
    
    
    
def cal_metrics(pred_prob, labels):
    pred = np.round(pred_prob)
    acc = metrics.accuracy_score(labels, pred)
    try:
        auroc = metrics.roc_auc_score(labels, pred_prob)
        auprc = metrics.average_precision_score(labels, pred_prob)
    except ValueError:
        auroc = float('nan')
        auprc = float('nan')
    f1 = metrics.f1_score(labels, pred)
    return acc, auroc, auprc, f1

def load_fold_data(fold_idx, fold_dir="./10fold"):
    fold_path = os.path.join(fold_dir, f'fold_{fold_idx}')
    def load_txt(filename):
        with open(os.path.join(fold_path, filename), 'r') as f:
            return [int(line.strip()) for line in f.readlines()]
            
    train_data = load_txt('train.txt')
    test_data = load_txt('test.txt')
    # train_mask = load_txt('train_mask.txt') # Not strictly needed if we map from gene list
    # test_mask = load_txt('test_mask.txt')
    
    return train_data, test_data