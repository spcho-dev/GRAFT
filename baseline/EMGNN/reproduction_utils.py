import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.utils import add_self_loops
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def cal_metrics(y_true, y_pred_prob):
    """
    y_true: 실제 라벨 (0 or 1)
    y_pred_prob: 모델의 예측 확률 (Positive Class Probability)
    """
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = metrics.accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_pred_prob)
        auprc = average_precision_score(y_true, y_pred_prob)
    except ValueError:
        auroc = float('nan')
        auprc = float('nan')
        
    f1 = f1_score(y_true, y_pred)
    return acc, auroc, auprc, f1

class EMGNN(torch.nn.Module):
    def __init__(self, nfeat, hidden_channels, n_layers, nclass, meta_x=None, args=None, data=None, node2idx=None):
        super().__init__()

        self.args = args
        self.linear = nn.Linear(nfeat, hidden_channels)
        self.meta_linear = nn.Linear(nfeat, hidden_channels)
        
        if(args.gcn):
            self.meta_gnn = GCNConv(hidden_channels, hidden_channels)
        elif(args.gat):
            self.meta_gnn = GATConv(hidden_channels, hidden_channels, heads=args.nb_heads, concat=False)
        elif(args.gin):
            self.meta_gnn = GINConv(nn.Sequential(
                                    nn.Linear(hidden_channels, hidden_channels), 
                                    nn.LeakyReLU(), 
                                    nn.BatchNorm1d(hidden_channels),
                                    nn.Linear(hidden_channels, hidden_channels)))

        self.classifier = nn.Linear(hidden_channels, nclass)
        self.dropout = args.dropout
        self.leakyrelu = nn.LeakyReLU(args.alpha)
        self.n_layers = n_layers

        lst = list()
        for i in range(n_layers):
            if(args.gcn):
                lst.append(GCNConv(hidden_channels, hidden_channels))
            elif(args.gat):
                lst.append(GATConv(hidden_channels, hidden_channels, heads=args.nb_heads, concat=False))
            elif(args.gin):
                lst.append(GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LeakyReLU(), nn.BatchNorm1d(hidden_channels),nn.Linear(hidden_channels, hidden_channels),nn.LeakyReLU())))
        self.conv = nn.ModuleList(lst)

        # construct meta graph : meta edge index
        x = data.x.float()
        self.nb_nodes = x.shape[0]
        
        # Fix for potential array/list issue
        node_names = np.array(data.node_names) 
        meta_edge_index = [[],[]]
  
        for i, node in enumerate(node_names):
            meta_edge_index[0].append(i)  # input node
            meta_edge_index[1].append(node2idx[node] + x.shape[0]) # add metanode       
        
        # Device handling relies on where the model is initialized (usually handled by .cuda() later)
        # But initialization creates tensors, so we create them on CPU first then move if needed.
        self.register_buffer('meta_edge_index_tensor', torch.tensor(meta_edge_index))
        
        # Handle self loops manually during forward or init? 
        # Original code did it here. We will do it in init but ensure device compatibility.
        self.meta_edge_index, _ = add_self_loops(self.meta_edge_index_tensor)
        
        # meta_x is passed as argument
        self.meta_x = meta_x
    
    def forward(self, x, edge_index, data, meta_edge_index=None, explain_x=None, captum=False, explain=False, edge_weight=None):  
        
        # Ensure meta_edge_index is on same device as x
        if self.meta_edge_index.device != x.device:
            self.meta_edge_index = self.meta_edge_index.to(x.device)
        if self.meta_x.device != x.device:
            self.meta_x = self.meta_x.to(x.device)

        if(meta_edge_index != None):
            self.meta_edge_index = meta_edge_index
     
        x = self.leakyrelu(self.linear(x))
        meta_x = self.leakyrelu(self.meta_linear(self.meta_x))

        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.leakyrelu(x) 
            x = F.dropout(x, self.dropout, training=self.training)
        
        # meta message-passing
        if(self.args.gat):
            x = self.meta_gnn(torch.cat((x, meta_x), dim=0), self.meta_edge_index)   
        else:
            x = self.meta_gnn(torch.cat((x, meta_x), dim=0), self.meta_edge_index)   
        
        x = self.leakyrelu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)