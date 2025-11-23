import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import pandas as pd
import math
import os
from torch.nn.parameter import Parameter
from torch_geometric.nn import ChebConv
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.nn.modules.module import Module

def cal_metrics(output, labels):
    outputTest = output.cpu().detach().numpy()
    outputTest = np.exp(outputTest)
    
    outputTest = outputTest[:,1]
    labelsTest = labels.cpu().numpy()
    
    AUROC = roc_auc_score(labelsTest, outputTest)
    AUPRC = average_precision_score(labelsTest, outputTest)
    preds = (outputTest > 0.5).astype(int)
    F1 = f1_score(labelsTest, preds)
    
    return AUROC, AUPRC, F1


def load_fold_data(fold_path):
    train_idx = np.loadtxt(os.path.join(fold_path, 'train.txt'), dtype=int)
    valid_idx = np.loadtxt(os.path.join(fold_path, 'valid.txt'), dtype=int)
    test_idx = np.loadtxt(os.path.join(fold_path, 'test.txt'), dtype=int)
    train_mask = np.loadtxt(os.path.join(fold_path, 'train_mask.txt'), dtype=int)
    valid_mask = np.loadtxt(os.path.join(fold_path, 'valid_mask.txt'), dtype=int)
    test_mask = np.loadtxt(os.path.join(fold_path, 'test_mask.txt'), dtype=int)
    labels = np.loadtxt(os.path.join(fold_path, 'labels.txt'), dtype=int)
    
    return train_idx, valid_idx, test_idx, train_mask, valid_mask, test_mask, labels


def get_train_test_indices_from_gene_names(feature_genename_file, train_idx, test_idx, sampleIndex, labelFrame):
    gene_names = pd.read_csv(feature_genename_file, header=None).iloc[:, 0].tolist()
    
    filtered_train_genes = [gene_names[i] for i in train_idx]
    filtered_test_genes = [gene_names[i] for i in test_idx]
    
    filtered_train_genes = [gene for gene in filtered_train_genes if gene in labelFrame.index]
    filtered_test_genes = [gene for gene in filtered_test_genes if gene in labelFrame.index]
    
    trainIndex = [labelFrame.index.get_loc(gene) for gene in filtered_train_genes]
    testIndex = [labelFrame.index.get_loc(gene) for gene in filtered_test_genes]
    
    return trainIndex, testIndex


def getData(positiveGenePath, negativeGenePath, geneList):
    positiveGene = pd.read_csv(positiveGenePath, header = None)
    positiveGene = list(positiveGene[0].values)
    positiveGene = list(set(geneList)&set(positiveGene))
    positiveGene.sort()
    negativeGene = pd.read_csv(negativeGenePath, header = None)     
    negativeGene = negativeGene[0]
    negativeGene = list(set(negativeGene)&set(geneList))
    negativeGene.sort()
    
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    positiveIndex = np.where(labelFrame == 1)[0]
    labelFrame.loc[negativeGene,:] = -1
    negativeIndex = np.where(labelFrame == -1)[0]
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    
    positiveIndex = list(positiveIndex)
    negativeIndex = list(negativeIndex)
    sampleIndex = positiveIndex + negativeIndex
    sampleIndex = np.array(sampleIndex)
    label = pd.DataFrame(data = [1]*len(positiveIndex) + [0]*len(negativeIndex))
    label = label.values.ravel()
    return  sampleIndex, label, labelFrame


def processingIncidenceMatrix(geneList, dataPath):
    feature_genename_file = f'{dataPath}/feature_genename.txt'  # feature_genename.txt
    filtered_geneList = pd.read_csv(feature_genename_file, header=None).iloc[:, 0].tolist()

    
    print(f"Original geneList size: {len(geneList)} → Filtered size: {len(filtered_geneList)}")
    
    ids = ['c2','c5']
    incidenceMatrix = pd.DataFrame(index=filtered_geneList)
    for id in ids:
        geneSetNameList = pd.read_csv('./Data/'+id+'Name.txt',sep='\t',header=None)
        geneSetNameList = list(geneSetNameList[0].values)
        z=0
        idList = list()
        for name in geneSetNameList:
            if(id=='c2'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q or 'CARCINOMA' in q or 'LEUKEMIA' in q or 'SARCOMA' in q):
                    # print(name)
                    pass
                else:
                    idList.append(z)
            elif(name[:2]=='HP'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q or 'CARCINOMA' in q or 'LEUKEMIA' in q or 'SARCOMA' in q):
                    # print(name)
                    pass
                else:
                    idList.append(z)
            else:
                idList.append(z)
            z=z+1
        genesetData = sp.load_npz('./Data/'+id+'_GenesetsMatrix.npz')
        
        incidenceMatrixTemp = pd.DataFrame(data=genesetData.A, index=geneList)
        incidenceMatrixTemp = incidenceMatrixTemp.loc[geneList]  # Ensure that the original geneList index is used
        incidenceMatrixTemp = incidenceMatrixTemp.reindex(index=filtered_geneList, fill_value=0)  # Missing genes filled with 0
        
        incidenceMatrixTemp = incidenceMatrixTemp.iloc[:, idList]
        
        incidenceMatrix = pd.concat([incidenceMatrix, incidenceMatrixTemp], axis=1)

    # column indexing with numbers
    incidenceMatrix.columns = np.arange(incidenceMatrix.shape[1])
    
    print(f"Final incidenceMatrix shape: {incidenceMatrix.shape}")
    
    return incidenceMatrix


def _generate_G_from_H_weight(H, W):
    n_edge = H.shape[1]
    DV = np.sum(H * W, axis=1)  # the degree of the node
    DE = np.sum(H, axis=0)  # the degree of the hyperedge
    invDE = np.mat(np.diag(1/DE))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    G = DV2 * H * W * invDE * HT * DV2
    return G



### DISFusion model ###

class graph_ChebNet(torch.nn.Module):
    def __init__(self, hdim = 256, dropout = 0.5):
        super(graph_ChebNet, self).__init__()
        self.conv1 = ChebConv(48, hdim, K=2) # pan-cancer
        # self.conv1 = ChebConv(3, hdim, K=2) # specific-cancer
        self.conv2 = ChebConv(hdim, hdim, K=2)
        self.conv3 = ChebConv(hdim, hdim, K=2)
        self.dropout = dropout
    def forward(self, x, edge):
        edge_index = edge

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)

        return x
    
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    
class hypergrph_HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5, n_class=2):
        super(hypergrph_HGNN, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_ch,n_hid)
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_hid)
        self.outLayer = nn.Linear(n_hid, n_class)
    def forward(self, x, G):
        x1 = F.relu(self.fc(x))
        x1= F.dropout(x1, self.dropout, training=self.training)
        
        x2 = F.relu(self.hgc1(x1, G)+x1) 
        x2 = F.dropout(x2, self.dropout, training=self.training) 
        
        x3 = F.relu(self.hgc2(x2, G)+x2) 
        x3 = F.dropout(x3, self.dropout, training=self.training) 
        
        x4 = F.relu(self.hgc3(x3, G)+x3)
        return x4
    
class DISFusion(nn.Module):
    def __init__(self, input_dim, length, lambdinter, attention, nb_classes, dropout = 0.5):
        super().__init__()
        self.lambdinter = lambdinter
        self.attention = attention
        self.dropout = dropout
        self.w_list = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=True) for _ in range(length)])
        self.y_list = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(length)])
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        self.logistic = LogReg(input_dim, nb_classes, self.dropout)
        self.concatFC=nn.Linear(input_dim * 2, input_dim)
        
    def combine_att(self, input1, input2):
        h_list = []
        h_list.append(input1)
        h_list.append(input2)
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h
    
    def combine_concat(self, input1, input2):
        x = torch.cat([input1,input2],1) 
        x = F.relu(self.concatFC(x))
        return x #x1= F.dropout(x1, self.dropout, training=self.training)
        
    def forward(self, input1, input2):
        if self.attention:
            h_fusion = self.combine_att(input1,input2)
        else:
            h_fusion = self.combine_concat(input1,input2)
        semi = self.logistic(h_fusion).squeeze(0)
        EPS = 1e-15
        batch_size = input1.size(0)
        feature_dim = input1.size(1)
        input1 = (input1 - input1.mean(dim=0)) / (input1.std(dim=0) + EPS)
        input2 = (input2 - input2.mean(dim=0)) / (input2.std(dim=0) + EPS)
        inter_c = input1.T @ input2 / batch_size
        on_diag_inter = torch.diagonal(inter_c).add_(-1).pow_(2).sum()
        off_diag_inter = off_diagonal(inter_c).pow_(2).sum()
        loss_inter = (on_diag_inter + self.lambdinter * off_diag_inter)
        
        return loss_inter, semi

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes, dropout):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.dropout = dropout
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        seq = F.dropout(seq, self.dropout, training=self.training)
        ret = self.fc(seq)
        return F.log_softmax(ret, dim=1)
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()