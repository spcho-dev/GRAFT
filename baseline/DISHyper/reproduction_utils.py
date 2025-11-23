import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, average_precision_score
from sklearn.metrics import average_precision_score, f1_score
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import os


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



def getData(positiveGenePath, negativeGenePath, geneList):
    positiveGene = pd.read_csv(positiveGenePath, header=None)[0].tolist()
    negativeGene = pd.read_csv(negativeGenePath, header=None)[0].tolist()
    
    positiveGene = sorted(set(positiveGene) & set(geneList))
    negativeGene = sorted(set(negativeGene) & set(geneList))
    
    labelFrame = pd.DataFrame(data=[0] * len(geneList), index=geneList)  # 중립(0)로 초기화
    labelFrame.loc[positiveGene, :] = 1  # 양성(1)
    labelFrame.loc[negativeGene, :] = 0  # 음성(0)
    
    positiveIndex = list(np.where(labelFrame.values.ravel() == 1)[0])
    negativeIndex = list(np.where(labelFrame.values.ravel() == 0)[0])
    
    sampleIndex = np.array(positiveIndex + negativeIndex)
    
    label = np.array([1] * len(positiveIndex) + [0] * len(negativeIndex))

    return sampleIndex, label, labelFrame


def processingIncidenceMatrix(geneList, dataPath):
    feature_genename_file = f'{dataPath}/feature_genename.txt'  # feature_genename.txt _ file path
    filtered_geneList = pd.read_csv(feature_genename_file, header=None).iloc[:, 0].tolist()

    
    print(f"Original geneList size: {len(geneList)} → Filtered size: {len(filtered_geneList)}")
    
    ids = ['c2','c5']
    incidenceMatrix = pd.DataFrame(index=filtered_geneList)
    for id in ids:
        geneSetNameList = pd.read_csv('../../Data/msigdb/'+id+'Name.txt',sep='\t',header=None)
        geneSetNameList = list(geneSetNameList[0].values)
        z=0
        idList = list()
        for name in geneSetNameList:
            if(id=='c2'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    # print(name)
                    pass
                else:
                    idList.append(z)
            elif(name[:2]=='HP'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    # print(name)
                    pass
                else:
                    idList.append(z)
            else:
                idList.append(z)
            z=z+1
        genesetData = sp.load_npz('../../Data/msigdb/'+id+'_GenesetsMatrix.npz')
        
        incidenceMatrixTemp = pd.DataFrame(data=genesetData.A, index=geneList)
        incidenceMatrixTemp = incidenceMatrixTemp.loc[geneList]  # Ensure that the original geneList index is used
        incidenceMatrixTemp = incidenceMatrixTemp.reindex(index=filtered_geneList, fill_value=0)  # Missing genes filled with 0
        
        incidenceMatrixTemp = incidenceMatrixTemp.iloc[:, idList]
        
        # Merge the new data into the overall incidence matrix
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
    # feature_genename.txt 파일에서 유전자 이름 읽기
    gene_names = pd.read_csv(feature_genename_file, header=None).iloc[:, 0].tolist()
    
    
    # train_idx, test_idx에 해당하는 유전자들만 필터링
    filtered_train_genes = [gene_names[i] for i in train_idx]
    filtered_test_genes = [gene_names[i] for i in test_idx]
    
    # labelFrame에 존재하는 유전자들만 필터링 (labelFrame의 인덱스는 유전자 이름)
    filtered_train_genes = [gene for gene in filtered_train_genes if gene in labelFrame.index]
    filtered_test_genes = [gene for gene in filtered_test_genes if gene in labelFrame.index]
    
    # sampleIndex에서 해당 유전자 이름을 가지고 있는 인덱스를 찾기
    trainIndex = [labelFrame.index.get_loc(gene) for gene in filtered_train_genes]
    testIndex = [labelFrame.index.get_loc(gene) for gene in filtered_test_genes]
    
    return trainIndex, testIndex


### DISHyper Model ###

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
    
    
class resHGNNLayer(nn.Module):
    def __init__(self,nhid,dropout=0.5):
        super(resHGNNLayer, self).__init__()
        self.hgc = HGNN_conv(nhid, nhid)
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, G, *args, **kwargs):
        h = self.hgc(x, G)
        h = h + x
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
class DISHyperNet(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, num_layers=3, dropout=0.5):
        super(DISHyperNet, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_ch, n_hid)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                resHGNNLayer(n_hid, dropout=dropout)
            )
        self.outLayer = nn.Linear(n_hid, n_class)
    def forward(self, x, G, *args, **kwargs):
        x = F.relu(self.fc(x))
        x= F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = layer(x, G)
        out = self.outLayer(x) 
        return F.log_softmax(out, dim=1)



