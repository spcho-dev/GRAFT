# preprocess_string_data.py

import os
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

def preprocess_and_save_string_data():
    """
    사용자의 STRING PPI, Multi-omics feature, 유전자 목록을 읽어
    TREE 모델이 사용할 수 있는 .h5 파일로 전처리하고 저장합니다.
    """
    print("--- [Step 1] 데이터 경로 설정 ---")
    # --- 데이터 경로 ---
    # 이 부분은 실제 파일 위치에 맞게 수정해야 할 수 있습니다.
    # base_data_dir = './new_data/STRING'
    # gene_name_file = os.path.join(base_data_dir, 'feature_genename.txt')
    # ppi_file = os.path.join(base_data_dir, 'STRING_ppi_edgelist.tsv')
    # feature_file = os.path.join(base_data_dir, 'multiomics_features_STRING.tsv')
    # pos_gene_file = os.path.join(base_data_dir, 'dataset/pan-cancer/715true.txt')
    # neg_gene_file = os.path.join(base_data_dir, 'dataset/pan-cancer/1231false.txt')
    
    # base_data_dir = './new_data/CPDB'
    # gene_name_file = os.path.join(base_data_dir, 'feature_genename.txt')
    # ppi_file = os.path.join(base_data_dir, 'CPDB_ppi_edgelist.tsv')
    # feature_file = os.path.join(base_data_dir, 'multiomics_features_CPDB.tsv')
    # pos_gene_file = os.path.join(base_data_dir, 'dataset/pan-cancer/787true.txt')
    # neg_gene_file = os.path.join(base_data_dir, 'dataset/pan-cancer/2002false.txt')
    
    base_data_dir = './new_data/BioGRID'
    gene_name_file = os.path.join(base_data_dir, 'feature_genename.txt')
    ppi_file = os.path.join(base_data_dir, 'BioGRID_ppi_edgelist.tsv')
    feature_file = os.path.join(base_data_dir, 'multiomics_features_BioGRID.tsv')
    pos_gene_file = os.path.join(base_data_dir, 'dataset/pan-cancer/763true.txt')
    neg_gene_file = os.path.join(base_data_dir, 'dataset/pan-cancer/1204false.txt')


    # --- 저장될 파일 경로 ---
    output_dir = './dataset/networks'
    os.makedirs(output_dir, exist_ok=True)
    h5_output_path = os.path.join(output_dir, 'BioGRID_multiomics.h5') # config.py와 이름 일치

    print("--- [Step 2] 마스터 유전자 목록 로드 ---")
    master_gene_list = pd.read_csv(gene_name_file, header=None)[0].tolist()
    gene_to_idx = {gene: i for i, gene in enumerate(master_gene_list)}
    n_genes = len(master_gene_list)
    print(f"총 {n_genes}개의 유전자를 기준으로 처리합니다.")

    print("--- [Step 3] PPI 네트워크를 인접 행렬로 변환 ---")
    ppi_df = pd.read_csv(ppi_file, sep='\t', usecols=['partner1', 'partner2'])
    ppi_df.dropna(inplace=True)
    # 유전자 이름을 인덱스로 변환
    src_nodes = ppi_df['partner1'].map(gene_to_idx).dropna()
    dst_nodes = ppi_df['partner2'].map(gene_to_idx).dropna()
    # Scipy 희소 행렬 생성
    adj = sp.csr_matrix((np.ones(len(src_nodes)), (src_nodes.astype(int), dst_nodes.astype(int))),
                        shape=(n_genes, n_genes))
    # 양방향 그래프로 만들고, 대각 성분 제거
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj = adj.tolil()
    adj.setdiag(0)
    adj = adj.tocsr()
    adj_numpy = adj.toarray()
    print(f"인접 행렬 생성 완료. Shape: {adj_numpy.shape}")

    print("--- [Step 4] Multi-omics 피처 행렬 로드 및 정렬 ---")
    feature_df = pd.read_csv(feature_file, sep='\t', index_col=0)
    # 마스터 유전자 리스트 순서에 맞게 피처 행렬 재정렬
    feature_df = feature_df.reindex(master_gene_list).fillna(0)
    feature_numpy = feature_df.values.astype(np.float32)
    print(f"피처 행렬 생성 완료. Shape: {feature_numpy.shape}")

    print("--- [Step 5] 레이블 및 데이터 분할 마스크 생성 ---")
    pos_genes = set(pd.read_csv(pos_gene_file, header=None)[0].tolist())
    neg_genes = set(pd.read_csv(neg_gene_file, header=None)[0].tolist())

    y = np.zeros(n_genes, dtype=np.int32)
    mask = np.zeros(n_genes, dtype=np.int32)

    for i, gene in enumerate(master_gene_list):
        if gene in pos_genes:
            y[i] = 1
            mask[i] = 1
        elif gene in neg_genes:
            y[i] = 0
            mask[i] = 1

    # 10-fold CV를 위한 데이터 분할
    labeled_indices = np.where(mask == 1)[0]
    labels = y[labeled_indices]

    # TREE 코드는 train/val/test를 고정하므로, 여기서 한 번만 분할합니다.
    # 8:1:1 비율로 분할
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_val_idx_ , test_idx_ = next(skf.split(labeled_indices, labels))
    train_val_indices = labeled_indices[train_val_idx_]
    train_val_labels = labels[train_val_idx_]
    test_indices = labeled_indices[test_idx_]

    # train_val을 다시 9:1로 나눠 train과 val 생성
    skf_tv = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
    train_idx_, val_idx_ = next(skf_tv.split(train_val_indices, train_val_labels))
    train_indices = train_val_indices[train_idx_]
    val_indices = train_val_indices[val_idx_]

    # 최종 마스크 생성
    train_mask = np.zeros(n_genes, dtype=np.int32)
    val_mask = np.zeros(n_genes, dtype=np.int32)
    test_mask = np.zeros(n_genes, dtype=np.int32)

    train_mask[train_indices] = 1
    val_mask[val_indices] = 1
    test_mask[test_indices] = 1
    print(f"데이터 분할 완료: Train({len(train_indices)}), Val({len(val_indices)}), Test({len(test_indices)})")

    print(f"--- [Step 6] 최종 .h5 파일 저장: {h5_output_path} ---")
    with h5py.File(h5_output_path, 'w') as f:
        f.create_dataset('network', data=adj_numpy, compression='gzip')
        f.create_dataset('features', data=feature_numpy, compression='gzip')
        f.create_dataset('y_train', data=y)
        f.create_dataset('y_val', data=y)
        f.create_dataset('y_test', data=y)
        f.create_dataset('mask_train', data=train_mask)
        f.create_dataset('mask_val', data=val_mask)
        f.create_dataset('mask_test', data=test_mask)
    print("--- 모든 작업 완료 ---")

if __name__ == '__main__':
    preprocess_and_save_string_data()