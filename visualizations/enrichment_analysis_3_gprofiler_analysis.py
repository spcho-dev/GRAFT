import os
import pandas as pd
import numpy as np
from gprofiler import GProfiler
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
from matplotlib.cm import ScalarMappable
import re


def save_top_terms_with_genes(df, name, top_n=15, outdir='./enrichment_analysis_results/'):
    os.makedirs(outdir, exist_ok=True)

    df = df.copy()
    df['name'] = df['name'].str.replace(r'\s*\(GO:\d+\)', '', regex=True)
    df['log10_pval'] = -np.log10(df['p_value'].clip(lower=1e-300))
    df = df.sort_values('log10_pval', ascending=False).head(top_n)

    # Select columns to save + contain a genetic list
    df_to_save = df[['native', 'name', 'p_value', 'intersection_size', 'term_size', 'query_size', 'intersections']]
    df_to_save.columns = ['Term_ID', 'Term_Name', 'P_value', 'Gene_Count', 'Term_Size', 'Query_Size', 'Genes']

    df_to_save.to_csv(f'{outdir}/top_{top_n}_{name}.csv', index=False)
    print(f'✅ Saved with gene lists: {outdir}/top_{top_n}_{name}.csv')


# The list of genes predicted by the model (known driver gene removed)
genes = pd.read_csv('./node_prediction_pancancer/predicted_top_genes.txt', header=None)[0].tolist()

# Run gProfiler (Latest GO, KEGG-based)
gp = GProfiler(return_dataframe=True)

go_bp = gp.profile(organism='hsapiens', query=genes, sources=['GO:BP'], no_evidences=False)
go_cc = gp.profile(organism='hsapiens', query=genes, sources=['GO:CC'], no_evidences=False)
go_mf = gp.profile(organism='hsapiens', query=genes, sources=['GO:MF'], no_evidences=False)
kegg  = gp.profile(organism='hsapiens', query=genes, sources=['KEGG'], no_evidences=False)

print(go_bp.columns.tolist())
print(go_cc.columns.tolist())
print(go_mf.columns.tolist())
print(kegg.columns.tolist())

save_top_terms_with_genes(go_bp, 'GO_BP')
save_top_terms_with_genes(go_cc, 'GO_CC')
save_top_terms_with_genes(go_mf, 'GO_MF')
save_top_terms_with_genes(kegg,  'KEGG')