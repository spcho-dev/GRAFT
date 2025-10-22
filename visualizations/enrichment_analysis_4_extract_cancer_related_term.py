import os
import pandas as pd

# --- Configuration ---
base_dir = "./CTD_Association_data_preprocessed"
enrichment_dir = "./enrichment_analysis_results"
output_cancer_id_path = os.path.join(base_dir, "all_cancer_ids.tsv")

# Top-level MeSH ID for cancer-related diseases
CANCER_MESH_ID = "MESH:D009369"

# --- Load Data ---
disease_df = pd.read_csv(f"{base_dir}/CTD_diseases_cleaned.tsv", sep="\t")
go_df = pd.read_csv(f"{base_dir}/CTD_GO_DiseaseGene_cleaned.tsv", sep="\t")
kegg_df = pd.read_csv(f"{base_dir}/CTD_Pathway_DiseaseGene_cleaned.tsv", sep="\t")

# --- Identify and Save All Cancer-Related Disease IDs ---
# A disease is cancer-related if its parent is the top-level cancer ID.
disease_df["IsCancer"] = disease_df["ParentIDs"].fillna("").apply(
    lambda x: CANCER_MESH_ID in x.split("|")
)
cancer_related_ids = set(disease_df[disease_df["IsCancer"]]["DiseaseID"])
if CANCER_MESH_ID in disease_df["DiseaseID"].values:
    cancer_related_ids.add(CANCER_MESH_ID)

pd.Series(sorted(cancer_related_ids), name="DiseaseID").to_csv(output_cancer_id_path, sep="\t", index=False)

# --- Process Enrichment Terms ---
# Reload the saved cancer IDs to ensure a clean, unique set.
cancer_related_ids = set(
    pd.read_csv(output_cancer_id_path, sep="\t")["DiseaseID"].dropna().unique()
)

# Create a mapping from GO/KEGG term to the set of cancer DiseaseIDs it's associated with.
go_map = go_df[go_df["DiseaseID"].isin(cancer_related_ids)].groupby("GOID")["DiseaseID"].apply(set).to_dict()
kegg_map_df = kegg_df[kegg_df["DiseaseID"].isin(cancer_related_ids)].copy()
kegg_map_df["PathwayID"] = kegg_map_df["PathwayID"].str.replace("KEGG:hsa", "KEGG:")
kegg_map = kegg_map_df.groupby("PathwayID")["DiseaseID"].apply(set).to_dict()

# Define keywords to identify cancer-related terms by name.
cancer_keywords = ['cancer', 'tumor', 'carcinoma', 'neoplasm', 'malignancy', 'oncogene']

# --- Filter and Save Cancer-Related Enrichment Terms ---
records = []
for file in sorted(os.listdir(enrichment_dir)):
    if not file.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(enrichment_dir, file))
    category = file.replace("top_15_", "").replace(".csv", "")

    for _, row in df.iterrows():
        term_id = row["Term_ID"]
        term_name = row["Term_Name"]
        source = "GO" if term_id.startswith("GO:") else "KEGG" if term_id.startswith("KEGG:") else "Other"

        # Check for cancer relevance based on MeSH ID mapping.
        is_mesh_related = False
        mesh_diseases = set()
        if source == "GO" and term_id in go_map:
            is_mesh_related = True
            mesh_diseases = go_map[term_id]
        elif source == "KEGG" and term_id in kegg_map:
            is_mesh_related = True
            mesh_diseases = kegg_map[term_id]

        # Check for cancer relevance based on keywords in the term name.
        is_keyword_related = any(kw in term_name.lower() for kw in cancer_keywords)

        if is_mesh_related or is_keyword_related:
            origin = []
            if is_mesh_related:
                origin.append("MeSH")
            if is_keyword_related:
                origin.append("Keyword")

            records.append({
                "Term_ID": term_id,
                "Term_Name": term_name,
                "Category": category,
                "Type": source,
                "Cancer_Relevance_Source": ", ".join(origin),
                "MeSH_DiseaseIDs": "|".join(sorted(mesh_diseases)) if is_mesh_related else ""
            })

# --- Save Final Results ---
df_cancer_terms = pd.DataFrame(records).drop_duplicates().sort_values("Category")

# Save as CSV
csv_path = os.path.join(enrichment_dir, "cancer_related_terms_with_diseaseids.csv")
df_cancer_terms.to_csv(csv_path, index=False)

# Save as a formatted TXT file for readability
txt_path = os.path.join(enrichment_dir, "cancer_related_terms_with_diseaseids.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for _, row in df_cancer_terms.iterrows():
        line = f"[{row['Cancer_Relevance_Source']}] {row['Term_Name']} ({row['Category']}, {row['Type']}, ID: {row['Term_ID']})"
        if row["MeSH_DiseaseIDs"]:
            line += f"  <-- MeSH-DiseaseIDs: {row['MeSH_DiseaseIDs']}"
        line += "\n"
        f.write(line)

print(f"Processing complete. Results saved to:")
print(f"   - {csv_path}")
print(f"   - {txt_path}")
