import pandas as pd

# --- File Paths ---
go_raw_path = "./CTD_Association_data/CTD_Phenotype-Disease_biological_process_associations.tsv"
pathway_raw_path = "./CTD_Association_data/CTD_diseases_pathways.tsv"

# --- Process GO Association File ---
# Define column names based on the CTD data dictionary.
go_columns = [
    "GOName",
    "GOID",
    "DiseaseName",
    "DiseaseID",
    "InferenceChemicalQty",
    "InferenceChemicalNames",
    "InferenceGeneQty",
    "InferenceGeneSymbols"
]
go_df = pd.read_csv(go_raw_path, sep="\t", comment="#", names=go_columns, header=None, low_memory=False)
go_cleaned = go_df[["GOID", "DiseaseID", "DiseaseName", "InferenceGeneSymbols"]]
go_cleaned.to_csv("./CTD_Association_data_preprocessed/CTD_GO_DiseaseGene_cleaned.tsv", sep="\t", index=False)


# --- Process Pathway Association File ---
pathway_columns = [
    "DiseaseName",
    "DiseaseID",
    "PathwayName",
    "PathwayID",
    "InferenceGeneSymbol"
]
pathway_df = pd.read_csv(pathway_raw_path, sep="\t", comment="#", names=pathway_columns, header=None, low_memory=False)
pathway_cleaned = pathway_df[["PathwayID", "DiseaseID", "DiseaseName", "InferenceGeneSymbol"]]
pathway_cleaned.to_csv("./CTD_Association_data_preprocessed/CTD_Pathway_DiseaseGene_cleaned.tsv", sep="\t", index=False)
