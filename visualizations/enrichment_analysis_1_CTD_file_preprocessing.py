import os
import pandas as pd

## Preprocessing

# File paths
input_path = "./CTD_Association_data/CTD_diseases.tsv"
output_path = "./CTD_Association_data_preprocessed/CTD_diseases_cleaned.tsv"

output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# Define field names (based on official CTD field order)
column_names = [
    "DiseaseName",
    "DiseaseID",
    "AltDiseaseIDs",
    "Definition",
    "ParentIDs",
    "TreeNumbers",
    "ParentTreeNumbers",
    "Synonyms",
    "SlimMappings"
]

# Read the TSV file, skipping comments and applying defined column names
df = pd.read_csv(input_path, sep="\t", comment="#", names=column_names, header=None)

# Select only the essential columns for analysis (ID, ParentID, Name, Synonyms)
df_cleaned = df[["DiseaseName", "DiseaseID", "ParentIDs", "Synonyms"]]

# Save the cleaned DataFrame
df_cleaned.to_csv(output_path, sep="\t", index=False)
print(f"Preprocessing complete: {output_path}")