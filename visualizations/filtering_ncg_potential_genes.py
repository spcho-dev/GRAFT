import pandas as pd
import os

# --- Configuration ---
# Input files
ncg_annotation_path = "./node_prediction_pancancer/NCG_cancerdrivers_annotation_supporting_evidence.tsv"
driver_path = '../Data/796true.txt'
nondriver_path = '../Data/2187false.txt'

# Output file
output_dir = "./node_prediction_pancancer"
output_filename = os.path.join(output_dir, "potential_driver_genes_ncg7.2.txt")

try:
    # Step 1: Load the main annotation file
    print(f"Loading NCG annotation data from '{ncg_annotation_path}'...")
    df_ncg = pd.read_csv(ncg_annotation_path, sep='\t')
    df_ncg.fillna('', inplace=True)

    # Step 2: Apply initial filtering to find potential candidates
    # These are genes not annotated as known drivers in major cancer gene databases.
    initial_candidates_df = df_ncg[
        (df_ncg['cgc_annotation'] == '') &
        (df_ncg['vogelstein_annotation'] == '') &
        (df_ncg['saito_annotation'] == '') &
        (df_ncg['NCG_oncogene'] != 1) &
        (df_ncg['NCG_tsg'] != 1)
    ]
    initial_potential_genes = initial_candidates_df['symbol'].drop_duplicates()
    print(f"Found {len(initial_potential_genes)} initial potential candidates from NCG.")

    # Step 3: Load known driver and non-driver gene sets for exclusion
    print(f"Loading known driver and non-driver genes for exclusion...")
    driver_genes = set(pd.read_csv(driver_path, header=None)[0].str.strip())
    nondriver_genes = set(pd.read_csv(nondriver_path, header=None)[0].str.strip())
    
    # Combine both sets into a single exclusion list
    genes_to_exclude = driver_genes | nondriver_genes
    print(f"Total genes to exclude: {len(genes_to_exclude)}")

    # Step 4: Filter the potential candidates by removing genes present in the exclusion list
    final_potential_genes = initial_potential_genes[
        ~initial_potential_genes.isin(genes_to_exclude)
    ]
    print(f"After exclusion, {len(final_potential_genes)} final potential candidates remain.")

    # Step 5: Save the final list to a text file
    os.makedirs(output_dir, exist_ok=True)
    final_potential_genes.to_csv(output_filename, index=False, header=False)

    print("\n" + "="*50)
    print("Success!")
    print(f"{len(final_potential_genes)} final potential cancer driver genes have been saved to:")
    print(f"   {output_filename}")
    print("="*50)

except FileNotFoundError as e:
    print(f"\nError: A required file was not found.")
    print(f"Please check the path: {e.filename}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
