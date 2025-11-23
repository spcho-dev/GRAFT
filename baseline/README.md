
# Baseline Models Reproduction

This directory contains resources to reproduce the performance of the eight baseline models compared in our study. For each model, we provide pre-trained model weights (checkpoints) and a Jupyter Notebook (`reproduce_test.ipynb`) to facilitate step-by-step inference and performance evaluation.

## Directory Structure

Each baseline model has its own dedicated subdirectory containing the necessary reproduction scripts and notebooks.

```text
baseline/
├── MODIG/
├── EMGNN/
├── GATOmics/
├── ECD-CDGI/
├── MNGCL/
│   ├── reproduction_utils.py # (Or model.py, utils.py) Model definitions & helpers
│   ├── reproduce_test.ipynb  # Jupyter notebook for reproduction
│   └── Data/                     # Learned model weight files and data
├── DISHyper/
├── DISFusion/
└── TREE/
```

-----

## Data Setup

To successfully run the reproduction scripts, please follow these steps regarding data files:

1. **Base Data:**  
   Basic input files required for the models are located in the `Data/` directory within this folder.

2. **Compressed Files:**  
   Some data files are provided in compressed formats (e.g., `.7z`, `.zip`) to save space.  
   **You must extract these files into the same directory before running any scripts.**

3. **Missing Large Files:**  
   Due to GitHub's file size limitations, **some large datasets (e.g., massive PPI networks, raw multi-omics matrices) could not be uploaded to this repository.**

   - If you encounter a `FileNotFoundError` related to large data files, please visit the **Official GitHub Repository** of the respective model (linked in the table below).
   - Download the original data files from the official source.
   - Place them in the corresponding path (e.g., `baseline/MODEL_NAME/Data/`) as required by the script.

-----

## Baseline Models & References

Below is the list of baseline models included in this comparison, along with their original publications and links to their official code repositories.

| Model | Reference | Official Repository |
| :--- | :--- | :--- |
| **MODIG** | Zhao, W., et al. (2022). MODIG: integrating multi-omics and multi-dimensional gene network for cancer driver gene identification based on graph attention network model. *Bioinformatics*, 38(21). | [Link](https://github.com/zjupgx/modig) |
| **EMGNN** | Chatzianastasis, M., et al. (2023). Explainable multilayer graph neural network for cancer gene prediction. *Bioinformatics*, 39(11). | [Link](https://github.com/zhanglab-aim/EMGNN) |
| **GATOmics** | Kong, G., et al. (2025). GATOmics: A Novel Multi-Omics Graph Attention Network Model for Cancer Driver Gene Detection. *ICASSP 2025*. | [Link](https://github.com/ggkong/GATOmics) |
| **ECD-CDGI** | Wang, T., et al. (2024). ECD-CDGI: An efficient energy-constrained diffusion model for cancer driver gene identification. *PLOS Computational Biology*, 20(8). | [Link](https://github.com/taowang11/ECD-CDGI) |
| **MNGCL** | Peng, W., et al. (2024). Multi-network graph contrastive learning for cancer driver gene identification. *IEEE TNSE*, 11(4). | [Link](https://github.com/weiba/MNGCL) |
| **DISHyper** | Deng, C., et al. (2024). Identifying new cancer genes based on the integration of annotated gene sets via hypergraph neural networks. *Bioinformatics*, 40(Suppl 1). | [Link](https://github.com/genemine/DISHyper) |
| **DISFusion** | Deng, C., et al. (2025). Improving Cancer Gene Prediction by Enhancing Common Information Between the PPI Network and Gene Functional Association. *AAAI*, 39(1). | [Link](https://github.com/CharlesDeng0814/DISFusion) |
| **TREE** | Su, X., et al. (2025). Interpretable identification of cancer genes across biological networks via transformer-powered graph representation learning. *Nat Biomed Eng*, 9(3). | [Link](https://github.com/Blair1213/TREE) |

-----

## Notes on Reproduced Performance (Recommended)

The performance reproduced in this repository may not match the exact results reported in the original publications.  
Since each model is evaluated on pre-defined folds and relies on external dependencies, slight differences from the original average scores can naturally occur.

If you wish to go beyond simple reproduction — for example:

- running the models on entire datasets,  
- training from scratch,  
- or applying the models to new or custom data —  

we recommend consulting each model’s **official GitHub repository** linked above.  
The official repositories include the full training pipelines, complete datasets, and configuration details that may not be included here due to repository size constraints.

This reproduction environment is designed to provide a **lightweight, accessible, and practical** way to explore the core behavior of the baseline models, while the official sources remain the best place for **full-scale experimentation and advanced use cases**.

