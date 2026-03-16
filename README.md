# TreeCRISPR-devel

Development repository for **TreeCRISPR** — a machine learning framework for predicting CRISPRi guide RNA efficacy and identifying high-quality GG motifs in target gene regions. This repository contains per-gene experiment folders with XGBoost models, SHAP analysis, GG motif QC, and fine-tuned model artifacts.

## Overview

CRISPRi (CRISPR interference) uses dCas9 fused to repressors to silence gene expression. Guide RNA (gRNA) efficacy varies substantially depending on sequence features. This repository develops and benchmarks predictive models for CRISPRi screen data, focusing on:

1. **GG Motif QC Analysis**: Identifying and statistically testing whether high-efficacy guides (LFC ≤ -0.5) are enriched for GG dinucleotide motifs at key positions
2. **XGBoost Regression**: Training gradient-boosted tree models to predict guide Log Fold Change (LFC) from sequence and epigenetic features
3. **XGBoost Classification**: Binary classification of guides as effective vs. ineffective
4. **SHAP Feature Importance**: Interpreting which features (epigenetic signals vs. sequence features) drive model predictions
5. **Model Fine-Tuning**: Refining pre-trained CRISPRi models on new datasets

## Repository Structure

Each top-level folder corresponds to a gene target or experimental condition:

```
TreeCRISPR-devel/
├── ALKE/                         # ALK gene — condition E
├── ALKH/                         # ALK gene — condition H
├── ALKL/                         # ALK gene — condition L
├── ALKME/                        # ALK gene — condition ME
├── ALKMH/                        # ALK gene — condition MH
├── ALKML/                        # ALK gene — condition ML
├── K/                            # K gene target
├── SAM/                          # SAM domain target
├── T300/                         # T300 target
└── TP/                           # TP gene target (e.g., TP53)
```

### Files Within Each Gene Folder

```
<GENE>/
├── <GENE>_pre_machine_learning.csv    # Raw features + LFC before ML preprocessing
├── <GENE>_filtered_sample.csv         # Filtered dataset
├── <GENE>_filtered_sampled.csv        # Filtered and downsampled dataset for training
├── <GENE>_PCA.py                      # PCA dimensionality reduction on features
├── <GENE>_diag.py                     # Diagnostic plots and exploratory analysis
├── <GENE>_final_push.py               # XGBoost regression (K-fold CV + Spearman eval)
├── <GENE>_final_push_class.py         # XGBoost classification (binary efficacy)
├── <GENE>_Final_Metrics.csv           # Cross-validation performance metrics
├── <GENE>_Final_SHAP_bar.png          # SHAP bar plot (top feature importances)
├── <GENE>_Final_SHAP_beeswarm.png     # SHAP beeswarm plot (feature impact distribution)
├── GG_finder.py                       # GG motif QC and LFC distribution analysis
├── CRISPRi_fine_tune.py               # Fine-tune CRISPRi models on new data
├── CRISPR_fine_tune_regression.py     # Regression-based fine-tuning
├── *.pkl                              # Saved XGBoost model files
└── *.json                             # Fine-tuned model parameters
```

## Module Descriptions

### GG_finder.py — GG Motif QC Analysis

Performs quality control analysis on Cas9 gRNA datasets to test whether high-quality guides (LFC between -1.5 and -0.5) are enriched for GG motifs at CRISPRi target positions.

**Usage**:
```bash
python GG_finder.py --input <gene>_filtered_sample.csv \
                    --plot motif_qc_combined.png \
                    --report motif_qc_report.txt \
                    --target 0.90
```

**Output**:
- Text report with GG count, non-GG count, purity %, and statistical tests
- Binomial test p-value (purity significance)
- Mann-Whitney U test p-value (LFC distribution: GG vs non-GG)
- Combined QC visualization plot

### `*_final_push.py` — XGBoost Regression

Trains an XGBoost regression model to predict guide LFC values from log-scaled features.

**Pipeline**:
1. Load `<GENE>_filtered_sampled.csv` with LFC labels
2. Log1p-transform features
3. K-Fold cross-validation (default: 5 folds)
4. Train XGBoost with `reg:squarederror` objective, early stopping (200 rounds)
5. Evaluate with Spearman rank correlation
6. Compute and print top-10 feature importances (by gain)
7. Save feature importance plot

### `*_final_push_class.py` — XGBoost Classification

Binary classification variant — predicts whether a guide is effective (LFC < threshold) or not.

### CRISPRi_fine_tune.py — Model Fine-Tuning

Fine-tunes a pre-trained CRISPRi model on new experimental data. Saves refined parameters as JSON for portability.

## Quick Start

### Prerequisites

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn scipy shap
```

### Run GG Motif QC

```bash
cd ALKE
python GG_finder.py --input ALKE_filtered_sample.csv --target 0.90
```

### Train XGBoost Regression Model

```bash
cd ALKE
python ALKE_final_push.py
```

### Train XGBoost Classification Model

```bash
cd ALKE
python ALKE_final_push_class.py
```

### Run PCA and Diagnostics

```bash
cd ALKE
python ALKE_PCA.py
python ALKE_diag.py
```

## Input Data Format

The CSV input files should contain:

| Column | Description |
|--------|-------------|
| ID     | Unique guide identifier |
| LFC    | Log Fold Change from CRISPRi screen |
| class  | Binary label (effective/ineffective) |
| ...    | Sequence and epigenetic feature columns |

## Output

- **`*_Final_Metrics.csv`**: Cross-validated Spearman rho and p-values per fold
- **`*_Final_SHAP_bar.png`**: SHAP bar chart showing mean absolute feature importance
- **`*_Final_SHAP_beeswarm.png`**: SHAP beeswarm plot showing feature impact direction
- **`*.pkl`**: Serialized XGBoost models for inference
- **`*.json`**: Fine-tuned model hyperparameters

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| xgboost | Gradient boosted tree models |
| scikit-learn | K-Fold cross-validation |
| scipy | Spearman correlation, Mann-Whitney U, binomial test |
| matplotlib / seaborn | Visualizations |
| shap | Model interpretability |
| argparse | CLI argument parsing |

## Relation to TreeCRISPRstandalone

This repository is the **active development branch** of the TreeCRISPR project. The production-ready standalone tool is available at [TreeCRISPRstandalone](https://github.com/vipinmenon1989/TreeCRISPRstandalone).

## Author

**Vipin Menon**
Post doctoral Fellow, Computational Biology (Genome Editing), Wei Lei Lab
University of Maryland, Baltimore
University of Maryland-Institute of health computing 
GitHub: [vipinmenon1989](https://github.com/vipinmenon1989)
