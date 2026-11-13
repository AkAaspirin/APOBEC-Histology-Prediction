# APOBEC Histology Prediction - Source Code & Data

This repository contains the source code and data tables for reproducing the results reported in the manuscript.

## Directory Structure
- `data/`: 
  - `new_labels_full.csv`: Original clinical and genomic labels.
  - `table2_metrics.csv`: Main prediction performance metrics.
  - `table6_survival_326.csv`: Reference survival analysis.
  - `table6_survival_91.csv`: 91-patient survival analysis.
  - `07_91_combo/`: OOF predictions for 91-patient subcohort.
  - `332_oof_*`: OOF predictions for 332-patient cohort.
- `scripts/`: 
  - `dataset_stage5.py`: Dataset class with fold-internal residual fitting.
  - `train_tmb.py`, `stage5_train.py`: Model training scripts.
  - `stage2_metrics.py`: Code for Spearman, CI, and FDR.
  - `stage5a_decoupling.py`: TMB decoupling analysis.
  - `stage5b_survival_91.R`, `stage6_survival_326.R`: Survival analyses.
  - `fig2-4_plot.py`: Scripts to generate main figures.

## Environment Requirements
Python dependencies: `pip install -r requirements.txt`
R dependencies: `survival`, `dplyr`

## Note
All paths in the scripts are relative (`../data/`). Data files are provided for reproducibility.
