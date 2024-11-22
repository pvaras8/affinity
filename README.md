# DYRK1A and BChE Molecule Prediction for Alzheimer's Disease

This repository is focused on developing predictive models to estimate the **pChEMBL values** of molecules targeting **DYRK1A** (Dual-specificity tyrosine-phosphorylation-regulated kinase 1A) and **BChE** (Butyrylcholinesterase), both of which are associated with **Alzheimer's disease**.

## Project Objectives

1. **DYRK1A Molecules**:
   - DYRK1A is a kinase that plays a critical role in neurodegeneration and is strongly linked to Alzheimer's disease. Predicting the pChEMBL values of potential inhibitors can assist in drug discovery efforts.

2. **BChE Molecules**:
   - **Butyrylcholinesterase (BChE)** is an enzyme involved in the breakdown of acetylcholine, a neurotransmitter crucial for cognitive function. In Alzheimer's disease, BChE activity increases as the disease progresses, making it an important target for therapeutic intervention. This project aims to predict the activity of molecules targeting BChE.

## Workflow

The project includes the following components:

- **Data Preprocessing**:
  - Molecule datasets are preprocessed, including chemical descriptors and feature extraction.
  
- **Model Training**:
  - A variety of machine learning models are implemented, including:
    - Gradient Boosting (`XGBoost`)
    - Random Forest (`RF`)
    - Neural Networks (`NN`)
    - Support Vector Regression (`SVR`)
    - Gaussian Processes (`GP`)

- **Optimization**:
  - Hyperparameter optimization is applied to enhance model performance.
  
- **Evaluation**:
  - The models are evaluated using metrics such as RMSE, MAE, and R².

## Repository Structure

- **`configs/`**: Configuration files for different models.
- **`data/`**: Datasets used for training and evaluation.
- **`models/`**: Trained models and results.
- **`mlartifacts/`**: Intermediate outputs and logs.
- **`optimization/`**: Hyperparameter optimization scripts.
- **`utils/`**: Helper functions for data processing and evaluation.
- **Training Scripts**:
  - `train_gp_final.py`: Training with Gaussian Processes.
  - `train_nn_final.py`: Training with Neural Networks.
  - `train_rf_final.py`: Training with Random Forests.
  - `train_xgboost_final.py`: Training with XGBoost.
  - `train_svr.py`: Training with Support Vector Regression.

## Requirements

To run the project, install the required dependencies:

```bash
pip install -r requirements.txt
# affinity
