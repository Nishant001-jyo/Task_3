# Week 3 - Unsupervised Learning & Model Evaluation

## Contents
- `clustering.py`  - PCA, K-Means, Hierarchical clustering on the Wine dataset (elbow, silhouette, dendrogram)
- `evaluation.py`  - Cross-validation, confusion matrix, precision/recall/F1, ROC-AUC, GridSearchCV on Breast Cancer dataset
- `run_all.py`     - Runs both scripts
- `outputs/`       - Figures, JSON metrics and cluster profile CSV produced by the scripts

## Run
```
pip install -r requirements.txt
python run_all.py
```
Datasets are bundled with scikit-learn, so no download is needed. All random seeds are fixed (42) for reproducibility.
