#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, roc_curve, precision_recall_curve
)

# --- HPC HEADLESS CONFIGURATION ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

if len(sys.argv) != 4:
    print("Usage: python xgb_finetune_shap.py <mixed_data.csv> <base_ngg_model.json> <output_prefix>")
    sys.exit(1)

input_csv, base_model_path, output_prefix = sys.argv[1:4]

# -------------------- Load Data --------------------
log(f"Loading dataset: {input_csv}")
df = pd.read_csv(input_csv)

y = df['class'].astype(int).values
# 5x penalty on Class 0 (the non-NGG "Bad Guys")
weights = np.where(y == 0, 5.0, 1.0)

# Drop non-feature columns
X_df = df.drop(columns=['class', 'ID', 'LFC'], errors='ignore')
feature_names = X_df.columns.tolist()
X = X_df.to_numpy()

log(f"Samples: {len(y)} | Features: {len(feature_names)}")
log(f"Strategy: Penalizing Class 0 to enforce strict NGG-dependency.")

# -------------------- 5-Fold Evaluation --------------------
log("Running 5-Fold CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_list = []
tprs, precs_list = [], []
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.005,
    'max_depth': 2,               # STRONGER: Dropped from 3 to 2. Forces simpler rules.
    'subsample': 0.6,             # NEW: Only use 60% of the 128 training samples per tree.
    'colsample_bytree': 0.6,      # NEW: Only use 60% of the features per tree.
    'alpha': 2.0,                 # NEW: L1 Regularization (forces useless features to exactly 0).
    'lambda': 5.0,                # NEW: L2 Regularization (smooths out extreme weights).
    'tree_method': 'auto'
}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], weight=weights[train_idx], feature_names=feature_names)
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], feature_names=feature_names)
    
    m = xgb.train(params, dtrain, num_boost_round=30, xgb_model=base_model_path)
    
    p = m.predict(dtest)
    predictions = (p >= 0.5).astype(int)
    
    metrics_list.append({
        'AUC': roc_auc_score(y[test_idx], p),
        'AUPRC': average_precision_score(y[test_idx], p),
        'F1': f1_score(y[test_idx], predictions, zero_division=0),
        'Precision': precision_score(y[test_idx], predictions, zero_division=0),
        'Recall': recall_score(y[test_idx], predictions, zero_division=0)
    })
    
    fpr, tpr, _ = roc_curve(y[test_idx], p)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    pr_p, pr_r, _ = precision_recall_curve(y[test_idx], p)
    precs_list.append(np.interp(mean_recall, pr_r[::-1], pr_p[::-1]))

# -------------------- Metrics Output --------------------
res_df = pd.DataFrame(metrics_list)
res_df.loc['MEAN'] = res_df.mean()
res_df.loc['STD'] = res_df.std()
res_df.to_csv(f"{output_prefix}_honest_metrics.csv")

# -------------------- Performance Plots --------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mean_fpr, np.mean(tprs, axis=0), color='b', label=f'Mean AUC: {res_df.at["MEAN","AUC"]:.2f}')
plt.plot([0,1],[0,1],'r--'); plt.title("ROC Curve"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mean_recall, np.mean(precs_list, axis=0), color='g', label=f'Mean AUPRC: {res_df.at["MEAN","AUPRC"]:.2f}')
plt.title("PR Curve"); plt.legend()
plt.tight_layout()
plt.savefig(f"{output_prefix}_performance.png", dpi=300)
plt.close()

# -------------------- Final Production Model --------------------
log("Training Final Production Model...")
dfull = xgb.DMatrix(X, label=y, weight=weights, feature_names=feature_names)
final_m = xgb.train(params, dfull, num_boost_round=50, xgb_model=base_model_path)
final_m.save_model(f"{output_prefix}_strict_editor.json")

# -------------------- SHAP Interpretation --------------------
log("Generating SHAP summary plot...")
explainer = shap.TreeExplainer(final_m)
# Use full dataset for global importance
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
plt.title(f"SHAP Global Importance: {output_prefix}")
plt.tight_layout()
plt.savefig(f"{output_prefix}_shap_summary.png", dpi=300)
plt.close()

log(f"✅ Completed. Model: {output_prefix}_strict_editor.json | Plots saved.")
