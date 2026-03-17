#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
)

# --- HPC HEADLESS CONFIGURATION ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def log(msg):
    print(f"[*] {msg}", flush=True)

if len(sys.argv) != 4:
    print("Usage: python evaluate_crispri.py <independent_data.csv> <public_model.json> <output_prefix>")
    sys.exit(1)

input_csv, model_path, output_prefix = sys.argv[1:4]

# 1. Load the Public Model First
log(f"Loading public model: {model_path}")
booster = xgb.Booster()
booster.load_model(model_path)
expected_features = booster.feature_names

# 2. Load the Independent Dataset
log(f"Loading independent dataset: {input_csv}")
df = pd.read_csv(input_csv)

if 'class' not in df.columns:
    log("CRITICAL ERROR: 'class' column missing from test data.")
    sys.exit(1)

y_true = df['class'].astype(int).values

# 3. Purge Metadata Columns
drop_cols = ['class', 'LFC', 'lfc', 'sgrnaid', 'sgRNA_ID', 'ID']
X_df = df.drop(columns=drop_cols, errors='ignore')

# 4. Handle Feature Alignment
if expected_features is None:
    log("WARNING: The public model does NOT contain saved feature names.")
    log("Assuming your CSV columns are perfectly aligned with the original training data.")
    X = X_df.copy()
else:
    missing_features = [f for f in expected_features if f not in X_df.columns]
    if missing_features:
        log(f"CRITICAL ERROR: Missing {len(missing_features)} features required by the model.")
        log(f"First 5 missing: {missing_features[:5]}")
        sys.exit(1)
    # Reorder ONLY the exact features the model expects
    X = X_df[expected_features].copy()

dtest = xgb.DMatrix(X, label=y_true)

# 5. Generate Predictions
log("Generating predictions...")
probs = booster.predict(dtest)
preds = (probs >= 0.5).astype(int)

# 6. Calculate Analytical Metrics
auc_val = roc_auc_score(y_true, probs)
auprc_val = average_precision_score(y_true, probs)
f1 = f1_score(y_true, preds, zero_division=0)
prec = precision_score(y_true, preds, zero_division=0)
rec = recall_score(y_true, preds, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

log("--- INDEPENDENT EVALUATION METRICS ---")
log(f"AUC:       {auc_val:.4f}")
log(f"AUPRC:     {auprc_val:.4f}")
log(f"F1 Score:  {f1:.4f}")
log(f"Precision: {prec:.4f}")
log(f"Recall:    {rec:.4f}")
log(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# Save metrics to CSV
metrics_df = pd.DataFrame([{
    'Dataset': input_csv,
    'Model': model_path,
    'AUC': auc_val,
    'AUPRC': auprc_val,
    'F1': f1,
    'Precision': prec,
    'Recall': rec,
    'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
}])
metrics_df.to_csv(f"{output_prefix}_evaluation_metrics.csv", index=False)

# 7. Visualization: The Segregation Plot
log("Generating Segregation and Curve Plots...")
plt.figure(figsize=(18, 5))

# Plot A: Probability Distribution
plt.subplot(1, 3, 1)
plt.hist(probs[y_true==0], bins=40, alpha=0.6, density=True, color='red', label='Class 0 (Inactive)')
plt.hist(probs[y_true==1], bins=40, alpha=0.6, density=True, color='green', label='Class 1 (Active)')
plt.title("Model Segregation Plot")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()

# Plot B: ROC Curve
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_true, probs)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_val:.3f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# Plot C: Precision-Recall Curve
plt.subplot(1, 3, 3)
p, r, _ = precision_recall_curve(y_true, probs)
plt.plot(r, p, color='purple', label=f'AUPRC = {auprc_val:.3f}')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.savefig(f"{output_prefix}_evaluation_plots.png", dpi=300)

log(f"✅ Success. Results saved with prefix: {output_prefix}")
