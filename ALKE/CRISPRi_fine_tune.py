import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_recall_curve, classification_report, 
                             precision_score, recall_score, f1_score)

# 1. SETUP & DATA LOADING
MODEL_PATH = 'K_xgb_clf.pkl'
DATA_PATH = 'ALKE_filtered_sampled.csv' 

df_alkme = pd.read_csv(DATA_PATH)

# Target and Features
y_binary = df_alkme['class'].values
X_alkme = df_alkme.drop(columns=['ID', 'LFC', 'class'])

# 2. LOAD & ALIGN MODEL
clf_model = joblib.load(MODEL_PATH)
booster = clf_model.get_booster()

# Ensure feature alignment (Critical since .pkl has no names)
if booster.feature_names is not None:
    X_alkme = X_alkme[booster.feature_names]
else:
    print("Warning: Index-based alignment active. Ensure 688 columns match original order.")

# 3. ANALYTICAL PARAMETERS (Optimized for Weak Signal)
# We use a slightly higher learning rate and depth to capture subtle sequence patterns
train_params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,    # Increased to help escape local minima
    'max_depth': 6,          # Increased depth to find complex sequence motifs
    'min_child_weight': 1,
    'gamma': 0.1,            # Regularization to prevent overfitting to noise
    'alpha': 1,              # L1 regularization
    'lambda': 1,             # L2 regularization
    'eval_metric': 'auc',
    'nthread': 4
}

# 4. LEAVE-ONE-OUT CROSS VALIDATION (LOOCV)
loo = LeaveOneOut()
all_probs = []
all_true = []

print(f"Starting LOOCV on {len(y_binary)} samples...")

for train_idx, test_idx in loo.split(X_alkme):
    X_tr, X_te = X_alkme.iloc[train_idx], X_alkme.iloc[test_idx]
    y_tr, y_te = y_binary[train_idx], y_binary[test_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te)
    
    # We add 100 rounds to significantly adapt the model to ALKE signal
    fold_booster = xgb.train(
        train_params, 
        dtrain, 
        num_boost_round=100, 
        xgb_model=booster
    )
    
    prob = fold_booster.predict(dtest)
    all_probs.append(prob[0])
    all_true.append(y_te[0])

# 5. FINAL METRICS & REPORTING
all_probs = np.array(all_probs)
all_true = np.array(all_true)
preds = (all_probs >= 0.5).astype(int)

roc_auc = roc_auc_score(all_true, all_probs)
aupr = average_precision_score(all_true, all_probs)

print("\n" + "="*40)
print(f"FINAL AUC:  {roc_auc:.4f}")
print(f"FINAL AUPR: {aupr:.4f}")
print("="*40)

with open('final_alke_metrics.txt', 'w') as f:
    f.write("ALKME Detailed Performance Report\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"AUPR: {aupr:.4f}\n\n")
    f.write(classification_report(all_true, preds))

# 6. SHAP & VISUALIZATION
print("Generating Plots...")
plt.switch_backend('Agg')

# Final model trained on all data
dfull = xgb.DMatrix(X_alkme, label=y_binary)
final_model = xgb.train(train_params, dfull, num_boost_round=100, xgb_model=booster)

# SHAP Beeswarm
explainer = shap.Explainer(final_model, X_alkme)
shap_values = explainer(X_alkme)
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_values, show=False)
plt.savefig('shap_beeswarm_final.png', bbox_inches='tight')
plt.close()

# PR Curve
precision, recall, _ = precision_recall_curve(all_true, all_probs)
plt.figure()
plt.plot(recall, precision, color='blue', label=f'AUPR = {aupr:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Final ALKME Precision-Recall Curve')
plt.legend()
plt.savefig('final_pr_curve.png')

print("Process Complete. Results saved to final_alke_metrics.txt and PNG files.")