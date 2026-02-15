import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_fscore_support

# 1. LOAD DATA
df = pd.read_csv('ALKH_extreme_balanced.csv')
y_binary = df['class'].values
X = df.drop(columns=['ID', 'LFC', 'class'])

# Apply Log-transform
X_log = np.log1p(X.clip(lower=0))

# 2. STEP 2: LOOCV CLASSIFICATION WITH INTERNAL FEATURE SELECTION
params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 4,
    'alpha': 2,
    'lambda': 2,
    'eval_metric': 'auc',
    'nthread': 8
}

loo = LeaveOneOut()
all_probs = []
all_true = []

print(f"Starting LOOCV with internal Feature Selection on {len(y_binary)} samples...")

for train_idx, test_idx in loo.split(X_log):
    X_train_fold = X_log.iloc[train_idx]
    y_train_fold = y_binary[train_idx]
    X_test_fold = X_log.iloc[test_idx]

    # Feature Selection (Gain-based) inside the fold to prevent data leakage
    pre_model = xgb.train({'objective': 'binary:logistic'}, 
                          xgb.DMatrix(X_train_fold, label=y_train_fold), 
                          num_boost_round=50)
    scores = pre_model.get_score(importance_type='gain')
    top_50_fold = sorted(scores, key=scores.get, reverse=True)[:50]

    # Train actual model on top features only
    dtrain = xgb.DMatrix(X_train_fold[top_50_fold], label=y_train_fold)
    dtest = xgb.DMatrix(X_test_fold[top_50_fold])
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    all_probs.append(model.predict(dtest)[0])
    all_true.append(y_binary[test_idx][0])

# 3. RESULTS CALCULATION
all_probs = np.array(all_probs)
all_true = np.array(all_true)
auc_score = roc_auc_score(all_true, all_probs)
aupr_score = average_precision_score(all_true, all_probs)
preds = (all_probs > 0.5).astype(int)

print("\n" + "="*40)
print(f"LOOCV AUC: {auc_score:.4f}")
print(f"LOOCV AUPR: {aupr_score:.4f}")
print("="*40)
print(classification_report(all_true, preds))

# 4. SAVE METRICS TO CSV
precision, recall, f1, _ = precision_recall_fscore_support(all_true, preds, average='binary')

metrics_dict = {
    'Metric': ['AUC', 'AUPR', 'Precision', 'Recall', 'F1_Score'],
    'Value': [auc_score, aupr_score, precision, recall, f1]
}
df_metrics = pd.DataFrame(metrics_dict)
df_metrics.to_csv("ALKH_metric.csv", index=False)
print(f"\n[INFO] Metrics successfully saved to ALKH_metric.csv")

# 5. FINAL MODEL SHAP ANALYSIS (Global Importance)
# For the final interpretation, we use the top features across the entire dataset
final_pre_model = xgb.train({'objective': 'binary:logistic'}, 
                            xgb.DMatrix(X_log, label=y_binary), 
                            num_boost_round=100)
final_scores = final_pre_model.get_score(importance_type='gain')
top_50_global = sorted(final_scores, key=final_scores.get, reverse=True)[:50]
X_filtered = X_log[top_50_global]

print("Generating SHAP plots for Top 50 features...")
plt.switch_backend('Agg') 

final_model = xgb.train(params, xgb.DMatrix(X_filtered, label=y_binary), num_boost_round=100)
explainer = shap.Explainer(final_model, X_filtered)
shap_values = explainer(X_filtered)

# Summary Beeswarm Plot
plt.figure(figsize=(12, 12))
shap.plots.beeswarm(shap_values, max_display=50, show=False)
plt.title(f"SHAP Impact: Top 50 Global Features (AUC: {auc_score:.4f})")
plt.savefig('shap_beeswarm_top50.png', bbox_inches='tight', dpi=300)
plt.close()

# Bar Plot
plt.figure(figsize=(12, 12))
shap.plots.bar(shap_values, max_display=50, show=False)
plt.savefig('shap_bar_top50.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. SAVE TOP FEATURES LOG
with open('top_50_biological_features.txt', 'w') as f:
    f.write("Feature\tGain_Score\n")
    for feat in top_50_global:
        f.write(f"{feat}\t{final_scores[feat]:.4f}\n")

print("Files generated: ALKH_metric.csv, shap_beeswarm_top50.png, shap_bar_top50.png, top_50_biological_features.txt")
