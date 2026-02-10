import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# 1. LOAD DATA
df = pd.read_csv('ALKE_filtered_sampled.csv')
y_binary = df['class'].values
X = df.drop(columns=['ID', 'LFC', 'class'])

# Apply Log-transform
X_log = np.log1p(X.clip(lower=0))

# 2. STEP 1: FEATURE SELECTION
pre_model = xgb.train({'objective': 'binary:logistic'}, xgb.DMatrix(X_log, label=y_binary), num_boost_round=100)
scores = pre_model.get_score(importance_type='gain')
top_50 = sorted(scores, key=scores.get, reverse=True)[:50]

X_filtered = X_log[top_50]
print(f"Refined model to Top 50 features including: {top_50[:5]}")

# 3. STEP 2: LOOCV CLASSIFICATION
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

print(f"Starting Filtered LOOCV on {len(y_binary)} samples...")

for train_idx, test_idx in loo.split(X_filtered):
    dtrain = xgb.DMatrix(X_filtered.iloc[train_idx], label=y_binary[train_idx])
    dtest = xgb.DMatrix(X_filtered.iloc[test_idx])
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    all_probs.append(model.predict(dtest)[0])
    all_true.append(y_binary[test_idx][0])

# 4. RESULTS
all_probs = np.array(all_probs)
auc_score = roc_auc_score(all_true, all_probs)
aupr_score = average_precision_score(all_true, all_probs)
preds = (all_probs > 0.5).astype(int)

print("\n" + "="*40)
print(f"FILTERED CLASSIFICATION AUC: {auc_score:.4f}")
print(f"FILTERED CLASSIFICATION AUPR: {aupr_score:.4f}")
print("="*40)
print(classification_report(all_true, preds))

# 5. SHAP ANALYSIS & PLOTTING
print("Generating SHAP plots for Top 50 features...")
plt.switch_backend('Agg') # Required for HPC/non-GUI environments

# Train the final production model on all samples using only Top 50 features
final_model = xgb.train(params, xgb.DMatrix(X_filtered, label=y_binary), num_boost_round=100)

# Calculate SHAP values
explainer = shap.Explainer(final_model, X_filtered)
shap_values = explainer(X_filtered)

# Plot 1: Summary Beeswarm Plot
plt.figure(figsize=(12, 12))
shap.plots.beeswarm(shap_values, max_display=50, show=False)
plt.title(f"SHAP Impact: Top 50 Features (AUC: {auc_score:.4f})")
plt.savefig('shap_beeswarm_top50.png', bbox_inches='tight', dpi=300)
plt.close()

# Plot 2: Bar Plot (Global Importance)
plt.figure(figsize=(12, 12))
shap.plots.bar(shap_values, max_display=50, show=False)
plt.savefig('shap_bar_top50.png', bbox_inches='tight', dpi=300)
plt.close()

# 6. SAVE TOP FEATURES LOG
with open('top_50_biological_features.txt', 'w') as f:
    f.write("Feature\tGain_Score\n")
    for feat in top_50:
        f.write(f"{feat}\t{scores[feat]:.4f}\n")

print("Files generated: shap_beeswarm_top50.png, shap_bar_top50.png, top_50_biological_features.txt")