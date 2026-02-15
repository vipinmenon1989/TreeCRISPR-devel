import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# 1. LOAD DATA
df = pd.read_csv('ALKL_filtered_sample.csv') 
y = df['class'].values
feature_names = df.drop(columns=['ID', 'LFC', 'class']).columns
X = df[feature_names]

# Transform & Scale
X_log = np.log1p(X.clip(lower=0))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# 2. LOOCV with LASSO (L1)
loo = LeaveOneOut()
all_probs = []

print(f"Starting LOOCV with Logistic Lasso on {len(y)} samples...")

# C=0.5 provides moderate regularization to select sparse features
model = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, class_weight='balanced')

for train_idx, test_idx in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train = y[train_idx]
    
    model.fit(X_train, y_train)
    all_probs.append(model.predict_proba(X_test)[0, 1])

all_probs = np.array(all_probs)
auc_score = roc_auc_score(y, all_probs)
aupr_score = average_precision_score(y, all_probs)
preds = (all_probs > 0.5).astype(int)

# 3. SAVE METRICS
precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary')
metrics_df = pd.DataFrame({
    'Metric': ['AUC', 'AUPR', 'Precision', 'Recall', 'F1_Score'],
    'Value': [auc_score, aupr_score, precision, recall, f1]
})
metrics_df.to_csv('ALKL_Lasso_Metrics.csv', index=False)

# 4. SHAP ANALYSIS (Directly on original features)
print("Generating SHAP figures...")
plt.switch_backend('Agg')

# Train final model on all data
model.fit(X_scaled, y)
n_nonzero = np.sum(model.coef_ != 0)
print(f"Lasso kept {n_nonzero} features out of {len(feature_names)}.")

explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer(X_scaled)

# Summary Beeswarm
plt.figure(figsize=(12, 12))
shap.summary_plot(shap_values, X_scaled_df, max_display=30, show=False)
plt.title(rf"ALKL SHAP: Lasso Logistic (AUC: {auc_score:.4f})")
plt.savefig('ALKL_Lasso_SHAP_beeswarm.png', bbox_inches='tight', dpi=300)
plt.close()

# Bar Plot
plt.figure(figsize=(12, 12))
shap.summary_plot(shap_values, X_scaled_df, plot_type="bar", max_display=30, show=False)
plt.savefig('ALKL_Lasso_SHAP_bar.png', bbox_inches='tight', dpi=300)
plt.close()

print("\n" + "="*40)
print(f"ALKL FINAL AUC: {auc_score:.4f}")
print(f"Features kept by Lasso: {n_nonzero}")
print("Files saved: ALKL_Lasso_Metrics.csv, ALKL_Lasso_SHAP_beeswarm.png, ALKL_Lasso_SHAP_bar.png")
