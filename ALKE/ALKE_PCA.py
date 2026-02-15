import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# 1. LOAD AND PREP
df = pd.read_csv('ALKE_filtered_sample.csv') 
y = df['class'].values
feature_names = df.drop(columns=['ID', 'LFC', 'class']).columns
X = df[feature_names]

X_log = np.log1p(X.clip(lower=0))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 2. PCA (80% Variance)
pca = PCA(n_components=0.80, random_state=42)
X_pca = pca.fit_transform(X_scaled)
n_comps = X_pca.shape[1]

# 3. LOOCV
loo = LeaveOneOut()
all_probs = []
print(f"Starting LOOCV with {n_comps} PCs on {len(y)} samples...")

# Stronger penalty (C=0.05) to handle small sample sizes
model = LogisticRegression(class_weight='balanced', penalty='l2', C=0.05, solver='liblinear')

for train_idx, test_idx in loo.split(X_pca):
    model.fit(X_pca[train_idx], y[train_idx])
    all_probs.append(model.predict_proba(X_pca[test_idx])[0, 1])

all_probs = np.array(all_probs)
auc_score = roc_auc_score(y, all_probs)
aupr_score = average_precision_score(y, all_probs)
preds = (all_probs > 0.5).astype(int)

# 4. SAVE METRICS
precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary')
metrics_df = pd.DataFrame({
    'Metric': ['AUC', 'AUPR', 'Precision', 'Recall', 'F1_Score', 'n_components', 'total_samples'],
    'Value': [auc_score, aupr_score, precision, recall, f1, n_comps, len(y)]
})
metrics_df.to_csv('ALKH_Final_Metrics.csv', index=False)

# 5. SHAP ANALYSIS
print("Generating SHAP figures...")
plt.switch_backend('Agg')
model.fit(X_pca, y)
explainer = shap.Explainer(model, X_pca)
shap_values_pca = explainer(X_pca)

# Map PC SHAP values back to original features
shap_values_original = np.dot(shap_values_pca.values, pca.components_)
shap_values_final = shap.Explanation(
    values=shap_values_original,
    data=X_scaled,
    feature_names=feature_names
)

# Plot 1: Beeswarm
plt.figure(figsize=(12, 12))
shap.plots.beeswarm(shap_values_final, max_display=30, show=False)
plt.title(rf"SHAP Impact: $|LFC| \in [0.8, 1.5]$ (AUC: {auc_score:.4f})")
plt.savefig('ALKH_Final_SHAP_beeswarm.png', bbox_inches='tight', dpi=300)
plt.close()

# Plot 2: Bar Plot
plt.figure(figsize=(12, 12))
shap.plots.bar(shap_values_final, max_display=30, show=False)
plt.savefig('ALKH_Final_SHAP_bar.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"\nFINAL ANALYSIS COMPLETE.")
print(f"AUC: {auc_score:.4f} | Samples: {len(y)} | Components: {n_comps}")
