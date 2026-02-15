import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, classification_report

# 1. LOAD AND PREP
df = pd.read_csv('ALKH_filtered_sampled.csv') 
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
pc_cols = [f'PC{i+1}' for i in range(n_comps)]

# 3. LOOCV
loo = LeaveOneOut()
all_probs = []
print(f"Starting LOOCV on {n_comps} PCs...")

model = LogisticRegression(class_weight='balanced', penalty='l2', C=0.1)

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
    'Metric': ['AUC', 'AUPR', 'Precision', 'Recall', 'F1_Score', 'n_components'],
    'Value': [auc_score, aupr_score, precision, recall, f1, n_comps]
})
metrics_df.to_csv('ALKH_PCA_Logistic_Metrics.csv', index=False)

# 5. SHAP ANALYSIS (Mapping PCs back to Features)
print("Generating SHAP figures...")
plt.switch_backend('Agg')

# Train final model on all data
model.fit(X_pca, y)

# Explainer for the Logistic Regression
explainer = shap.Explainer(model, X_pca)
shap_values_pca = explainer(X_pca)

# Logic: Map PC SHAP values back to original feature space
# We multiply SHAP values by the PCA components matrix
shap_values_original = np.dot(shap_values_pca.values, pca.components_)

# Create a SHAP Explanation object for the original features
shap_values_final = shap.Explanation(
    values=shap_values_original,
    data=X_scaled,
    feature_names=feature_names
)

# Plot 1: Summary Beeswarm (Top 30 original features)
plt.figure(figsize=(10, 10))
shap.plots.beeswarm(shap_values_final, max_display=30, show=False)
plt.title(f"SHAP Impact: Original Features (LOOCV AUC: {auc_score:.4f})")
plt.savefig('ALKH_PCA_SHAP_beeswarm.png', bbox_inches='tight', dpi=300)
plt.close()

# Plot 2: Bar Plot (Global Importance)
plt.figure(figsize=(10, 10))
shap.plots.bar(shap_values_final, max_display=30, show=False)
plt.savefig('ALKH_PCA_SHAP_bar.png', bbox_inches='tight', dpi=300)
plt.close()

print("\n" + "="*40)
print(f"FINAL AUC: {auc_score:.4f}")
print("Files saved: ALKH_PCA_Logistic_Metrics.csv, ALKH_PCA_SHAP_beeswarm.png, ALKH_PCA_SHAP_bar.png")