import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
df = pd.read_csv('ALKH_filtered_sampled.csv')
y = df['class'].values
X = df.drop(columns=['ID', 'LFC', 'class'])

# Apply Log-transform
X_log = np.log1p(X.clip(lower=0))

# 2. LOGISTIC REGRESSION SETUP
# 'liblinear' is best for small datasets with L1 (Lasso) penalty.
# C=0.1 is a strong penalty to prevent overfitting 688 features.
model_lr = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42)
scaler = StandardScaler()

loo = LeaveOneOut()
all_probs = []
all_true = []

print(f"Starting LOOCV with Logistic Lasso on {len(y)} samples...")

for train_idx, test_idx in loo.split(X_log):
    X_train, X_test = X_log.iloc[train_idx], X_log.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Logistic Regression is sensitive to scale; must scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_lr.fit(X_train_scaled, y_train)
    
    # Get probability for the positive class
    prob = model_lr.predict_proba(X_test_scaled)[0, 1]
    all_probs.append(prob)
    all_true.append(y_test[0])

# 3. RESULTS
all_probs = np.array(all_probs)
all_true = np.array(all_true)
auc_score = roc_auc_score(all_true, all_probs)
aupr_score = average_precision_score(all_true, all_probs)
preds = (all_probs > 0.5).astype(int)

print("\n" + "="*40)
print(f"LASSO LOGISTIC AUC: {auc_score:.4f}")
print(f"LASSO LOGISTIC AUPR: {aupr_score:.4f}")
print("="*40)
print(classification_report(all_true, preds))

# 4. SAVE METRICS
precision, recall, f1, _ = precision_recall_fscore_support(all_true, preds, average='binary')
df_metrics = pd.DataFrame({
    'Metric': ['AUC', 'AUPR', 'Precision', 'Recall', 'F1_Score'],
    'Value': [auc_score, aupr_score, precision, recall, f1]
})
df_metrics.to_csv("ALKH_logistic_metrics.csv", index=False)

# 5. FEATURE IMPORTANCE (Coefficients)
# Train on full data to see which features Lasso kept
X_scaled_all = scaler.fit_transform(X_log)
model_lr.fit(X_scaled_all, y)

# Get features where coefficient is NOT zero
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_lr.coef_[0]
})
top_features = importance[importance['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False)

top_features.to_csv("top_lasso_features.csv", index=False)
print(f"Lasso kept {len(top_features)} features. Top features saved to top_lasso_features.csv")
