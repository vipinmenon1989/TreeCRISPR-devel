#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

if len(sys.argv) != 3:
    print("Usage: python enlor_crispra.py <mixed_data.csv> <output_prefix>")
    sys.exit(1)

input_csv, output_prefix = sys.argv[1:2], sys.argv[2]

# -------------------- Load Data --------------------
log(f"Loading dataset: {input_csv}")
df = pd.read_csv(input_csv)

y = df['class'].astype(int).values
X_df = df.drop(columns=['class', 'ID', 'LFC'], errors='ignore')
feature_names = X_df.columns.tolist()
X = X_df.to_numpy()

# Scaling is MANDATORY for Elastic Net
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log(f"Samples: {len(y)} | Features: {len(feature_names)}")

# -------------------- 5-Fold Evaluation --------------------
log("Running 5-Fold CV with ENLOR (Elastic Net Logistic Regression)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_list = []

# Model Config: 
# l1_ratio=0.5 (Balanced Lasso/Ridge)
# C=0.1 (Strong regularization to handle small N)
model = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    C=0.1, 
    max_iter=5000,
    class_weight={0: 5, 1: 1}, # Maintaining your 5x penalty on Class 0
    random_state=42
)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    
    p = model.predict_proba(X_test)[:, 1]
    predictions = (p >= 0.5).astype(int)
    
    # Check if both classes are present in test set for AUC
    try:
        auc = roc_auc_score(y_test, p)
    except ValueError:
        auc = np.nan

    metrics_list.append({
        'AUC': auc,
        'AUPRC': average_precision_score(y_test, p),
        'F1': f1_score(y_test, predictions, zero_division=0),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0)
    })

# -------------------- Metrics Output --------------------
res_df = pd.DataFrame(metrics_list)
res_df.loc['MEAN'] = res_df.mean()
res_df.to_csv(f"{output_prefix}_enlor_metrics.csv")
log("Metrics saved.")

# -------------------- Feature Importance --------------------
log("Extracting Feature Coefficients...")
model.fit(X_scaled, y) # Final fit on full data
importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0]
})
importance['Abs_Coeff'] = importance['Coefficient'].abs()
importance = importance.sort_values(by='Abs_Coeff', ascending=False)

importance.to_csv(f"{output_prefix}_enlor_coefficients.csv", index=False)

# Plotting Top 20 Features
plt.figure(figsize=(10, 8))
top_20 = importance.head(20)
plt.barh(top_20['Feature'], top_20['Coefficient'], color='skyblue')
plt.gca().invert_yaxis()
plt.title(f"Top ENLOR Coefficients (N=103)")
plt.xlabel("Coefficient Weight (Direction of Effect)")
plt.tight_layout()
plt.savefig(f"{output_prefix}_feature_importance.png")

log(f"✅ Completed. Results in {output_prefix}_enlor_metrics.csv")
