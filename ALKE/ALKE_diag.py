import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold # Using 10-fold for speed over 860 samples

# 1. DATA SETUP
df = pd.read_csv('ALKE_filtered_sampled.csv')
y_lfc = df['LFC'].values
X_all = df.drop(columns=['ID', 'LFC', 'class'])

# Split features into Sequence vs Epigenetics
X_epi = X_all.iloc[:, :78]   # First 78 columns (H3K..., etc)
X_seq = X_all.iloc[:, 78:]   # The nucleotide columns (A1, T1...)

# 2. COMPETITION SETTINGS
params = {'objective': 'reg:squarederror', 'learning_rate': 0.05, 'max_depth': 4}
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def get_spearman(X_data, y_data):
    preds_total = []
    actual_total = []
    for train_idx, test_idx in kf.split(X_data):
        dtrain = xgb.DMatrix(X_data.iloc[train_idx], label=y_data[train_idx])
        dte = xgb.DMatrix(X_data.iloc[test_idx])
        model = xgb.train(params, dtrain, num_boost_round=100)
        preds_total.extend(model.predict(dte))
        actual_total.extend(y_data[test_idx])
    return spearmanr(actual_total, preds_total)[0]

# 3. RUN DIAGNOSTIC
print("Running Signal Diagnostic...")
rho_seq = get_spearman(X_seq, y_lfc)
rho_epi = get_spearman(X_epi, y_lfc)
rho_all = get_spearman(X_all, y_lfc)

print("\n" + "="*35)
print(f"SEQUENCE ONLY RHO:    {rho_seq:.4f}")
print(f"EPIGENETIC ONLY RHO:  {rho_epi:.4f}")
print(f"COMBINED SIGNAL RHO:  {rho_all:.4f}")
print("="*35)
