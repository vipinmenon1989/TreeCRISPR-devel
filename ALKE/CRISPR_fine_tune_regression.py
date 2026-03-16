import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut

# 1. SETUP
# Use your REGRESSION pkl file here
MODEL_PATH = 'K_xgb_model_reg.pkl' 
DATA_PATH = 'ALKE_filtered_sampled.csv'

df = pd.read_csv(DATA_PATH)
y_lfc = df['LFC'].values
# Drop metadata; ensure the remaining 688 features match the pkl order
X = df.drop(columns=['ID', 'LFC', 'class'])

# 2. LOAD REGRESSION MODEL
reg_model = joblib.load(MODEL_PATH)
# Access the booster directly for the xgb.train interface
booster = reg_model.get_booster()

# 3. REGRESSION PARAMETERS
# 'reg:squarederror' is the standard for LFC prediction
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'nthread': 4
}

# 4. LEAVE-ONE-OUT CROSS VALIDATION
loo = LeaveOneOut()
all_preds = []
all_actual = []

print(f"Starting Spearman LOOCV on {len(y_lfc)} samples...")

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y_lfc[train_idx], y_lfc[test_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te)
    
    # Fine-tuning the regression model on ALKME data
    fold_model = xgb.train(params, dtrain, num_boost_round=50, xgb_model=booster)
    
    pred = fold_model.predict(dtest)
    all_preds.append(pred[0])
    all_actual.append(y_te[0])

# 5. SPEARMAN CALCULATION
rho, pval = spearmanr(all_actual, all_preds)

print("\n" + "="*35)
print(f"SPEARMAN RHO: {rho:.4f}")
print(f"P-VALUE:      {pval:.4e}")
print("="*35)

# 6. RESULTS VISUALIZATION
plt.switch_backend('Agg')
plt.figure(figsize=(8, 6))
plt.scatter(all_actual, all_preds, alpha=0.4, s=10, color='crimson')
# Add a trend line
m, b = np.polyfit(all_actual, all_preds, 1)
plt.plot(all_actual, m*np.array(all_actual) + b, color='black', linestyle='--', lw=1)

plt.xlabel('Measured LFC (ALKME)')
plt.ylabel('Predicted LFC')
plt.title(f'Regression Fine-Tuning: Spearman Rho = {rho:.3f}')
plt.grid(alpha=0.2)
plt.savefig('spearman_lfc_correlation.png')

# Save raw predictions for your records
pd.DataFrame({'Actual': all_actual, 'Predicted': all_preds}).to_csv('lfc_predictions.csv', index=False)
