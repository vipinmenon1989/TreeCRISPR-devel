import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# 1. DATA LOADING
df = pd.read_csv('ALKE_filtered_sampled.csv')
y_lfc = df['LFC'].values
X = df.drop(columns=['ID', 'LFC', 'class'])

# 2. DATA TRANSFORMATION
# Raw signals are usually log-normal. Transforming them makes the 
# decision boundaries much more effective.
X_log = np.log1p(X.clip(lower=0)) # Handle negative values if any exist

# 3. ROBUST PARAMETERS
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 5,  # L1 regularization to prune noisy features
    'lambda': 5, # L2 regularization for stability
    'nthread': 8
}

# 4. CROSS-VALIDATION
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_preds = []
all_actual = []

print("Running Optimized Fresh Model (Log-scaled)...")

for train_idx, test_idx in kf.split(X_log):
    X_tr, X_te = X_log.iloc[train_idx], X_log.iloc[test_idx]
    y_tr, y_te = y_lfc[train_idx], y_lfc[test_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest = xgb.DMatrix(X_te)
    
    # Training from scratch for 200 rounds with early stopping
    model = xgb.train(params, dtrain, num_boost_round=200)
    
    all_preds.extend(model.predict(dtest))
    all_actual.extend(y_te)

# 5. FINAL EVALUATION
rho, pval = spearmanr(all_actual, all_preds)

print("\n" + "="*35)
print(f"OPTIMIZED SPEARMAN RHO: {rho:.4f}")
print(f"P-VALUE:               {pval:.4e}")
print("="*35)

# 6. FEATURE IMPORTANCE
# Let's see if Epigenetics or Sequence wins after log-scaling
final_model = xgb.train(params, xgb.DMatrix(X_log, label=y_lfc), num_boost_round=100)
importance = final_model.get_score(importance_type='gain')
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTOP 10 FEATURES BY GAIN:")
for f, v in top_features:
    print(f"{f}: {v:.4f}")

plt.switch_backend('Agg')
xgb.plot_importance(final_model, max_num_features=20, importance_type='gain')
plt.savefig('alke_optimized_importance.png')
