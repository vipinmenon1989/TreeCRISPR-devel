import pandas as pd
import numpy as np

df = pd.read_csv('ALKL_filtered_sample.csv')
X = df.drop(columns=['ID', 'LFC', 'class'])

# 1. Sparsity Check
sparsity = (X == 0).mean().mean()
print(f"Global Sparsity: {sparsity:.2%}")

# 2. Max Correlation Check
corrs = X.corrwith(df['LFC']).abs()
print(f"Max Correlation with LFC: {corrs.max():.4f}")
print(f"Number of features with Corr > 0.2: {(corrs > 0.2).sum()}")

# 3. Variance Check
cvs = X.std() / X.mean()
print(f"Median Coefficient of Variation: {cvs.median():.4f}")

# 4. Outlier Check (Z-score > 5)
from scipy import stats
z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
outlier_pct = (z_scores > 5).mean()
print(f"Percentage of extreme outliers: {outlier_pct:.2%}")
