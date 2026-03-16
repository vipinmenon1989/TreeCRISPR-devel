import os
import pandas as pd
import numpy as np

df_1 = pd.read_csv("chromatin_signals_combined.csv")
df_2 = pd.read_csv("CRISPRi_machine_learning_30nt.csv")
merged_df = pd.merge(df_1, df_2, on="ID", how="inner")
merged_df.to_csv('CRISPR_epigenetics.csv', index=False)
