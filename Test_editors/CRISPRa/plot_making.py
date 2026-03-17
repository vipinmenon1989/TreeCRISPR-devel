import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Blunt Truth: Required for headless HPC to prevent X11 display crashing
import matplotlib
matplotlib.use('Agg')

# 1. Configuration
input_dir = './'  # Directory containing your individual CSVs
output_dir = 'evaluation_results_pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

metrics_map = {
    'AUC': 'AUC',
    'AUPR': 'AUPRC',  # Maps your requested metric to the actual CSV column
    'F1': 'F1',
    'Precision': 'Precision',
    'Recall': 'Recall'
}

# 2. Integration
all_files = glob.glob(os.path.join(input_dir, "*.csv"))
if not all_files:
    raise FileNotFoundError("No CSV files found in the specified directory.")

# Append all dataframes into a list, then concatenate (O(n) efficiency)
df_list = [pd.read_csv(f) for f in all_files]
full_df = pd.concat(df_list, ignore_index=True)

# 3. Data Transformation & Trimming
# Trims dataset names (e.g., "ALKE_test_dataset.csv" -> "ALKE")
full_df['Dataset_Clean'] = full_df['Dataset'].astype(str).str.split('_').str[0]

# Ensure metric columns are numeric to prevent plotting errors
for col in metrics_map.values():
    if col in full_df.columns:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

# Explicitly save the integrated dataset so you can verify the concatenation
master_csv_path = os.path.join(output_dir, 'integrated_master_dataset.csv')
full_df.to_csv(master_csv_path, index=False)
print(f"Verified Integration: Master dataset saved to {master_csv_path}")

# 4. Plotting & PDF Generation
sns.set_theme(style="whitegrid", context="talk")

for label, col_name in metrics_map.items():
    if col_name not in full_df.columns:
        print(f"Skipping {label}: Column {col_name} not found.")
        continue
    
    plt.figure(figsize=(12, 6))
    
    # Dataset on X-axis, Model as Hue for comparative analysis
    sns.barplot(
        data=full_df, 
        x='Dataset_Clean', 
        y=col_name, 
        hue='Model', 
        palette='viridis'
    )
    
    plt.title(f'Comparative Analysis: {label}', fontsize=16, fontweight='bold')
    plt.ylabel(f'{label} Score')
    plt.xlabel('Dataset')
    plt.ylim(0, 1.1)
    
    # Prevent legend from overlapping bars
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Model')
    
    # Save as PDF
    save_path = os.path.join(output_dir, f'barplot_{label}.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()  # Clears memory
    
    print(f"Generated PDF: {save_path}")

print("All tasks completed.")
