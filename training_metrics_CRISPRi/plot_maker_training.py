import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

# Essential for headless HPC
import matplotlib
matplotlib.use('Agg')

# 1. Configuration
input_dir = './'  # Directory containing your training CSVs
output_dir = 'training_metrics_pdfs'
os.makedirs(output_dir, exist_ok=True)

metrics_map = {
    'AUC': 'AUC',
    'AUPR': 'AUPRC', 
    'F1': 'F1',
    'Precision': 'Precision',
    'Recall': 'Recall'
}

# 2. Integration & Data Extraction
all_files = glob.glob(os.path.join(input_dir, "*.csv"))
if not all_files:
    raise FileNotFoundError("No CSV files found in the specified directory.")

data_records = []

for f in all_files:
    # Deduce and trim the editor name: e.g., "ALKME_honest_metrics.csv" -> "ALKME"
    file_name = os.path.basename(f).replace('.csv', '').split('_')[0]
    
    # Read CSV, setting the first unnamed column as the index
    df = pd.read_csv(f, index_col=0)
    
    # Verify the file has the necessary pre-calculated rows
    if 'MEAN' in df.index and 'STD' in df.index:
        mean_vals = df.loc['MEAN']
        std_vals = df.loc['STD']
        
        # Build a dictionary for this specific editor's metrics
        record = {'Editor': file_name}
        for label, col_name in metrics_map.items():
            if col_name in df.columns:
                # Store both the height (mean) and the error bar (std)
                record[f'{label}_mean'] = float(mean_vals[col_name])
                record[f'{label}_std'] = float(std_vals[col_name])
                
        data_records.append(record)
    else:
        print(f"Warning: {f} does not contain 'MEAN' and 'STD' rows. Skipping.")

# Compile into a single integrated dataframe
full_df = pd.DataFrame(data_records)

# Explicitly save the integrated dataset for analytical verification
master_csv_path = os.path.join(output_dir, 'integrated_training_metrics.csv')
full_df.to_csv(master_csv_path, index=False)
print(f"Verified Integration: Master dataset saved to {master_csv_path}")

# 3. Plotting Logic
for label in metrics_map.keys():
    mean_col = f'{label}_mean'
    std_col = f'{label}_std'
    
    if mean_col not in full_df.columns:
        print(f"Skipping {label}: Data not found.")
        continue
    
    plt.figure(figsize=(10, 6))
    
    # Matplotlib's bar function allows direct assignment of error bars
    plt.bar(
        x=full_df['Editor'], 
        height=full_df[mean_col], 
        yerr=full_df[std_col], 
        capsize=5,             
        color='steelblue', 
        edgecolor='black',
        alpha=0.8
    )
    
    plt.title(f'Training Performance: {label}', fontsize=16, fontweight='bold')
    plt.ylabel(f'{label} Score')
    plt.xlabel('Editor')
    plt.ylim(0, 1.1)  
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save as PDF
    save_path = os.path.join(output_dir, f'training_barplot_{label}.pdf')
    plt.savefig(save_path, format='pdf')
    plt.close()  
    
    print(f"Generated PDF: {save_path}")

print("Task complete. All metric PDFs with error bars generated.")
