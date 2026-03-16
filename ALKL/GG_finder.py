import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless mode for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, binomtest
import argparse
import sys

def run_qc_analysis():
    parser = argparse.ArgumentParser(description="Cas9 gRNA Motif QC and LFC Analysis")
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument("--plot", default="motif_qc_combined.png", help="Output image filename")
    parser.add_argument("--report", default="motif_qc_report.txt", help="Output text report filename")
    parser.add_argument("--target", type=float, default=0.90, help="Target purity (0.90 = 90%)")
    args = parser.parse_args()

    # 1. Load Data
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # 2. Filter: LFC between -1.5 and -0.5
    subset = df[(df['lfc'] <= -0.5) & (df['lfc'] >= -1.5)].copy()
    subset['Group'] = subset['GG_motif'].apply(lambda x: 'GG' if x == 'GG' else 'non-GG')

    # 3. Counts & Purity
    counts = subset['Group'].value_counts()
    n_gg = int(counts.get('GG', 0))
    n_non_gg = int(counts.get('non-GG', 0))
    total = n_gg + n_non_gg
    purity = (n_gg / total) * 100 if total > 0 else 0.0

    # 4. Statistics
    # Mann-Whitney U (Distribution difference)
    mw_p_val = 1.0
    if n_gg > 0 and n_non_gg > 0:
        _, mw_p_val = mannwhitneyu(subset[subset['Group']=='GG']['lfc'], 
                                   subset[subset['Group']=='non-GG']['lfc'])
    
    # Binomial Test (Purity significance)
    # Using n=total for compatibility with newer Scipy versions
    binom_p = 1.0
    if total > 0:
        res = binomtest(n_gg, n=total, p=args.target, alternative='less')
        binom_p = res.pvalue

    # 5. Write Report
    with open(args.report, 'w') as f:
        f.write("CAS9 MOTIF QC & LFC ANALYSIS\n")
        f.write("============================\n")
        f.write(f"Total Sequences (-1.5 <= LFC <= -0.5): {total}\n")
        f.write(f"GG Counts: {n_gg}\n")
        f.write(f"non-GG Counts: {n_non_gg}\n")
        f.write(f"Purity: {purity:.2f}% (Target: {args.target*100}%)\n\n")
        f.write(f"STATISTICS:\n")
        f.write(f"Purity P-value (Binomial): {binom_p:.4e}\n")
        f.write(f"LFC Dist P-value (Mann-Whitney): {mw_p_val:.4e}\n")
        
        if binom_p < 0.05:
            f.write("\nCONCLUSION: FAILED. Purity is significantly below target.\n")
        else:
            f.write("\nCONCLUSION: PASSED purity check.\n")

    # 6. Generate BOTH Plots (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- LEFT PLOT: LFC Distribution (Box + Strip) ---
    sns.boxplot(x='Group', y='lfc', data=subset, ax=axes[0], 
                palette='viridis', order=['non-GG', 'GG'])
    sns.stripplot(x='Group', y='lfc', data=subset, ax=axes[0], 
                  color='black', alpha=0.3, jitter=True, order=['non-GG', 'GG'])
    
    axes[0].set_title(f"LFC Distribution\n(MWU P-val: {mw_p_val:.4e})")
    axes[0].set_ylabel("Log Fold Change (LFC)")
    axes[0].set_ylim(-1.5, -0.5)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # --- RIGHT PLOT: Counts (Bar Plot) ---
    # Prepare data for bar plot
    bar_data = pd.DataFrame({
        'Motif': ['GG', 'non-GG'],
        'Count': [n_gg, n_non_gg]
    })
    
    sns.barplot(x='Motif', y='Count', data=bar_data, ax=axes[1], 
                palette='viridis', order=['non-GG', 'GG'])
    
    # Add text labels on top of bars
    for index, row in bar_data.iterrows():
        # Find the correct x location based on order
        x_loc = 1 if row['Motif'] == 'GG' else 0
        axes[1].text(x_loc, row['Count'], f"{row['Count']}", 
                     color='black', ha="center", va="bottom", fontweight='bold')

    axes[1].set_title(f"Sequence Counts\n(Purity: {purity:.2f}%)")
    axes[1].set_ylabel("Number of gRNAs")

    # Final Layout Adjustments
    plt.tight_layout()
    plt.savefig(args.plot, dpi=300)
    print(f"Done. Report: {args.report} | Plot: {args.plot}")

if __name__ == "__main__":
    run_qc_analysis()