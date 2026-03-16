import pyBigWig
from pybedtools import BedTool
import sys
import numpy as np
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_signals_for_bigwig(bigwig_file, bed_file, extension):
    """
    For a given bigwig and extension value, compute the summed chromatin signal 
    for each interval (defined in bed_file). Returns a dictionary mapping each interval ID to the signal.
    """
    results = {}
    bw = pyBigWig.open(bigwig_file)
    for interval in BedTool(bed_file):
        chrom = interval.chrom
        # Extend the interval upstream and downstream by extension
        start = int(interval.start) - extension
        end = int(interval.end) + extension
        interval_id = interval[3]  # assuming the ID is in the 4th column
        if chrom not in bw.chroms():
            continue
        try:
            chromatin_signal = bw.values(chrom, start, end)
            # Replace NaN values with zero
            chromatin_signal = [0 if np.isnan(x) else x for x in chromatin_signal]
            total_signal = sum(chromatin_signal)
            results.setdefault(interval_id, total_signal)
        except RuntimeError as e:
            print(f"Error fetching values for {chrom}:{start}-{end} in {bigwig_file}: {e}")
    bw.close()
    return results

def process_and_save_combined(bigwig_files, bed_file, extensions, output_csv='chromatin_signals_combined.csv'):
    """
    Process all provided bigwig files and extensions concurrently, and save the combined results 
    to a CSV file. The header columns are constructed as 'bigwigBase_extension'.
    """
    # Get the list of intervals (IDs) from the bed file (preserving order)
    interval_ids = [interval[3] for interval in BedTool(bed_file)]
    unique_ids = list(dict.fromkeys(interval_ids))
    
    # Prepare a dictionary to store results for each combination.
    # The key will be a column name and the value the signal dictionary.
    combined_results = {}
    header_cols = []
    
    # Prepare tasks for parallel processing: each task is a (bigwig_file, extension) combination.
    tasks = []
    with ProcessPoolExecutor() as executor:
        future_to_key = {}
        for bw_file in bigwig_files:
            base_name = os.path.basename(bw_file)
            base_name = os.path.splitext(base_name)[0]
            for ext in extensions:
                col_name = f"{base_name}_{ext}"
                header_cols.append(col_name)
                future = executor.submit(compute_signals_for_bigwig, bw_file, bed_file, ext)
                future_to_key[future] = col_name

        # Collect the results as they complete
        for future in as_completed(future_to_key):
            col_name = future_to_key[future]
            try:
                result = future.result()
                combined_results[col_name] = result
            except Exception as exc:
                print(f"{col_name} generated an exception: {exc}")
    
    # Prepare a dictionary to store the final combined results for each interval ID.
    all_results = {id_: {} for id_ in unique_ids}
    for col in header_cols:
        signal_dict = combined_results.get(col, {})
        for id_ in unique_ids:
            all_results[id_][col] = signal_dict.get(id_, 0)
    
    # Write combined results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header: first column is ID then each bigwig/extension combination
        writer.writerow(['ID'] + header_cols)
        for id_ in unique_ids:
            row = [id_]
            for col in header_cols:
                row.append(all_results[id_].get(col, 0))
            writer.writerow(row)
    
    print(f"Combined results saved to {output_csv}")

if __name__ == "__main__":
    # Example usage:
    # python script.py my_intervals.bed "500,150,250" H2AZ.bigwig H3K4me1.bigwig ...
    bed_file = sys.argv[1]
    extensions = list(map(int, sys.argv[2].split(',')))
    bigwig_files = sys.argv[3:]
    process_and_save_combined(bigwig_files, bed_file, extensions)
