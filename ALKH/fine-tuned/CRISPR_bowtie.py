import pandas as pd
import subprocess
import os
import sys
import argparse

def execute_mapping(input_csv, output_csv, bowtie_index):
    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        sys.exit(f"Error: Could not find input file '{input_csv}'")
        
    required_cols = ['sgrnaid', 'lfc', 'Extended_sequence']
    for col in required_cols:
        if col not in df.columns:
            sys.exit(f"Error: Missing required column '{col}'. Your file contains: {list(df.columns)}")
            
    # 1. Generate Temporary FASTA
    fasta_tmp = "temp_input.fasta"
    print("Writing temporary FASTA file...")
    with open(fasta_tmp, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['sgrnaid']}\n{row['Extended_sequence']}\n")
            
    # 2. Execute Bowtie Alignment (Bowtie 1.2.2 Syntax)
    sam_tmp = "temp_output.sam"
    bowtie_cmd = [
        "bowtie",
        "-f",           # Input is FASTA format
        "-S",           # Force SAM output format
        "-v", "0",      # Strict exact match (0 mismatches)
        bowtie_index,   # Positional Arg 1: Index basename
        fasta_tmp,      # Positional Arg 2: Query file
        sam_tmp         # Positional Arg 3: Output file
    ]
    
    print(f"Executing Bowtie: {' '.join(bowtie_cmd)}")
    try:
        subprocess.run(bowtie_cmd, capture_output=True, text=True, check=True)
        print("Bowtie alignment finished successfully.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Bowtie failed with error:\n{e.stderr}")
    except FileNotFoundError:
        sys.exit("Error: 'bowtie' command not found. Ensure Bowtie is in your PATH.")

    # 3. Parse SAM Output
    print("Parsing SAM file for genomic coordinates...")
    mapping_results = {}
    with open(sam_tmp, "r") as f:
        for line in f:
            if line.startswith("@"):
                continue
                
            cols = line.strip().split("\t")
            if len(cols) < 11:
                continue
                
            qname = cols[0]         # sgrnaid
            flag = int(cols[1])     # Bitwise FLAG
            rname = cols[2]         # chr
            pos = int(cols[3])      # 1-based leftmost mapping POS
            seq = cols[9]           # sequence length
            
            if flag & 4:            # Skip if unmapped
                continue
                
            strand = "-" if flag & 16 else "+"
            start = pos
            end = pos + len(seq) - 1
            
            mapping_results[qname] = {
                "chr": rname,
                "start": start,
                "end": end,
                "strand": strand
            }

    # 4. Merge Coordinates
    print("Merging coordinates back to original dataset...")
    df["chr"] = df["sgrnaid"].map(lambda x: mapping_results.get(x, {}).get("chr", "Unmapped"))
    df["start"] = df["sgrnaid"].map(lambda x: mapping_results.get(x, {}).get("start", "NA"))
    df["end"] = df["sgrnaid"].map(lambda x: mapping_results.get(x, {}).get("end", "NA"))
    df["strand"] = df["sgrnaid"].map(lambda x: mapping_results.get(x, {}).get("strand", "NA"))
    
    mapped_df = df[df["chr"] != "Unmapped"].copy()
    dropped = len(df) - len(mapped_df)
    print(f"Filtered out {dropped} unmapped sequences. Retained {len(mapped_df)} correctly mapped sgRNAs.")
    
    # 5. Format Output
    final_cols = ["sgrnaid", "lfc", "Extended_sequence", "chr", "start", "end", "strand"]
    mapped_df = mapped_df[final_cols]
    
    mapped_df.to_csv(output_csv, index=False)
    
    if os.path.exists(fasta_tmp):
        os.remove(fasta_tmp)
    if os.path.exists(sam_tmp):
        os.remove(sam_tmp)
        
    print(f"Pipeline complete. Final file saved as: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map sgRNA sequences to genome, retaining lfc data.")
    parser.add_argument("input_csv", help="Path to input CSV (must contain sgrnaid, lfc, Extended_sequence)")
    parser.add_argument("output_csv", help="Path for the final output CSV")
    parser.add_argument("bowtie_index", help="Path to the Bowtie index prefix")
    
    args = parser.parse_args()
    execute_mapping(args.input_csv, args.output_csv, args.bowtie_index)
