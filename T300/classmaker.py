import pandas as pd
import argparse
import sys

def assign_class(input_csv, output_csv):
    print(f"Loading data from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        sys.exit(f"Error: Could not find input file '{input_csv}'")
        
    if 'LFC' not in df.columns:
        sys.exit("Error: 'lfc' column is missing from the input file.")
        
    # Vectorized assignment: Negative lfc = 1, Zero/Positive lfc = 0
    print("Evaluating 'lfc' values and generating 'class' column...")
    df['class'] = (df['LFC'] > 0).astype(int)
    
    # Save the dataframe, retaining all original columns plus the new 'class'
    df.to_csv(output_csv, index=False)
    print(f"Processing complete. Final file saved as: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign binary class based on lfc values.")
    parser.add_argument("input_csv", help="Path to input CSV (must contain 'lfc' column)")
    parser.add_argument("output_csv", help="Path for the final output CSV")
    
    args = parser.parse_args()
    assign_class(args.input_csv, args.output_csv)
