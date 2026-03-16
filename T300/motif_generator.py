import pandas as pd
import argparse

def process_crispra_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        target_pams = ['CGG', 'AGG', 'TGG', 'GGG']

        # 1. POSITIVE SET (The Anchor): lfc > 4
        # In CRISPRa, these are your primary hits of interest.
        pos_set = df[
            (df['PAM'].isin(target_pams)) & 
            (df['lfc'] > 1)
        ].copy()
        
        quota = len(pos_set)
        if quota == 0:
            print("Error: No rows found for Positive Set (LFC > 4) with target PAMs.")
            return

        print(f"Positive Set (Anchor): {quota} rows identified.")

        # 2. NEGATIVE POOL: lfc <= -4
        neg_pool = df[df['lfc'] < 0].copy()

        # 3. SELECTION FROM NEGATIVE POOL TO MATCH POSITIVE QUOTA
        # Prioritize target PAMs first
        prio_neg = neg_pool[neg_pool['PAM'].isin(target_pams)]
        rem_neg = neg_pool[~neg_pool['PAM'].isin(target_pams)]
        
        if len(neg_pool) < quota:
            print(f"Warning: Only {len(neg_pool)} total rows found with LFC <= -4. Using all available.")
            selected_neg = neg_pool
        else:
            num_prio = len(prio_neg)
            if num_prio >= quota:
                # If target PAMs exceed quota, sample from them
                selected_neg = prio_neg.sample(n=quota, random_state=42)
            else:
                # Take all target PAMs, then fill the remainder from other PAMs
                needed = quota - num_prio
                fill_rows = rem_neg.sample(n=needed, random_state=42)
                selected_neg = pd.concat([prio_neg, fill_rows])

        print(f"Negative Selection: {len(selected_neg)} rows (LFC <= -4) matched to Positive Set.")

        # 4. COMBINE AND SHUFFLE
        final_df = pd.concat([pos_set, selected_neg])
        final_df = final_df.sample(frac=1, random_state=42)
        
        final_df.to_csv(output_file, index=False)
        print(f"Success: Final balanced dataset of {len(final_df)} rows written to {output_file}.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CRISPRa data: Match Negative Set to Positive Set size.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    process_crispra_data(args.input, args.output)
