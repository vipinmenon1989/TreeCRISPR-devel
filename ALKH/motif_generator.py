import pandas as pd
import argparse

def process_sgrna_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        target_pams = ['CGG', 'AGG', 'TGG', 'GGG']

        # 1. NEGATIVE SET: Strict PAM AND -8 <= lfc <= -4
        neg_set = df[
            (df['PAM'].isin(target_pams)) & 
            (df['lfc'] >= -3.0) & 
            (df['lfc'] <= -0.8)
        ].copy()
        
        quota = len(neg_set)
        if quota == 0:
            print("No rows found for Negative Set (-8 to -4 with target PAMs).")
            return

        print(f"Negative Set (Strict): {quota} rows.")

        # 2. POSITIVE POOL: lfc > 0
        pos_pool = df[df['lfc'] > 0].copy()

        # 3. SELECTION FROM POSITIVE POOL
        # Prioritize target PAMs from the positive pool
        prio_pos = pos_pool[pos_pool['PAM'].isin(target_pams)]
        rem_pos = pos_pool[~pos_pool['PAM'].isin(target_pams)]
        
        if len(pos_pool) < quota:
            print(f"Error: Only {len(pos_pool)} total rows found with LFC > 0. Cannot reach {quota}.")
            selected_pos = pos_pool
        else:
            # Take all priority ones (if any), then fill from the remainder
            num_prio = len(prio_pos)
            if num_prio >= quota:
                selected_pos = prio_pos.sample(n=quota, random_state=42)
            else:
                needed = quota - num_prio
                fill_rows = rem_pos.sample(n=needed, random_state=42)
                selected_pos = pd.concat([prio_pos, fill_rows])

        print(f"Positive Selection: {len(selected_pos)} rows.")

        # 4. COMBINE BOTH SETS (142 + 142 = 284)
        final_df = pd.concat([neg_set, selected_pos])
        
        # Shuffle the final output
        final_df = final_df.sample(frac=1, random_state=42)
        
        final_df.to_csv(output_file, index=False)
        print(f"Success: Total {len(final_df)} rows written to {output_file}.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    process_sgrna_data(args.input, args.output)