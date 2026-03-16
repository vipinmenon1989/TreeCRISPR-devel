import RNA
import sys
import numpy as np
from Bio.SeqUtils import MeltingTemp as mt

def targetsequence(input_file):
    try:
        with open(input_file, 'r') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        return f"Error: File {input_file} not found."

    out = open("CRISPRi_machine_learning_30nt.csv", 'w')

    # 1. Header Generation
    nuc = ['A', 'T', 'G', 'C']
    header = (
        list(map(lambda x: 'A' + str(x), range(1, 31))) + 
        list(map(lambda x: 'T' + str(x), range(1, 31))) + 
        list(map(lambda x: 'G' + str(x), range(1, 31))) + 
        list(map(lambda x: 'C' + str(x), range(1, 31))) + 
        list(map(lambda x: 'AA'+str(x), range(1, 30))) + 
        list(map(lambda x: 'TA' + str(x), range(1, 30))) + 
        list(map(lambda x: 'GA' + str(x), range(1, 30))) + 
        list(map(lambda x: 'CA' + str(x), range(1, 30))) + 
        list(map(lambda x: 'AT' + str(x), range(1, 30))) + 
        list(map(lambda x: 'TT' + str(x), range(1, 30))) + 
        list(map(lambda x: 'GT' + str(x), range(1, 30))) + 
        list(map(lambda x: 'CT' + str(x), range(1, 30))) + 
        list(map(lambda x: 'AG' + str(x), range(1, 30))) + 
        list(map(lambda x: 'TG' + str(x), range(1, 30))) + 
        list(map(lambda x: 'GG' + str(x), range(1, 30))) + 
        list(map(lambda x: 'CG' + str(x), range(1, 30))) + 
        list(map(lambda x: 'AC' + str(x), range(1, 30))) + 
        list(map(lambda x: 'TC' + str(x), range(1, 30))) + 
        list(map(lambda x: 'GC' + str(x), range(1, 30))) + 
        list(map(lambda x: 'CC' + str(x), range(1, 30)))
    )
    
    out.write('ID,' + ','.join(header) + ',Entropy,Energy,GCcount,Gchigh,GClow,MeltingTemperature,LFC,A,T,G,C,AA,AT,AG,AC,CA,CC,CG,CT,GA,GC,GG,GT,TA,TC,TG,TT\n')

    del lines[0] # Remove header

    dinuct = []
    for j in range(4):
        for t in range(4):
            dinuct.append(nuc[t] + nuc[j])

    # 2. Row-by-Row Processing
    for line in lines:
        if not line.strip(): 
            continue
            
        seq = line.strip().split(',')
        
        # REVERTED: Retaining the exact original ID. Duplicate IDs will just be written to new rows.
        ids = seq[0] 
        
        indel = seq[1]
        complete_sequence = seq[2].upper()
        target_sequence = seq[2][4:24].upper()
        
        row_data = []

        for y in nuc:
            for x in complete_sequence:
                row_data.append('1' if y == x else '0')

        for w in dinuct:
            for z in range(len(complete_sequence) - 1):
                row_data.append('1' if w == complete_sequence[z:z+2] else '0')

        entropy_sum = 0
        lentseq = len(target_sequence)
        if lentseq > 0:
            for ant in nuc:
                count = target_sequence.count(ant) / lentseq
                if count > 0:
                    entropy_sum += -(count * np.log2(count))
        row_data.append(str(round(entropy_sum, 1)))

        energy_cal = round(RNA.fold(target_sequence)[-1], 0)
        row_data.append(str(energy_cal))

        gc_count = target_sequence.count('G') + target_sequence.count('C')
        gc_content = round((gc_count / float(lentseq)) * 100, 0) if lentseq > 0 else 0
        row_data.append(str(gc_content))
        row_data.append('1' if gc_count >= 10 else '0')
        row_data.append('1' if gc_count < 10 else '0')

        row_data.append(str(mt.Tm_NN(target_sequence)))
        row_data.append(indel)

        row_data.append(str(complete_sequence.count('A')))
        row_data.append(str(complete_sequence.count('T')))
        row_data.append(str(complete_sequence.count('G')))
        row_data.append(str(complete_sequence.count('C')))

        dinuct_global_order = ['AA', 'AT', 'AG', 'AC', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        for d in dinuct_global_order:
            row_data.append(str(complete_sequence.count(d)))

        out.write(ids + ',' + ','.join(row_data) + '\n')

    out.close()
    return "Done"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sequence_30.py <input_sam.csv>")
        sys.exit(1)
    
    args = sys.argv[1]
    result = targetsequence(args)
    print(result)
