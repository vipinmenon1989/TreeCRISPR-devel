import sys
def bedmaker(file1):
    f = open(file1,'r')
    out = open('CRISPR_histone.bed','w')
    f = f.readlines()
    del f[0]
    for line in f:
        info = line.strip().split(',')
        chroms = info[3]
        start = info[4]
        end = info[5]
        strand = info[6]
        score = '0'
        attri = info[0]
        rgb = '255,0,0'
        out.write(str(chroms) + '\t' + str(start) + '\t' + str(end) + '\t' + str(attri) + '\t' + str(score) + '\t' + str(strand) + '\t' + str(rgb) + '\n')

    out.close()
    return out
args = sys.argv[1]
result = bedmaker(args)
print (result)




