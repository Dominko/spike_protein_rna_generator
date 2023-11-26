import gzip
import random
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from Bio import SeqIO, SeqRecord
from Bio.SeqUtils import CodonAdaptationIndex
from sklearn.decomposition import PCA
from tqdm import tqdm

paths = []
out_paths = []

base_path = "datasets/raw/spike_nuc_X.fasta.gz"

output_data_path = "datasets/raw/spike_nuc_clean_95_percent_X"

pattern = re.compile("[^AatTgGcC*?]")

lens = [3813, 3816, 3807, 3822, 3804]

for i in range(0, 15):
    path = base_path.replace("X", str(i+1))
    out_path = output_data_path.replace("X", str(i+1))
    out_paths.append(out_path)
    paths.append(path)

def is_gene_valid(seq):
    if len(seq) % 3 != 0:
        return False
    if re.search(pattern, str(seq)):
        return False
    
    return True


for i in range(len(paths)):
    valid = []
    wrong_start = 0
    outliers = 0
    path = paths[i]
    out_path = out_paths[i]
    print(path)
    print(out_path)
    with gzip.open(path, "rt") as handle:
        for seq_record in tqdm(SeqIO.parse(handle, "fasta")):
            if len(seq_record.seq) in lens and is_gene_valid(seq_record.seq):
                # print(len(seq_record.seq))
                triplets = [seq_record.seq[i:i+3] for i in range(0, len(seq_record.seq), 3)]
                if not 'TAG' in triplets[0 : len(triplets)-1] and not 'TAA' in triplets[0 : len(triplets)-1] and not 'TGA' in triplets[0 : len(triplets)-1]:
                    # print(len(seq_record.seq))
                    valid.append(str(seq_record.seq))
                else:
                    wrong_start += 1



    print(outliers)
    print(len(valid))
    print(len(set(valid)))

    valid = set(valid)
    with open(out_path,'w') as f:
        for string in valid:
            f.write(string + ",\r\n")
