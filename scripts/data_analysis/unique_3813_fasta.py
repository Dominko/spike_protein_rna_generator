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

base_path = "datasets/raw/spike_nuc_X.fasta.gz"

output_data_path = "datasets/raw/spike_nuc_clean_3813.fasta"

pattern = re.compile("[^AatTgGcC*?]")

lens = [3813, 3816, 3807, 3822, 3804]

for i in range(0, 15):
    path = base_path.replace("X", str(i+1))
    paths.append(path)

def is_gene_valid(seq):
    if len(seq) % 3 != 0:
        return False
    if re.search(pattern, str(seq)):
        return False
    
    return True

valid = []
outliers = 0
wrong_start = 0
for path in paths:
    with gzip.open(path, "rt") as handle:
        for seq_record in tqdm(SeqIO.parse(handle, "fasta")):
            if len(seq_record.seq) in lens and is_gene_valid(seq_record.seq):

                triplets = [seq_record.seq[i:i+3] for i in range(0, len(seq_record.seq), 3)]
                if not 'TAG' in triplets[0 : len(triplets)-1] and not 'TAA' in triplets[0 : len(triplets)-1] and not 'TGA' in triplets[0 : len(triplets)-1]:
                    # print(len(seq_record.seq))
                    valid.append(seq_record)
                else:
                    wrong_start += 1
print(wrong_start)
print(len(valid))

# valid = set(valid)
with open(output_data_path,'w') as f:
    SeqIO.write(valid, output_data_path, "fasta")