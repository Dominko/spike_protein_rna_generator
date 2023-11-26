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

base_path = "datasets/raw/spike_nuc_clean_95_percent_X"

output_data_path = "datasets/raw/spike_nuc_clean_95_percent_full"

pattern = re.compile("[^AatTgGcC*?]")

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
for i in range(len(paths)):
    path = paths[i]
    print(path)
    with open(path, "r") as handle:
        lines = handle.readlines()
        for line in tqdm(lines):
            line = line.replace(",", "").replace("\n", "")
            valid.append(line)

    print(len(valid))
    # print(len(set(valid)))

valid = set(valid)
print(len(valid))
# print(valid)
with open(output_data_path,'w') as f:
    for string in valid:
        # print(string)
        f.write(string + "\r\n")
