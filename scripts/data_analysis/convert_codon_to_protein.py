import gzip
import random
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from Bio import Seq, SeqIO, SeqRecord
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import CodonAdaptationIndex
from sklearn.decomposition import PCA
from tqdm import tqdm

base_path = "sample_outputs/2023_07_21__16_28_20/rnaformer_large_covid_cai__generated__de_novo.fasta"

output_data_path = "datasets/protein_out_test.fasta"


pattern = re.compile("[^AatTgGcC*?]")
def is_gene_valid(seq):
    if len(seq) % 3 != 0:
        return False
    if re.search(pattern, str(seq)):
        return False
    
    return True

out = []
for record in SeqIO.parse(base_path, "fasta"):
    if is_gene_valid(record.seq):
        print(record.seq)
        record_out = record
        record_out.seq = record.seq.translate(to_stop=False)
        out.append(record_out)
        

print(out)

# valid = set(valid)
with open(output_data_path,'w') as f:
    SeqIO.write(out, output_data_path, "fasta")