import gzip
import os
import sys

from Bio import SeqIO, SeqRecord
from tqdm import tqdm

raw_data_path = "datasets/raw/spike_nuc_16.fasta.gz"
output_data_path = "datasets/raw/spikenuc0415_clean_n_16.fasta"

out = []
n = 0

with gzip.open(raw_data_path, "rt") as handle:
    for seq_record in tqdm(SeqIO.parse(handle, "fasta")):
        if ("N" or "n") not in str(seq_record.seq):
            out.append(seq_record)
        # n+=1
        # if n >= 11675917:
        #     print(seq_record)

        # if n >= 500000:
        #     break

print(n)
print(len(out))
SeqIO.write(out, output_data_path, "fasta")