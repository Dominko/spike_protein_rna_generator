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

base_path = "datasets/test"

output_data_path = "datasets/test.fasta"

out = []
with open(base_path, "r") as handle:
    for line in tqdm(handle.readlines()):
        record = SeqRecord(Seq(line.strip(",\r\n")))
        out.append(record)
        

# valid = set(valid)
with open(output_data_path,'w') as f:
    SeqIO.write(out, output_data_path, "fasta")