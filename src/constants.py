import numpy as np

CODON_INDICES = {
    "AAA": 0,
    "AAC": 1,
    "AAG": 2,
    "AAT": 3,

    "ACA": 4,
    "ACC": 5,
    "ACG": 6,
    "ACT": 7,
 
    "AGA": 8,
    "AGC": 9,
    "AGG": 10,
    "AGT": 11,
 
    "ATA": 12,
    "ATC": 13,
    "ATG": 14,
    "ATT": 15,
 
    "CAA": 16,
    "CAC": 17,
    "CAG": 18,
    "CAT": 19,

    "CCA": 20,
    "CCC": 21,
    "CCG": 22,
    "CCT": 23,
 
    "CGA": 24,
    "CGC": 25,
    "CGG": 26,
    "CGT": 27,
 
    "CTA": 28,
    "CTC": 29,
    "CTG": 30,
    "CTT": 31,

    "GAA": 32,
    "GAC": 33,
    "GAG": 34,
    "GAT": 35,

    "GCA": 36,
    "GCC": 37,
    "GCG": 38,
    "GCT": 39,
 
    "GGA": 40,
    "GGC": 41,
    "GGG": 42,
    "GGT": 43,
 
    "GTA": 44,
    "GTC": 45,
    "GTG": 46,
    "GTT": 47,

    "TAA": 48,
    "TAC": 49,
    "TAG": 50,
    "TAT": 51,

    "TCA": 52,
    "TCC": 53,
    "TCG": 54,
    "TCT": 55,
 
    "TGA": 56,
    "TGC": 57,
    "TGG": 58,
    "TGT": 59,
 
    "TTA": 60,
    "TTC": 61,
    "TTG": 62,
    "TTT": 63,

    "": 64
}

CODONS = list(CODON_INDICES.keys())
START_TOKEN = ">"

CAI_TEMPLATE = np.ones(64)
# IMMUNOGENICITY_Q1 = 50.667
# IMMUNOGENICITY_Q3 = 51.167