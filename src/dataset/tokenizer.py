import numpy as np
import torch
from torch.nn import functional as F

from ..constants import CODON_INDICES, CODONS, START_TOKEN


class Tokenizer:
    def __init__(
        self,
        max_seq_len: int,
        one_hot=True,
        prepend_start_token=False,
    ):
        self.max_seq_len = max_seq_len
        self.prepend_start_token = prepend_start_token

        if prepend_start_token:
            self.enc_dict = {
                letter: idx for idx, letter in enumerate([START_TOKEN] + CODONS)
            }
            self.max_seq_len += 1
            self.dec_dict = {
                idx: amino_acid for amino_acid, idx in CODON_INDICES.items()
            }
        else:
            self.enc_dict = CODON_INDICES
        self.dec_dict = {idx: amino_acid for amino_acid, idx in self.enc_dict.items()}
        self.one_hot = one_hot

    def encode(self, sequence):
        enc = []
        sequence = sequence.ljust(self.max_seq_len, "-")
        # for aa in sequence[: self.max_seq_len]:
        if self.prepend_start_token:
            enc.append(self.enc_dict[sequence[0]])
            sequence = sequence[1:]

        for aa in [sequence[i:i+3] for i in range(0, len(sequence), 3)]:
            enc.append(self.enc_dict[aa])

        if self.one_hot:
            return F.one_hot(torch.tensor(enc), len(CODON_INDICES)).float()
        else:
            return torch.tensor(enc)

    def decode(self, batch):
        sequence_size = batch.size()
        batch_size = sequence_size[0]
        seq_len = sequence_size[1]

        batch_seq = []

        if self.one_hot:
            h = torch.max(batch, dim=-1).indices
        else:
            h = batch

        for batch_idx in range(batch_size):
            seq = ""
            for seq_idx in range(seq_len):
                seq += self.dec_dict[int(h[batch_idx][seq_idx])]
            batch_seq.append(seq)

        return batch_seq
