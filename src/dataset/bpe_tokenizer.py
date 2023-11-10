import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from ..constants import CODON_INDICES, CODONS, START_TOKEN
from ..dataset.bpe_trainer import BPE_Trainer


class BPE_Tokenizer:
    def __init__(
        self,
        tokenizer_path,
        max_seq_len: int,
        one_hot=True,
        prepend_start_token=False
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.prepend_start_token = prepend_start_token
        self.one_hot = one_hot
        self.max_seq_len = max_seq_len
        # if prepend_start_token:
        #     self.max_seq_len += 1
        self.tokenizer.enable_padding(direction="right", pad_id=1, length=self.max_seq_len)
        

        

    def encode(self, sequence):
        enc = []
        # for aa in sequence[: self.max_seq_len]:
        if self.prepend_start_token:
            enc.append(self.tokenizer.get_vocab()["<BOS>"])

        sequence = BPE_Trainer.preencode_nuc(sequence.strip(">"))

        enc = enc + self.tokenizer.encode(sequence).ids

        # print(len(enc))

        if self.one_hot:
            return F.one_hot(torch.tensor(enc), len(self.tokenizer.get_vocab())).float()
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
            # for seq_idx in range(seq_len):
            #     seq += self.dec_dict[int(h[batch_idx][seq_idx])]
            seq = self.tokenizer.decode_batch(h[batch_idx])
            batch_seq.append(seq)

        return batch_seq

