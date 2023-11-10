import numpy as np
import torch

from ..configs import DatasetConfigs
from ..constants import CAI_TEMPLATE, START_TOKEN
from .bpe_tokenizer import BPE_Tokenizer
from .tokenizer import Tokenizer


def load_sequences_file(filename):
    with open(filename, "r") as file:
        viral_seqs = file.readlines()
    return [viral_seq.replace("\n", "") for viral_seq in viral_seqs]


def load_codon_adaptation_indices(
    codon_adaptation_indices_filepath
):
    codon_adaptation_indices = np.loadtxt(codon_adaptation_indices_filepath, delimiter=",", dtype=float)
    # print(codon_adaptation_indices)
    #     if one_hot:
    #         immunogenicity_scores = np.array(
    #             [EXTRA_ATTRIBUTE_ONE_HOT[score] for score in immunogenicity_scores]
    #         )
    # else:
    #     if one_hot:
    #         immunogenicity_scores = np.array(
    #             [EXTRA_ATTRIBUTE_ONE_HOT[1] for _ in range(len(sequences))]
    #         )
    #     else:
    #         immunogenicity_scores = np.array([1 for _ in range(len(sequences))])

    return torch.tensor(codon_adaptation_indices).float()


class SequenceDataset:
    def __init__(
        self,
        dataset_configs: DatasetConfigs,
        tokenizer: str,
        split: str,
        max_seq_len: int,
        sequence_one_hot: bool = True,
        label_one_hot: bool = True,
        prepend_start_token: bool = False,
        tokenizer_path: str = "",
    ):
        self.max_seq_len = max_seq_len
        self.sequence_one_hot = sequence_one_hot
        self.label_one_hot = label_one_hot
        self.prepend_start_token = prepend_start_token
        self.load_codon_adaptation_indices = dataset_configs.load_codon_adaptation_indices

        if tokenizer == "Base":
            self.tokenizer = Tokenizer(
                self.max_seq_len, self.sequence_one_hot, prepend_start_token
            )
        if tokenizer == "BPE":
            self.tokenizer = BPE_Tokenizer(
               tokenizer_path, self.max_seq_len, self.sequence_one_hot, prepend_start_token
            )

        if split == "train":
            sequences_filepath = dataset_configs.train.sequences_path
            codon_adaptation_indices_filepath = (
                dataset_configs.train.codon_adaptation_indices_path
            )
        elif split == "val":
            sequences_filepath = dataset_configs.val.sequences_path
            codon_adaptation_indices_filepath = (
                dataset_configs.val.codon_adaptation_indices_path
            )
        elif split == "test":
            sequences_filepath = dataset_configs.test.sequences_path
            codon_adaptation_indices_filepath = (
                dataset_configs.test.codon_adaptation_indices_path
            )

        self.sequences = load_sequences_file(sequences_filepath)

        if self.load_codon_adaptation_indices:
            self.codon_adaptation_indices = load_codon_adaptation_indices(
                codon_adaptation_indices_filepath
            )
        else:
            self.codon_adaptation_indices = torch.tensor(np.vstack([CAI_TEMPLATE]*len(self.sequences))).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.prepend_start_token:
            sequence = START_TOKEN + self.sequences[idx]
        else:
            sequence = self.sequences[idx]

        return (
            self.tokenizer.encode(sequence),
            self.codon_adaptation_indices[idx],
        )