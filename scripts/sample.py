import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

from src.configs import TestingConfigs, TrainingConfigs
from src.constants import CAI_TEMPLATE, START_TOKEN
from src.dataset.dataset import SequenceDataset
from src.models.lstm import VaxLSTM
from src.models.rnaformer import RNAformer
from src.models.rnaformer_single import RNAformerS

# from src.models.vaxformer import Vaxformer
from src.utils import common_utils, model_utils


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Vaxformer project"
    )
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--num_sequences", type=int, required=True)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    run_name = os.path.basename(args.config_filepath).replace(".yaml", "")
    configs = TestingConfigs(**common_utils.load_yaml(args.config_filepath))
    train_configs = TrainingConfigs(
        **common_utils.load_yaml(configs.pretrained_model_configs_path)
    )

    common_utils.setup_random_seed(configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.outputs_dir)
    )
    device = common_utils.setup_device(configs.device)
    print(f"Running on {device}")

    train_dataset = SequenceDataset(
        train_configs.dataset_configs,
        train_configs.model_configs.tokenizer,
        "train",
        train_configs.model_configs.hyperparameters.max_seq_len,
        sequence_one_hot=False,
        label_one_hot=False,
        prepend_start_token=True,
        tokenizer_path=train_configs.model_configs.tokenizer_path
    )

    if train_configs.model_configs.model_type == "rnaformer":
        kwargs = {
            # "padding_idx": train_dataset.tokenizer.enc_dict["-"],
            # "start_idx": train_dataset.tokenizer.enc_dict[START_TOKEN],
            "start_idx": 0,
        }
        model = RNAformer(train_configs.model_configs, device, **kwargs).to(device)

        model.load_state_dict(
            torch.load(configs.pretrained_model_state_dict_path, device)[
                "model_state_dict"
            ]
        )

        model.eval()

        desired_CAI = None
        if configs.desired_cai_path:
            desired_CAI = np.loadtxt(configs.desired_cai_path, delimiter=",", dtype=float)
        else:
            desired_CAI = CAI_TEMPLATE
        print(desired_CAI)

        generated_seqs = {
            "generated": train_dataset.tokenizer.decode(
                model.generate_sequences(
                    args.num_sequences, desired_CAI, temperature=0.8, batch_size=20, topk=configs.topk
                )
            )
        }

        immunogenicity_seqs = {}
        for immunogenicity, sequences in generated_seqs.items():
            sequences_count = defaultdict(lambda: 0)
            for sequence in sequences:
                sequences_count[sequence] += 1
            immunogenicity_seqs[immunogenicity] = sequences_count

    elif train_configs.model_configs.model_type == "rnaformer_single":
        kwargs = {
            # "padding_idx": train_dataset.tokenizer.enc_dict["-"],
            # "start_idx": train_dataset.tokenizer.enc_dict[START_TOKEN],
            "start_idx": 0,
        }
        model = RNAformerS(train_configs.model_configs, device, **kwargs).to(device)

        model.load_state_dict(
            torch.load(configs.pretrained_model_state_dict_path, device)[
                "model_state_dict"
            ]
        )

        model.eval()

        desired_CAI = None
        if configs.desired_cai_path:
            desired_CAI = np.loadtxt(configs.desired_cai_path, delimiter=",", dtype=float)
        else:
            desired_CAI = CAI_TEMPLATE
        print(desired_CAI)

        generated_seqs = {
            "generated": train_dataset.tokenizer.decode(
                model.generate_sequences(
                    args.num_sequences, desired_CAI, temperature=0.8, batch_size=20, topk=configs.topk
                )
            )
        }

        immunogenicity_seqs = {}
        for immunogenicity, sequences in generated_seqs.items():
            sequences_count = defaultdict(lambda: 0)
            for sequence in sequences:
                sequences_count[sequence] += 1
            immunogenicity_seqs[immunogenicity] = sequences_count

    elif train_configs.model_configs.model_type == "lstm":
        kwargs = {
            "padding_idx": train_dataset.tokenizer.enc_dict["-"],
            "start_idx": train_dataset.tokenizer.enc_dict[START_TOKEN],
        }
        model = VaxLSTM(train_configs.model_configs, device, **kwargs).to(device)

        model.load_state_dict(
            torch.load(configs.pretrained_model_state_dict_path, device)[
                "model_state_dict"
            ]
        )

        model.eval()

        generated_seqs = {
            "low": train_dataset.tokenizer.decode(
                model.generate_sequences(
                    args.num_sequences, 0, temperature=0.8, batch_size=500
                )
            ),
            "intermediate": train_dataset.tokenizer.decode(
                model.generate_sequences(
                    args.num_sequences, 1, temperature=0.8, batch_size=500
                )
            ),
            "high": train_dataset.tokenizer.decode(
                model.generate_sequences(
                    args.num_sequences, 2, temperature=0.8, batch_size=500
                )
            ),
        }

        immunogenicity_seqs = {}
        for immunogenicity, sequences in generated_seqs.items():
            sequences_count = defaultdict(lambda: 0)
            for sequence in sequences:
                sequences_count[sequence] += 1
            immunogenicity_seqs[immunogenicity] = sequences_count

    de_novo_sequences = {
        immunogencity: {} for immunogencity in list(immunogenicity_seqs.keys())
    }
    for immunogenicity, sequences_dict in immunogenicity_seqs.items():
        print(f" ====== {immunogenicity} immunogenicity sequences ======")
        for sequence, count in sequences_dict.items():
            # Make sure the model is not copying from train data
            if sequence not in train_dataset.sequences:
                de_novo_sequences[immunogenicity].update({sequence: count})

        print(
            f"Generated: {len(sequences_dict)} --- De novo: {len(de_novo_sequences[immunogenicity])}"
        )

        full_filename = os.path.join(
            outputs_dir, f"{run_name}__{immunogenicity}__full.fasta"
        )
        de_novo_filename = os.path.join(
            outputs_dir, f"{run_name}__{immunogenicity}__de_novo.fasta"
        )

        train_config_filepath = os.path.join(
            outputs_dir, "train_config.yaml"
        )
        test_config_filepath = os.path.join(
            outputs_dir, "test_config.yaml"
        )

        common_utils.save_training_configs(train_configs, outputs_dir)
        common_utils.save_test_configs(configs, outputs_dir)

        model_utils.write_sequences_to_fasta(sequences_dict, full_filename)
        model_utils.write_sequences_to_fasta(
            de_novo_sequences[immunogenicity], de_novo_filename
        )


if __name__ == "__main__":
    main()
