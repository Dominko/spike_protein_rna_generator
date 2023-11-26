import argparse
import os
import sys

import torch

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

from src.dataset import bpe_trainer


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Spike Protein RNA generator"
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--preprocessed_output_path", type=str, required=True)
    parser.add_argument("--tokeniser_path", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    bpe_trainer.BPE_Trainer.train_tokenizer(args.train_path, args.preprocessed_output_path, args.tokeniser_path)

if __name__ == "__main__":
    main()