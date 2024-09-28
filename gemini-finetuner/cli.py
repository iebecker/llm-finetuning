import os
import argparse
import pandas as pd
import json
import time
import glob
from google.cloud import storage

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]


def train():
    print("train()")


def main(args=None):
    print("CLI Arguments:", args)

    if args.generate:
        train()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model",
    )


    args = parser.parse_args()

    main(args)