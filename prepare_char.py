#!/usr/bin/env python3
"""
Prepare character-level training data from input.txt.

Usage: python prepare_char.py <dataset_name>
"""
import os
import sys
import pickle
import numpy as np


def prepare_dataset(dataset_name):
    data_dir = os.path.join("data", dataset_name)
    input_file = os.path.join(data_dir, "input.txt")

    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    print(f"Read {len(data):,} characters")

    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"Unique characters: {vocab_size}")

    ids = [stoi[ch] for ch in data]
    n = len(ids)
    train_ids = ids[:int(n * 0.9)]
    val_ids = ids[int(n * 0.9):]

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, "train.bin"))
    val_ids.tofile(os.path.join(data_dir, "val.bin"))

    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Done! Created train.bin, val.bin, meta.pkl in {data_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_char.py <dataset_name>")
        sys.exit(1)
    prepare_dataset(sys.argv[1])
