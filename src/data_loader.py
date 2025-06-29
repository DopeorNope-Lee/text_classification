"""Lightweight DataLoader – no disk writes.

Provides `get_dataset()` that returns a tokenised `DatasetDict` and
`label2id` mapping, ready for training regardless of backbone.
"""
from __future__ import annotations

import os
from typing import Tuple, Dict

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("BASE_MODEL", "skt/kobert-base-v1")
_MAX_LEN = int(os.getenv("MAX_LEN", 128))


def get_dataset(tokenizer, max_len):
    """Return `(tokenised_datasetdict, label2id)` without touching disk."""
    raw_ds = load_dataset("hblim/customer-complaints")
    id2label = raw_ds["train"].features["label"].names  # ['billing', 'delivery', 'product']
    label2id = {lbl: idx for idx, lbl in enumerate(id2label)}

    def _encode(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)
        enc["labels"] = batch["label"]  # already int
        return enc

    tokenised = raw_ds.map(_encode, batched=True, remove_columns=["text", "label"])
    tokenised.set_format(type="torch")
    print('Dataset loaded')
    return tokenised, label2id