"""Dataset loading and preprocessing utilities for all Lab 6 phases."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from lab6_encoder_decoder.utils.vocab import Vocabulary


def simple_tokenize(text: str) -> List[str]:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.split(" ")


@dataclass
class SplitData:
    train: List[Tuple[str, str]]
    val: List[Tuple[str, str]]
    test: List[Tuple[str, str]]


def split_pairs(
    pairs: Sequence[Tuple[str, str]], train_ratio: float, val_ratio: float, seed: int = 42
) -> SplitData:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return SplitData(
        train=pairs[:n_train],
        val=pairs[n_train : n_train + n_val],
        test=pairs[n_train + n_val :],
    )


def get_fallback_eng_hindi_pairs(n: int = 200) -> List[Tuple[str, str]]:
    english_templates = [
        "i am going to school",
        "he is reading a book",
        "she likes music",
        "we are learning machine learning",
        "this is a beautiful day",
        "they are playing cricket",
        "please open the door",
        "the food is very tasty",
        "where is the nearest station",
        "i need some water",
    ]
    hindi_templates = [
        "मैं स्कूल जा रहा हूँ",
        "वह एक किताब पढ़ रहा है",
        "उसे संगीत पसंद है",
        "हम मशीन लर्निंग सीख रहे हैं",
        "यह एक सुंदर दिन है",
        "वे क्रिकेट खेल रहे हैं",
        "कृपया दरवाज़ा खोलिए",
        "खाना बहुत स्वादिष्ट है",
        "निकटतम स्टेशन कहाँ है",
        "मुझे थोड़ा पानी चाहिए",
    ]
    pairs = []
    for i in range(n):
        idx = i % len(english_templates)
        pairs.append((english_templates[idx], hindi_templates[idx]))
    return pairs


def load_eng_hindi_pairs() -> Tuple[List[Tuple[str, str]], bool]:
    """Try Kaggle/local files first. Returns (pairs, is_fallback)."""
    local_candidates = [
        Path("eng_hindi.csv"),
        Path("data/eng_hindi.csv"),
        Path("data/english_hindi.csv"),
    ]
    for p in local_candidates:
        if p.exists():
            df = pd.read_csv(p)
            cols = {c.lower(): c for c in df.columns}
            eng_col = cols.get("english") or cols.get("en")
            hin_col = cols.get("hindi") or cols.get("hi")
            if eng_col and hin_col:
                pairs = [(str(e), str(h)) for e, h in zip(df[eng_col], df[hin_col])]
                if len(pairs) >= 200:
                    return pairs, False

    try:
        import kagglehub  # type: ignore

        _ = kagglehub.dataset_download("harshitx/english-hindi-translation-dataset")
    except Exception:
        pass

    return get_fallback_eng_hindi_pairs(200), True


class ParallelTextDataset(Dataset):
    """Generic parallel text dataset yielding src/tgt IDs."""

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len_src: int,
        max_len_tgt: int,
    ) -> None:
        self.pairs = list(pairs)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.pairs[idx]
        src_ids = self.src_vocab.encode(simple_tokenize(src), self.max_len_src)
        tgt_ids = self.tgt_vocab.encode(simple_tokenize(tgt), self.max_len_tgt)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def build_parallel_loaders(
    split: SplitData,
    max_len_src: int,
    max_len_tgt: int,
    batch_size: int,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    train_src_tok = [simple_tokenize(s) for s, _ in split.train]
    train_tgt_tok = [simple_tokenize(t) for _, t in split.train]
    src_vocab.build(train_src_tok)
    tgt_vocab.build(train_tgt_tok)

    train_ds = ParallelTextDataset(split.train, src_vocab, tgt_vocab, max_len_src, max_len_tgt)
    val_ds = ParallelTextDataset(split.val, src_vocab, tgt_vocab, max_len_src, max_len_tgt)
    test_ds = ParallelTextDataset(split.test, src_vocab, tgt_vocab, max_len_src, max_len_tgt)
    gen = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        src_vocab,
        tgt_vocab,
    )


def load_opus_en_es(sample_size: int = 10000, seed: int = 42) -> List[Tuple[str, str]]:
    ds = load_dataset("opus_books", "en-es", split="train")
    idx = np.random.default_rng(seed).choice(len(ds), size=min(sample_size, len(ds)), replace=False)
    pairs = []
    for i in idx:
        item = ds[int(i)]["translation"]
        pairs.append((item["en"], item["es"]))
    return pairs


def load_cnn_dailymail_pairs(sample_size: int = 5000, seed: int = 42) -> List[Tuple[str, str]]:
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
    idx = np.random.default_rng(seed).choice(len(ds), size=min(sample_size, len(ds)), replace=False)
    pairs = []
    for i in idx:
        item = ds[int(i)]
        pairs.append((item["article"], item["highlights"]))
    return pairs
