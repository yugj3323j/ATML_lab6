"""Vocabulary utilities for Lab 6 encoder-decoder tasks."""

from collections import Counter
from typing import Dict, Iterable, List


SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class Vocabulary:
    """Simple token-id vocabulary with special token support."""

    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}

    @property
    def pad_idx(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def sos_idx(self) -> int:
        return self.token_to_id["<SOS>"]

    @property
    def eos_idx(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_idx(self) -> int:
        return self.token_to_id["<UNK>"]

    def build(self, texts: Iterable[List[str]]) -> None:
        counter = Counter()
        for tokens in texts:
            counter.update(tokens)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, tokens: List[str], max_len: int) -> List[int]:
        ids = [self.sos_idx]
        ids.extend([self.token_to_id.get(tok, self.unk_idx) for tok in tokens[: max_len - 2]])
        ids.append(self.eos_idx)
        if len(ids) < max_len:
            ids.extend([self.pad_idx] * (max_len - len(ids)))
        return ids[:max_len]

    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        out = []
        for idx in ids:
            tok = self.id_to_token.get(int(idx), "<UNK>")
            if tok == "<EOS>":
                break
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return out

    def __len__(self) -> int:
        return len(self.token_to_id)
