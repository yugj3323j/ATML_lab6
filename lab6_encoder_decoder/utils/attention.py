"""Bahdanau attention implementation for sequence-to-sequence models."""

from typing import Tuple

import torch
from torch import nn


class BahdanauAttention(nn.Module):
    """Additive attention: score(h_t, h_s) = v^T tanh(W_h h_t + W_s h_s)."""

    def __init__(self, decoder_hidden_size: int, encoder_hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(decoder_hidden_size, decoder_hidden_size, bias=False)
        self.key = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        self.energy = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # decoder_hidden: [batch, dec_hidden]
        # encoder_outputs: [batch, src_len, enc_hidden]
        q = self.query(decoder_hidden).unsqueeze(1)  # [batch, 1, dec_hidden]
        k = self.key(encoder_outputs)  # [batch, src_len, dec_hidden]
        scores = self.energy(torch.tanh(q + k)).squeeze(-1)  # [batch, src_len]
        scores = scores.masked_fill(~src_mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
