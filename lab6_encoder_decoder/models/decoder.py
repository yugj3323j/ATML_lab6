"""Reusable LSTM decoder modules, with and without attention."""

from typing import Optional, Tuple

import torch
from torch import nn

from lab6_encoder_decoder.utils.attention import BahdanauAttention


class LSTMDecoder(nn.Module):
    """Vanilla decoder for non-attentive seq2seq tasks."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_size: int,
        num_layers: int,
        pad_idx: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embedding(input_tokens.unsqueeze(1))
        out, (hidden, cell) = self.lstm(emb, (hidden, cell))
        logits = self.output_layer(out.squeeze(1))
        return logits, hidden, cell


class AttentiveLSTMDecoder(nn.Module):
    """Bahdanau-attentive decoder."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_size: int,
        encoder_hidden_size: int,
        num_layers: int,
        pad_idx: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_size, encoder_hidden_size)
        self.input_projection = nn.Linear(emb_dim + encoder_hidden_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size + encoder_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        collect_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        emb = self.dropout(self.embedding(input_tokens))
        dec_top = hidden[-1]
        context, attn_weights = self.attention(dec_top, encoder_outputs, src_mask)
        combined = torch.cat([emb, context], dim=-1)
        projected = self.input_projection(combined).unsqueeze(1)
        out, (hidden, cell) = self.lstm(projected, (hidden, cell))
        logits = self.output_layer(torch.cat([out.squeeze(1), context], dim=-1))
        if collect_attention:
            return logits, hidden, cell, attn_weights
        return logits, hidden, cell, None
