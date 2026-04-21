"""Reusable Seq2Seq wrappers for attention and non-attention models."""

from typing import List, Optional, Tuple

import torch
from torch import nn


class Seq2Seq(nn.Module):
    """Generic sequence-to-sequence wrapper."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, pad_idx: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return src != self.pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        with_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.output_layer.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        attention_maps: Optional[List[torch.Tensor]] = [] if with_attention else None

        encoder_outputs, (hidden, cell) = self.encoder(src)
        src_mask = self.make_src_mask(src)

        input_tokens = tgt[:, 0]
        for t in range(1, tgt_len):
            if with_attention:
                logits, hidden, cell, attn = self.decoder(
                    input_tokens,
                    hidden,
                    cell,
                    encoder_outputs,
                    src_mask,
                    collect_attention=True,
                )
                if attention_maps is not None and attn is not None:
                    attention_maps.append(attn.detach().cpu())
            else:
                logits, hidden, cell = self.decoder(input_tokens, hidden, cell)
            outputs[:, t] = logits
            teacher = torch.rand(1, device=src.device).item() < teacher_forcing_ratio
            next_token = logits.argmax(dim=1)
            input_tokens = tgt[:, t] if teacher else next_token
        return outputs, attention_maps
