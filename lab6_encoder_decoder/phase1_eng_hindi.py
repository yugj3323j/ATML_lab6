"""Phase 1: English to Hindi translation using LSTM encoder-decoder in PyTorch."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lab6_encoder_decoder.models.decoder import LSTMDecoder
from lab6_encoder_decoder.models.encoder import LSTMEncoder
from lab6_encoder_decoder.models.seq2seq import Seq2Seq
from lab6_encoder_decoder.utils.dataset import build_parallel_loaders, load_eng_hindi_pairs, split_pairs
from lab6_encoder_decoder.utils.metrics import corpus_bleu_from_lists

PHASE = "1"
EPOCHS = 300
MAX_LEN = 30
INIT_BATCH_SIZE = 64
CHECKPOINT_DIR = Path("lab6_encoder_decoder/saved_models")
OUTPUT_DIR = Path("lab6_encoder_decoder/outputs")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(src_vocab_size: int, tgt_vocab_size: int, src_pad_idx: int, tgt_pad_idx: int) -> Seq2Seq:
    encoder = LSTMEncoder(
        vocab_size=src_vocab_size,
        emb_dim=256,
        hidden_size=512,
        num_layers=2,
        pad_idx=src_pad_idx,
        bidirectional=True,
        dropout=0.2,
    )
    decoder = LSTMDecoder(
        vocab_size=tgt_vocab_size, emb_dim=256, hidden_size=1024, num_layers=2, pad_idx=tgt_pad_idx, dropout=0.2
    )
    return Seq2Seq(encoder, decoder, pad_idx=src_pad_idx)


def bridge_bidirectional_states(hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert [layers*2, batch, 512] to [layers, batch, 1024]
    layers_times_dirs, batch, h = hidden.shape
    layers = layers_times_dirs // 2
    hidden = hidden.view(layers, 2, batch, h)
    cell = cell.view(layers, 2, batch, h)
    hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
    cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
    return hidden, cell


def forward_with_bridge(model: Seq2Seq, src: torch.Tensor, tgt: torch.Tensor, tf: float) -> torch.Tensor:
    batch_size, tgt_len = tgt.size()
    vocab_size = model.decoder.output_layer.out_features
    outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

    encoder_outputs, (hidden, cell) = model.encoder(src)
    hidden, cell = bridge_bidirectional_states(hidden, cell)
    input_tokens = tgt[:, 0]
    for t in range(1, tgt_len):
        logits, hidden, cell = model.decoder(input_tokens, hidden, cell)
        outputs[:, t] = logits
        teacher = torch.rand(1, device=src.device).item() < tf
        input_tokens = tgt[:, t] if teacher else logits.argmax(dim=1)
    return outputs


def train_or_eval_epoch(
    model: Seq2Seq,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool = True,
    tf_ratio: float = 0.5,
) -> float:
    model.train(train)
    total_loss = 0.0
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        try:
            outputs = forward_with_bridge(model, src, tgt, tf_ratio)
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.item())
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and src.size(0) > 1:
                torch.cuda.empty_cache()
                half = src.size(0) // 2
                chunks = [(src[:half], tgt[:half]), (src[half:], tgt[half:])]
                chunk_loss = 0.0
                for cs, ct in chunks:
                    if train:
                        optimizer.zero_grad(set_to_none=True)
                    out = forward_with_bridge(model, cs, ct, tf_ratio)
                    l = criterion(out[:, 1:].reshape(-1, out.size(-1)), ct[:, 1:].reshape(-1))
                    if train:
                        l.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    chunk_loss += float(l.item())
                total_loss += chunk_loss / 2.0
                continue
            raise
    return total_loss / max(1, len(loader))


@torch.no_grad()
def greedy_decode(model: Seq2Seq, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = MAX_LEN) -> List[List[int]]:
    model.eval()
    encoder_outputs, (hidden, cell) = model.encoder(src)
    hidden, cell = bridge_bidirectional_states(hidden, cell)
    bsz = src.size(0)
    tokens = torch.full((bsz,), sos_idx, device=src.device, dtype=torch.long)
    finished = torch.zeros(bsz, dtype=torch.bool, device=src.device)
    out_ids = [[] for _ in range(bsz)]
    for _ in range(max_len):
        logits, hidden, cell = model.decoder(tokens, hidden, cell)
        tokens = logits.argmax(dim=1)
        for i, tok in enumerate(tokens.tolist()):
            if not finished[i]:
                out_ids[i].append(tok)
                if tok == eos_idx:
                    finished[i] = True
    return out_ids


def evaluate_bleu(model: Seq2Seq, loader, tgt_vocab, device: torch.device) -> float:
    refs, preds = [], []
    for src, tgt in loader:
        src = src.to(device)
        pred_ids = greedy_decode(model, src, tgt_vocab.sos_idx, tgt_vocab.eos_idx)
        for i in range(len(pred_ids)):
            refs.append(tgt_vocab.decode(tgt[i].tolist()))
            preds.append(tgt_vocab.decode(pred_ids[i]))
    return corpus_bleu_from_lists(refs, preds)


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs, fallback = load_eng_hindi_pairs()
    if fallback:
        print("[Phase 1] WARNING: Kaggle dataset unavailable, using fallback 200 pairs.")
    split = split_pairs(pairs, train_ratio=0.7, val_ratio=0.15, seed=42)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_parallel_loaders(
        split, MAX_LEN, MAX_LEN, INIT_BATCH_SIZE
    )

    model = build_model(len(src_vocab), len(tgt_vocab), src_vocab.pad_idx, tgt_vocab.pad_idx).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "bleu": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_or_eval_epoch(model, train_loader, criterion, optimizer, device, train=True, tf_ratio=0.5)
        val_loss = train_or_eval_epoch(model, val_loader, criterion, optimizer, device, train=False, tf_ratio=0.0)
        bleu = evaluate_bleu(model, val_loader, tgt_vocab, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["bleu"].append(bleu)
        print(
            f"[Phase {PHASE} | Epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} bleu={bleu:.4f}"
        )
        if epoch % 50 == 0:
            ckpt_path = CHECKPOINT_DIR / f"phase1_checkpoint_epoch{epoch}.pt"
            torch.save({"model_state": model.state_dict(), "history": history}, ckpt_path)

    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device)
    samples = [
        "i am going to school",
        "please open the door",
        "the food is very tasty",
        "we are learning machine learning",
        "where is the nearest station",
    ]
    print("\n[Phase 1] Sample translations:")
    for s in samples:
        src_ids = torch.tensor([src_vocab.encode(s.lower().split(), MAX_LEN)], device=device)
        pred_ids = greedy_decode(model, src_ids, tgt_vocab.sos_idx, tgt_vocab.eos_idx, MAX_LEN)[0]
        print(f"[Phase 1] EN: {s}")
        print(f"[Phase 1] HI: {' '.join(tgt_vocab.decode(pred_ids))}")

    final_model_path = CHECKPOINT_DIR / "phase1_eng_hindi_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "src_vocab": src_vocab.token_to_id,
            "tgt_vocab": tgt_vocab.token_to_id,
            "train_pairs": split.train,
            "val_pairs": split.val,
            "test_pairs": split.test,
            "history": history,
            "config": {"max_len": MAX_LEN},
        },
        final_model_path,
    )
    with open(OUTPUT_DIR / "phase1_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f)
    print(f"[Phase 1] Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"[Phase 1] Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"[Phase 1] Test BLEU: {test_bleu:.4f}")
    print(f"[Phase 1] Loss curve data train: {history['train_loss']}")
    print(f"[Phase 1] Loss curve data val: {history['val_loss']}")
    print("=== PHASE 1 COMPLETE ===")


if __name__ == "__main__":
    main()
