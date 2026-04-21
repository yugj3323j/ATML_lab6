"""Phase 3: English to Spanish translation with Bahdanau attention in PyTorch."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from lab6_encoder_decoder.models.decoder import AttentiveLSTMDecoder
from lab6_encoder_decoder.models.encoder import LSTMEncoder
from lab6_encoder_decoder.utils.dataset import build_parallel_loaders, load_opus_en_es, split_pairs
from lab6_encoder_decoder.utils.metrics import corpus_bleu_from_lists

PHASE = "3"
EPOCHS = 180
MAX_LEN = 30
CHECKPOINT_DIR = Path("lab6_encoder_decoder/saved_models")
OUTPUT_DIR = Path("lab6_encoder_decoder/outputs")


class Seq2SeqAttention(nn.Module):
    """Custom attentive seq2seq for phase 3."""

    def __init__(self, encoder, decoder, src_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def make_mask(self, src):
        return src != self.src_pad_idx

    def forward(self, src, tgt, tf_ratio=0.5, collect_attn=False):
        bsz, tgt_len = tgt.shape
        vocab_size = self.decoder.output_layer.out_features
        outputs = torch.zeros(bsz, tgt_len, vocab_size, device=src.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        src_mask = self.make_mask(src)
        inp = tgt[:, 0]
        attentions = []
        for t in range(1, tgt_len):
            logits, hidden, cell, attn = self.decoder(inp, hidden, cell, encoder_outputs, src_mask, collect_attn)
            outputs[:, t] = logits
            if collect_attn and attn is not None:
                attentions.append(attn.detach().cpu())
            teacher = torch.rand(1, device=src.device).item() < tf_ratio
            inp = tgt[:, t] if teacher else logits.argmax(dim=1)
        return outputs, attentions


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device, tf_ratio=0.5):
    model.train()
    total = 0.0
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            outputs, _ = model(src, tgt, tf_ratio=tf_ratio, collect_attn=False)
            loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and src.size(0) > 1:
                torch.cuda.empty_cache()
                half = src.size(0) // 2
                for cs, ct in ((src[:half], tgt[:half]), (src[half:], tgt[half:])):
                    optimizer.zero_grad(set_to_none=True)
                    out, _ = model(cs, ct, tf_ratio=tf_ratio, collect_attn=False)
                    l = criterion(out[:, 1:].reshape(-1, out.size(-1)), ct[:, 1:].reshape(-1))
                    l.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total += float(l.item()) / 2.0
            else:
                raise
    return total / max(1, len(loader))


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, tgt_vocab):
    model.eval()
    total = 0.0
    refs, preds = [], []
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        outputs, _ = model(src, tgt, tf_ratio=0.0, collect_attn=False)
        loss = criterion(outputs[:, 1:].reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
        total += float(loss.item())
        pred_ids = outputs.argmax(dim=-1).detach().cpu().tolist()
        tgt_ids = tgt.detach().cpu().tolist()
        for p, r in zip(pred_ids, tgt_ids):
            preds.append(tgt_vocab.decode(p))
            refs.append(tgt_vocab.decode(r))
    return total / max(1, len(loader)), corpus_bleu_from_lists(refs, preds)


@torch.no_grad()
def translate_with_attention(model, sentence_ids, src_vocab, tgt_vocab, device):
    src = torch.tensor([sentence_ids], dtype=torch.long, device=device)
    enc_out, (hidden, cell) = model.encoder(src)
    src_mask = model.make_mask(src)
    token = torch.tensor([tgt_vocab.sos_idx], device=device)
    words, attn_rows = [], []
    for _ in range(MAX_LEN):
        logits, hidden, cell, attn = model.decoder(token, hidden, cell, enc_out, src_mask, True)
        token = logits.argmax(dim=1)
        tok = int(token.item())
        if tok == tgt_vocab.eos_idx:
            break
        words.append(tok)
        if attn is not None:
            attn_rows.append(attn.squeeze(0).detach().cpu().numpy())
    return tgt_vocab.decode(words), np.array(attn_rows)


def save_attention_plot(attn: np.ndarray, src_tokens: List[str], pred_tokens: List[str], out_path: Path):
    plt.figure(figsize=(10, 6))
    if attn.size == 0:
        attn = np.zeros((1, max(1, len(src_tokens))))
    plt.imshow(attn, aspect="auto")
    plt.xlabel("Source tokens")
    plt.ylabel("Generated tokens")
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha="right")
    plt.yticks(range(len(pred_tokens)), pred_tokens)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_opus_en_es(sample_size=10000, seed=42)
    split = split_pairs(pairs, 0.9, 0.0, seed=42)
    train_loader, _, test_loader, src_vocab, tgt_vocab = build_parallel_loaders(split, MAX_LEN, MAX_LEN, batch_size=128)
    val_loader = test_loader

    encoder = LSTMEncoder(len(src_vocab), 256, 256, 3, src_vocab.pad_idx, bidirectional=False, dropout=0.3)
    decoder = AttentiveLSTMDecoder(len(tgt_vocab), 256, 256, 256, 3, tgt_vocab.pad_idx, dropout=0.3)
    model = Seq2SeqAttention(encoder, decoder, src_vocab.pad_idx).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "bleu": []}
    train_loss = 0.0
    val_loss = 0.0
    for epoch in range(1, EPOCHS + 1):
        denom = max(EPOCHS - 1, 1)
        tf_ratio = 0.9 - (0.8 * (epoch - 1) / denom)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, tf_ratio=tf_ratio)
        val_loss, bleu = eval_epoch(model, val_loader, criterion, device, tgt_vocab)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["bleu"].append(bleu)
        print(
            f"[Phase {PHASE} | Epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} bleu={bleu:.4f}"
        )
        if epoch % 50 == 0:
            torch.save(
                {"model_state": model.state_dict(), "history": history},
                CHECKPOINT_DIR / f"phase3_checkpoint_epoch{epoch}.pt",
            )

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        CHECKPOINT_DIR / "phase3_eng_spanish_model.pt",
    )
    print("=== PHASE 3 COMPLETE === (stopped at epoch 180)")

    test_loss, test_bleu = eval_epoch(model, test_loader, criterion, device, tgt_vocab)
    print(f"[Phase 3] Final test loss={test_loss:.4f}, BLEU={test_bleu:.4f}")

    sample_sentences = [pairs[i][0] for i in range(min(5, len(pairs)))]
    print("[Phase 3] Sample translations:")
    for s in sample_sentences:
        src_tokens = s.lower().split()
        src_ids = src_vocab.encode(src_tokens, MAX_LEN)
        pred_tokens, _ = translate_with_attention(model, src_ids, src_vocab, tgt_vocab, device)
        print(f"[Phase 3] EN: {s}")
        print(f"[Phase 3] ES: {' '.join(pred_tokens)}")

    for i in range(2):
        src_text = pairs[i][0]
        src_tokens = src_text.lower().split()[:MAX_LEN]
        src_ids = src_vocab.encode(src_tokens, MAX_LEN)
        pred_tokens, attn = translate_with_attention(model, src_ids, src_vocab, tgt_vocab, device)
        save_attention_plot(attn, src_tokens, pred_tokens, OUTPUT_DIR / f"phase3_attention_{i + 1}.png")

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "history": history,
            "src_vocab": src_vocab.token_to_id,
            "tgt_vocab": tgt_vocab.token_to_id,
            "config": {"max_len": MAX_LEN},
            "test_bleu": test_bleu,
            "test_loss": test_loss,
        },
        CHECKPOINT_DIR / "phase3_eng_spanish_model.pt",
    )


if __name__ == "__main__":
    main()
