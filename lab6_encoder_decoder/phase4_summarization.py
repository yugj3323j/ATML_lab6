"""Phase 4: Abstractive summarization with attentive encoder-decoder LSTM."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from lab6_encoder_decoder.models.decoder import AttentiveLSTMDecoder
from lab6_encoder_decoder.models.encoder import LSTMEncoder
from lab6_encoder_decoder.utils.dataset import build_parallel_loaders, load_cnn_dailymail_pairs, split_pairs
from lab6_encoder_decoder.utils.metrics import rouge_scores

PHASE = "4"
EPOCHS = 300
ARTICLE_MAX = 400
SUMMARY_MAX = 100
CHECKPOINT_DIR = Path("lab6_encoder_decoder/saved_models")


class Seq2SeqSumm(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx

    def make_mask(self, src):
        return src != self.src_pad_idx

    def forward(self, src, tgt, tf_ratio=0.5):
        bsz, tgt_len = tgt.shape
        vocab_size = self.decoder.output_layer.out_features
        outputs = torch.zeros(bsz, tgt_len, vocab_size, device=src.device)
        enc_out, (hidden, cell) = self.encoder(src)
        mask = self.make_mask(src)
        inp = tgt[:, 0]
        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(inp, hidden, cell, enc_out, mask, False)
            outputs[:, t] = logits
            teacher = torch.rand(1, device=src.device).item() < tf_ratio
            inp = tgt[:, t] if teacher else logits.argmax(dim=1)
        return outputs


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def truncate_pairs(pairs):
    out = []
    for a, s in pairs:
        article = " ".join(a.split()[:ARTICLE_MAX])
        summary = " ".join(s.split()[:SUMMARY_MAX])
        out.append((article, summary))
    return out


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            out = model(src, tgt, tf_ratio=0.5)
            loss = criterion(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
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
                    o = model(cs, ct, tf_ratio=0.5)
                    l = criterion(o[:, 1:].reshape(-1, o.size(-1)), ct[:, 1:].reshape(-1))
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
        out = model(src, tgt, tf_ratio=0.0)
        loss = criterion(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        total += float(loss.item())
        pred_ids = out.argmax(dim=-1).detach().cpu().tolist()
        tgt_ids = tgt.detach().cpu().tolist()
        refs.extend([" ".join(tgt_vocab.decode(x)) for x in tgt_ids])
        preds.extend([" ".join(tgt_vocab.decode(x)) for x in pred_ids])
    return total / max(1, len(loader)), rouge_scores(refs, preds)


@torch.no_grad()
def summarize(model, src_ids, tgt_vocab, device):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    enc_out, (hidden, cell) = model.encoder(src)
    mask = model.make_mask(src)
    token = torch.tensor([tgt_vocab.sos_idx], device=device)
    out = []
    for _ in range(SUMMARY_MAX):
        logits, hidden, cell, _ = model.decoder(token, hidden, cell, enc_out, mask, False)
        token = logits.argmax(dim=1)
        tid = int(token.item())
        if tid == tgt_vocab.eos_idx:
            break
        if tid not in (tgt_vocab.pad_idx, tgt_vocab.sos_idx):
            out.append(tid)
    return tgt_vocab.decode(out)


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = truncate_pairs(load_cnn_dailymail_pairs(sample_size=5000, seed=42))
    split = split_pairs(pairs, 0.8, 0.1, seed=42)
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_parallel_loaders(
        split, ARTICLE_MAX, SUMMARY_MAX, batch_size=32
    )

    encoder = LSTMEncoder(len(src_vocab), 256, 512, 2, src_vocab.pad_idx, bidirectional=False, dropout=0.2)
    decoder = AttentiveLSTMDecoder(len(tgt_vocab), 256, 512, 512, 2, tgt_vocab.pad_idx, dropout=0.2)
    model = Seq2SeqSumm(encoder, decoder, src_vocab.pad_idx).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx, label_smoothing=0.1)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "rouge1": [], "rouge2": [], "rougeL": []}
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, r = eval_epoch(model, val_loader, criterion, device, tgt_vocab)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["rouge1"].append(r["rouge1"])
        history["rouge2"].append(r["rouge2"])
        history["rougeL"].append(r["rougeL"])
        print(
            f"[Phase {PHASE} | Epoch {epoch}/{EPOCHS}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"bleu={r['rougeL']:.4f}"
        )
        if epoch % 50 == 0:
            torch.save(
                {"model_state": model.state_dict(), "history": history},
                CHECKPOINT_DIR / f"phase4_checkpoint_epoch{epoch}.pt",
            )

    _, test_rouge = eval_epoch(model, test_loader, criterion, device, tgt_vocab)
    print(f"[Phase 4] ROUGE-1: {test_rouge['rouge1']:.4f}")
    print(f"[Phase 4] ROUGE-2: {test_rouge['rouge2']:.4f}")
    print(f"[Phase 4] ROUGE-L: {test_rouge['rougeL']:.4f}")

    print("\n[Phase 4] Sample Summaries:")
    for i in range(3):
        article, reference = split.test[i]
        src_ids = src_vocab.encode(article.lower().split(), ARTICLE_MAX)
        pred = " ".join(summarize(model, src_ids, tgt_vocab, device))
        print("------------------------------------------------------------")
        print(f"ARTICLE : {article[:400]}")
        print(f"REF SUM : {reference[:300]}")
        print(f"PRED SUM: {pred}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history,
            "src_vocab": src_vocab.token_to_id,
            "tgt_vocab": tgt_vocab.token_to_id,
            "config": {"article_max": ARTICLE_MAX, "summary_max": SUMMARY_MAX},
            "test_rouge": test_rouge,
        },
        CHECKPOINT_DIR / "phase4_summarizer_model.pt",
    )
    print("=== PHASE 4 COMPLETE ===")


if __name__ == "__main__":
    main()
