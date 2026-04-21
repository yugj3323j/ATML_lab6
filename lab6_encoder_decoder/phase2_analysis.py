"""Phase 2: Performance analysis for the Phase 1 English-Hindi model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

from lab6_encoder_decoder.models.decoder import LSTMDecoder
from lab6_encoder_decoder.models.encoder import LSTMEncoder
from lab6_encoder_decoder.models.seq2seq import Seq2Seq
from lab6_encoder_decoder.phase1_eng_hindi import MAX_LEN, bridge_bidirectional_states
from lab6_encoder_decoder.utils.dataset import ParallelTextDataset
from lab6_encoder_decoder.utils.metrics import bleu_stats, sentence_bleu_score
from lab6_encoder_decoder.utils.vocab import Vocabulary

PHASE = "2"
MODEL_PATH = Path("lab6_encoder_decoder/saved_models/phase1_eng_hindi_model.pt")
OUTPUT_DIR = Path("lab6_encoder_decoder/outputs")


def rebuild_vocab(mapping):
    vocab = Vocabulary()
    vocab.token_to_id = dict(mapping)
    vocab.id_to_token = {v: k for k, v in mapping.items()}
    return vocab


def build_model(src_vocab_size: int, tgt_vocab_size: int, src_pad_idx: int, tgt_pad_idx: int) -> Seq2Seq:
    encoder = LSTMEncoder(src_vocab_size, 256, 512, 2, src_pad_idx, bidirectional=True, dropout=0.2)
    decoder = LSTMDecoder(tgt_vocab_size, 256, 1024, 2, tgt_pad_idx, dropout=0.2)
    return Seq2Seq(encoder, decoder, src_pad_idx)


@torch.no_grad()
def translate_one(model: Seq2Seq, src_ids: torch.Tensor, tgt_vocab: Vocabulary, device: torch.device):
    encoder_outputs, (hidden, cell) = model.encoder(src_ids.unsqueeze(0).to(device))
    hidden, cell = bridge_bidirectional_states(hidden, cell)
    token = torch.tensor([tgt_vocab.sos_idx], device=device)
    out = []
    for _ in range(MAX_LEN):
        logits, hidden, cell = model.decoder(token, hidden, cell)
        token = logits.argmax(dim=1)
        idx = int(token.item())
        if idx == tgt_vocab.eos_idx:
            break
        if idx not in (tgt_vocab.pad_idx, tgt_vocab.sos_idx):
            out.append(idx)
    return tgt_vocab.decode(out)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(MODEL_PATH, map_location=device)

    src_vocab = rebuild_vocab(ckpt["src_vocab"])
    tgt_vocab = rebuild_vocab(ckpt["tgt_vocab"])
    history = ckpt["history"]
    test_pairs: List[Tuple[str, str]] = ckpt["test_pairs"]
    test_ds = ParallelTextDataset(test_pairs, src_vocab, tgt_vocab, MAX_LEN, MAX_LEN)

    model = build_model(len(src_vocab), len(tgt_vocab), src_vocab.pad_idx, tgt_vocab.pad_idx).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 1 Loss Curves")
    plt.tight_layout()
    curve_path = OUTPUT_DIR / "phase2_loss_curve.png"
    plt.savefig(curve_path)

    sentence_scores, records = [], []
    start = time.perf_counter()
    for i in range(len(test_ds)):
        src_ids, tgt_ids = test_ds[i]
        pred = translate_one(model, src_ids, tgt_vocab, device)
        ref = tgt_vocab.decode(tgt_ids.tolist())
        score = sentence_bleu_score(ref, pred)
        sentence_scores.append(score)
        records.append(
            {
                "src": " ".join(src_vocab.decode(src_ids.tolist())),
                "ref": " ".join(ref),
                "pred": " ".join(pred),
                "bleu": score,
            }
        )
    total_ms = (time.perf_counter() - start) * 1000.0
    avg_ms = total_ms / max(1, len(test_ds))

    stats = bleu_stats(sentence_scores)
    sorted_records = sorted(records, key=lambda x: x["bleu"])
    worst_3 = sorted_records[:3]
    best_3 = sorted_records[-3:][::-1]
    worst_5 = sorted_records[:5]

    print(f"[Phase {PHASE}] BLEU Min: {stats['min']:.4f}")
    print(f"[Phase {PHASE}] BLEU Max: {stats['max']:.4f}")
    print(f"[Phase {PHASE}] BLEU Mean: {stats['mean']:.4f}")
    print(f"[Phase {PHASE}] BLEU Std: {stats['std']:.4f}")
    print(f"[Phase {PHASE}] Inference time/sentence (ms): {avg_ms:.4f}")

    print("\n[Phase 2] 3 Best Translations:")
    for row in best_3:
        print(f"BLEU={row['bleu']:.4f} | SRC={row['src']} | REF={row['ref']} | PRED={row['pred']}")

    print("\n[Phase 2] 3 Worst Translations:")
    for row in worst_3:
        print(f"BLEU={row['bleu']:.4f} | SRC={row['src']} | REF={row['ref']} | PRED={row['pred']}")

    print("\n[Phase 2] Confusion Analysis (5 Worst):")
    for row in worst_5:
        print("------------------------------------------------------------")
        print(f"SRC : {row['src']}")
        print(f"EXP : {row['ref']}")
        print(f"PRED: {row['pred']}")
        print(f"BLEU: {row['bleu']:.4f}")
    print(f"[Phase 2] Saved loss curve: {curve_path}")
    print("=== PHASE 2 COMPLETE ===")


if __name__ == "__main__":
    main()
