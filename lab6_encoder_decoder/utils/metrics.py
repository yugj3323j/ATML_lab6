"""Metric helpers for BLEU and ROUGE reporting in Lab 6."""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def sentence_bleu_score(reference_tokens: Sequence[str], prediction_tokens: Sequence[str]) -> float:
    if not prediction_tokens:
        return 0.0
    smoothie = SmoothingFunction().method1
    return float(sentence_bleu([list(reference_tokens)], list(prediction_tokens), smoothing_function=smoothie))


def corpus_bleu_from_lists(references: List[List[str]], predictions: List[List[str]]) -> float:
    if not references:
        return 0.0
    scores = [sentence_bleu_score(r, p) for r, p in zip(references, predictions)]
    return float(np.mean(scores))


def bleu_stats(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": float(statistics.mean(scores)),
        "std": float(statistics.pstdev(scores)),
    }


def rouge_scores(references: List[str], predictions: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        r1.append(score["rouge1"].fmeasure)
        r2.append(score["rouge2"].fmeasure)
        rl.append(score["rougeL"].fmeasure)
    if not r1:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    return {"rouge1": float(np.mean(r1)), "rouge2": float(np.mean(r2)), "rougeL": float(np.mean(rl))}
