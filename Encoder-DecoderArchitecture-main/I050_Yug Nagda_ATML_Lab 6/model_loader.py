from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Tuple

from tensorflow.keras.models import load_model
from transformers import MarianMTModel, MarianTokenizer

from project_paths import resolve_asset


logger = logging.getLogger(__name__)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_all_models(config: dict, *, allow_missing_spanish: bool = True) -> Tuple[Any, ...]:
    """
    Loads all models + tokenizers.

    Returns:
      hindi_model, hindi_tokenizer_data, summarizer_model, summarizer_tokenizer_data,
      spanish_tokenizer, spanish_model
    """

    files = config["files"]
    folders = config["folders"]

    # Hindi (local)
    hindi_model_path = resolve_asset(files["hindi_model"], folders["models"])
    hindi_tok_path = resolve_asset(files["hindi_tokenizer"], folders["models"])

    logger.info("Loading Hindi model from %s", hindi_model_path)
    hindi_model = load_model(str(hindi_model_path))
    tok = _load_pickle(hindi_tok_path)

    # Summarizer (local)
    sum_model_path = resolve_asset(files["summarizer_model"], folders["models"])
    sum_tok_path = resolve_asset(files["summarizer_tokenizer"], folders["models"])

    logger.info("Loading summarizer model from %s", sum_model_path)
    summarizer_model = load_model(str(sum_model_path))
    sum_tok = _load_pickle(sum_tok_path)

    # Spanish (may download)
    spanish_pretrained = config.get("spanish_pretrained", "Helsinki-NLP/opus-mt-en-es")
    spanish_tokenizer = None
    spanish_model = None
    try:
        logger.info("Loading Spanish model from %s", spanish_pretrained)
        spanish_tokenizer = MarianTokenizer.from_pretrained(spanish_pretrained)
        spanish_model = MarianMTModel.from_pretrained(spanish_pretrained)
    except Exception as e:
        if allow_missing_spanish:
            logger.warning("Spanish model unavailable (%s). Skipping Spanish.", e)
        else:
            raise RuntimeError(f"Failed to load Spanish model '{spanish_pretrained}': {e}") from e

    return hindi_model, tok, summarizer_model, sum_tok, spanish_tokenizer, spanish_model

