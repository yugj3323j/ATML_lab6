from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from inference import summarize_text, translate_to_hindi, translate_to_spanish
from model_loader import load_all_models
from project_paths import get_project_root, load_config, resolve_output_dir


logger = logging.getLogger(__name__)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_input_text(config: dict) -> tuple[str, Optional[Path]]:
    """
    Loads input text from predefined `input/` folder.

    If no file exists, falls back to a small embedded sample so the pipeline
    still runs end-to-end (while logging a warning).
    """

    root = get_project_root()
    input_dir = root / config["folders"]["input"]

    preferred = ["input_text.txt", "input.txt", "data.txt"]
    for name in preferred:
        p = input_dir / name
        if p.exists() and p.is_file():
            return _read_text_file(p), p

    if input_dir.exists():
        txt_files = sorted(input_dir.glob("*.txt"))
        if txt_files:
            # Deterministic: pick the first file alphabetically.
            p = txt_files[0]
            logger.warning("No preferred input filename found; using %s", p)
            return _read_text_file(p), p

    # Graceful fallback: lets `python main.py` work immediately.
    sample = "How are you? I hope you are doing well today."
    logger.warning("No input text found under %s. Using embedded sample.", input_dir)
    return sample, None


def run_pipeline(*, input_text: Optional[str] = None, input_path: Optional[Path] = None) -> dict[str, Any]:
    config = load_config()
    out_dir = resolve_output_dir(config["folders"]["output"])

    if input_path is not None:
        input_text = _read_text_file(input_path)

    if input_text is None:
        input_text, resolved_path = load_input_text(config)
    else:
        resolved_path = input_path

    text = (input_text or "").strip()
    if not text:
        raise ValueError("Input text is empty after preprocessing.")

    logger.info("Loading models...")
    hindi_model, hindi_tok, summarizer_model, summarizer_tok, spanish_tokenizer, spanish_model = load_all_models(
        config, allow_missing_spanish=True
    )

    # ─── Preprocessing (minimal wrapper) ─────────────────────────────────────
    logger.info("Preprocessing input text (%d chars).", len(text))

    # ─── Model execution ────────────────────────────────────────────────────
    logger.info("Running Hindi translation...")
    hindi_result = translate_to_hindi(text, hindi_model, hindi_tok)

    spanish_result: Optional[str] = None
    if spanish_model is not None and spanish_tokenizer is not None:
        try:
            logger.info("Running Spanish translation...")
            spanish_result = translate_to_spanish(text, spanish_tokenizer, spanish_model)
        except Exception as e:
            logger.warning("Spanish translation failed (%s). Continuing.", e)

    logger.info("Running summarization...")
    summary_result = summarize_text(text, summarizer_model, summarizer_tok)

    # ─── Output generation ─────────────────────────────────────────────────
    timestamp = datetime.now().isoformat(timespec="seconds")
    outputs: dict[str, Any] = {
        "timestamp": timestamp,
        "input": text,
        "input_path": str(resolved_path) if resolved_path else None,
        "outputs": {
            "hindi": hindi_result,
            "spanish": spanish_result,
            "summary": summary_result,
        },
    }

    (out_dir / "translation_hindi.txt").write_text(hindi_result, encoding="utf-8")
    if spanish_result is not None:
        (out_dir / "translation_spanish.txt").write_text(spanish_result, encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary_result, encoding="utf-8")
    (out_dir / "run_output.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Pipeline completed. Outputs written to %s", out_dir)
    return outputs

