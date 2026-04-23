from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Project root is the directory containing this file.
    Using this keeps paths correct even if the working directory changes.
    """

    return Path(__file__).resolve().parent


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Loads config.json if present; otherwise returns defaults.
    """

    root = get_project_root()
    cfg_path = config_path or (root / "config.json")
    if not cfg_path.exists():
        logger.warning("Config file not found at %s; using defaults.", cfg_path)
        return {
            "folders": {"models": "models", "input": "input", "output": "outputs"},
            "files": {
                "hindi_model": "eng_hin_translation_model.h5",
                "hindi_tokenizer": "tokenizer_data.pkl",
                "summarizer_model": "custom_summarizer_model.keras",
                "summarizer_tokenizer": "summarizer_tokenizer_data.pkl",
            },
            "spanish_pretrained": "Helsinki-NLP/opus-mt-en-es",
        }

    return _read_json(cfg_path)


def resolve_asset(filename: str, preferred_subdir: str) -> Path:
    """
    Resolve a file within the project directory structure.

    Priority:
    1) <project_root>/<preferred_subdir>/<filename>
    2) <project_root>/<filename> (for legacy layouts)
    3) Search anywhere under the project root for <filename>
    """

    root = get_project_root()
    candidates = [
        root / preferred_subdir / filename,
        root / filename,
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: search anywhere under project root
    matches = list(root.rglob(filename))
    if matches:
        logger.warning(
            "Using asset found via search for %r at %s. "
            "Consider placing it under %s/.",
            filename,
            matches[0],
            preferred_subdir,
        )
        return matches[0]

    raise FileNotFoundError(
        f"Required file {filename!r} not found under {root}. "
        f"Tried: {[str(x) for x in candidates] + ['<anywhere under project root>']}"
    )


def resolve_output_dir(folder_name: str) -> Path:
    root = get_project_root()
    out_dir = root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

