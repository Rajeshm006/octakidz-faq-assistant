import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import FAQDataset


DATASET_PATH = Path(__file__).parent / "data" / "faq_dataset.json"


def load_faq_dataset(path: Path = DATASET_PATH) -> FAQDataset:
    """Load and strictly validate the complete local JSON knowledge source."""
    if not path.is_file():
        raise RuntimeError(f"FAQ dataset is missing at {path}.")

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("FAQ dataset is unreadable or contains invalid JSON.") from exc

    if isinstance(raw, list):
        candidate = {"questions": raw}
    elif isinstance(raw, dict) and isinstance(raw.get("questions"), list):
        candidate = raw
    else:
        raise RuntimeError(
            "FAQ dataset must be a list of entries or an object with a questions list."
        )

    try:
        return FAQDataset.model_validate(candidate)
    except ValidationError as exc:
        raise RuntimeError(
            "FAQ dataset validation failed: entries require id, category, question, and answer."
        ) from exc

