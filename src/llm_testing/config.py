"""Load settings.yaml, resolve paths against the project root, and fetch the API key."""

from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(relative: str | Path) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_text(relative: str | Path) -> str:
    return project_path(relative).read_text(encoding="utf-8").strip()


def get_api_key() -> str:
    """OPENAI_API_KEY from the environment or .env; prompt as a last resort."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    return os.getenv("OPENAI_API_KEY") or getpass.getpass("Enter OpenAI API Key: ")
