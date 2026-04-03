from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import LLMSettings
from .utils import ensure_directory, parse_optional_bool

DEFAULT_QWEN_BASE_URL = "https://www.autodl.art/api/v1/chat/completions"
DEFAULT_QWEN_MODEL = "Qwen3.5-397B-A17B"


class ConfigManager:
    """Read and write local JSON configuration files."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_dir = ensure_directory(self.project_root / "config")

    def load(self, name: str) -> dict[str, Any]:
        path = self.config_dir / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.config_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def field_aliases(self) -> dict[str, Any]:
        return self.load("field_aliases.json")

    def business_rules(self) -> dict[str, Any]:
        return self.load("business_rules.json")

    def permissions(self) -> dict[str, Any]:
        return self.load("permissions.json")

    def skill_root(self) -> Path:
        return ensure_directory(self.project_root / "skills")

    def llm_settings(self, overrides: dict[str, Any] | None = None) -> LLMSettings:
        qwen_api_key = (
            os.getenv("QWEN_API_KEY")
            or os.getenv("AUTODL_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("API_KEY")
            or None
        )
        qwen_model = os.getenv("QWEN_MODEL") or (DEFAULT_QWEN_MODEL if qwen_api_key else None)
        qwen_base_url = os.getenv("QWEN_BASE_URL") or (DEFAULT_QWEN_BASE_URL if qwen_api_key or qwen_model else None)
        settings = LLMSettings(
            base_url=(
                os.getenv("LLM_BASE_URL")
                or qwen_base_url
                or os.getenv("OPENAI_BASE_URL")
                or None
            ),
            api_key=(
                os.getenv("LLM_API_KEY")
                or qwen_api_key
                or os.getenv("OPENAI_API_KEY")
                or None
            ),
            model=os.getenv("LLM_MODEL") or qwen_model or None,
            temperature=float(os.getenv("LLM_TEMPERATURE") or os.getenv("QWEN_TEMPERATURE") or "0.1"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1800")),
            enable_thinking=parse_optional_bool(os.getenv("LLM_ENABLE_THINKING")),
        )
        if overrides:
            payload = settings.model_dump(mode="json")
            normalized_overrides = {key: value for key, value in overrides.items() if value not in (None, "")}
            if "enable_thinking" in overrides:
                normalized_overrides["enable_thinking"] = parse_optional_bool(overrides.get("enable_thinking"))
            payload.update(normalized_overrides)
            settings = LLMSettings(**payload)
        return settings
