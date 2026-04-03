from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    value = re.sub(r"[\s\r\n\t_./\\-]+", "", value)
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "", value)
    return value


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "output"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dataframe_preview(df: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    preview = df.head(limit).copy()
    preview = preview.where(pd.notna(preview), "")
    return preview.to_dict(orient="records")


def detect_header_candidates(raw_df: pd.DataFrame, limit: int = 5) -> list[int]:
    scores: list[tuple[int, int]] = []
    for index in raw_df.index[:10]:
        row = raw_df.loc[index]
        non_empty = int(row.notna().sum())
        unique_text = len({str(item).strip() for item in row.dropna().tolist() if str(item).strip()})
        score = non_empty + unique_text
        scores.append((int(index), score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return [row_index for row_index, _ in scores[:limit]]


def first_non_null(series: pd.Series) -> Any:
    valid = series.dropna()
    if valid.empty:
        return pd.NA
    return valid.iloc[0]


def jsonify(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return jsonify(value.model_dump())
    if hasattr(value, "dict"):
        return jsonify(value.dict())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return dataframe_preview(value, limit=20)
    if isinstance(value, dict):
        return {key: jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonify(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型未返回可解析的 JSON 对象。")
    return json.loads(text[start : end + 1])


def merge_unique_texts(*groups: Iterable[str] | str | None) -> list[str]:
    merged: list[str] = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, str):
            values = [group]
        else:
            values = group
        for item in values:
            value = str(item).strip()
            if value and value not in merged:
                merged.append(value)
    return merged


def parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"无法解析布尔值：{value}")
