from __future__ import annotations

import json
from typing import Any

ROLE_ORDER = [
    "data_analyst",
    "tactical_analyst",
    "news_lineup_analyst",
    "risk_officer",
]

ROLE_LABELS = {
    "data_analyst": "数据分析师",
    "tactical_analyst": "战术分析师",
    "news_lineup_analyst": "新闻与阵容分析师",
    "risk_officer": "风险官",
    "summarizer": "汇总官",
}

ROLE_BRIEFS = {
    "data_analyst": "关注近期战绩、进球/失球、排名、历史交锋、赔率或公开预测。",
    "tactical_analyst": "关注阵型、风格、攻防匹配、控球、反击、定位球等因素。",
    "news_lineup_analyst": "关注伤停、停赛、轮换、教练采访、赛前新闻。",
    "risk_officer": "专门唱反调，找预测中最容易翻车的点。",
}

ROLE_JSON_SCHEMA = {
    "role": "string",
    "match_id": "string",
    "predicted_outcome": "HOME_WIN | DRAW | AWAY_WIN",
    "conclusion": "string",
    "confidence": "low | medium | high",
    "key_evidence": ["string"],
    "risks": ["string"],
    "citations": [{"source_id": "string", "note": "string"}],
    "one_line": "string",
}

SUMMARY_JSON_SCHEMA = {
    "match_id": "string",
    "predicted_outcome": "HOME_WIN | DRAW | AWAY_WIN",
    "confidence": "low | medium | high",
    "key_evidence": ["string"],
    "max_risk": "string",
    "role_summaries": {"role": "one-line view"},
    "agreements": ["string"],
    "conflicts": ["string"],
    "decision": "string",
    "citations": [{"source_id": "string", "note": "string"}],
}

DAILY_BATCH_JSON_SCHEMA = {
    "role_analyses": [ROLE_JSON_SCHEMA],
    "predictions": [SUMMARY_JSON_SCHEMA],
}


def build_role_prompt(
    role: str,
    match_payload: dict[str, Any],
    sources_payload: list[dict[str, Any]],
    lessons: str,
) -> str:
    prompt = {
        "task": "Analyze one World Cup match as an independent role. Do not infer a schedule outside the match record.",
        "role": ROLE_LABELS[role],
        "role_brief": ROLE_BRIEFS[role],
        "constraints": [
            "Use only the provided match record, source list, and historical lessons.",
            "Write all natural-language fields in Chinese.",
            "Return JSON only. No markdown.",
            "Cite source_id values from the provided sources or match.sources.",
            "This is for research and entertainment only; do not provide betting advice.",
        ],
        "match_record": match_payload,
        "sources": sources_payload,
        "historical_lessons": lessons or "No historical lessons yet.",
        "output_schema": ROLE_JSON_SCHEMA,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_summary_prompt(
    match_payload: dict[str, Any],
    role_payloads: list[dict[str, Any]],
    sources_payload: list[dict[str, Any]],
    lessons: str,
) -> str:
    prompt = {
        "task": "Act as the summarizer after four independent role analyses are complete.",
        "constraints": [
            "Compare where roles agree and conflict.",
            "Choose a final outcome from HOME_WIN, DRAW, AWAY_WIN.",
            "Explain how conflicts were resolved.",
            "Write all natural-language fields in Chinese.",
            "Return JSON only. No markdown.",
            "This is for research and entertainment only; do not provide betting advice.",
        ],
        "match_record": match_payload,
        "role_analyses": role_payloads,
        "sources": sources_payload,
        "historical_lessons": lessons or "No historical lessons yet.",
        "output_schema": SUMMARY_JSON_SCHEMA,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_daily_batch_prompt(
    matches_payload: list[dict[str, Any]],
    sources_payload: list[dict[str, Any]],
    lessons: str,
) -> str:
    prompt = {
        "task": (
            "For each World Cup match, produce four independent role analyses "
            "and then one summarizer prediction. This is a batch version of the harness."
        ),
        "roles": [
            {"role": role, "label": ROLE_LABELS[role], "brief": ROLE_BRIEFS[role]}
            for role in ROLE_ORDER
        ],
        "constraints": [
            "Use only the provided match records, source list, and historical lessons.",
            "For each match, include exactly one analysis for each role in role_analyses.",
            "For each match, include exactly one final prediction in predictions.",
            "The first four roles should be written as independent views; summarizer compares their conflicts.",
            "Opta and public probabilities may be used as one evidence source, but final prediction must explain the broader synthesis.",
            "Write all natural-language fields in Chinese.",
            "Return JSON only. No markdown. No reasoning text.",
            "Do not provide betting advice.",
        ],
        "matches": matches_payload,
        "sources": sources_payload,
        "historical_lessons": lessons or "No historical lessons yet.",
        "output_schema": DAILY_BATCH_JSON_SCHEMA,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)
