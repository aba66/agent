from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import write_text
from .models import FinalPrediction, MatchRecord, RoleAnalysis
from .prompts import ROLE_LABELS, ROLE_ORDER
from .providers import BaseLLMProvider


@dataclass
class MatchAnalysisResult:
    match: MatchRecord
    role_analyses: list[RoleAnalysis]
    final_prediction: FinalPrediction


class AgentOrchestrator:
    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def analyze_match(
        self,
        match: MatchRecord,
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> MatchAnalysisResult:
        prediction_match = build_prediction_input_match(match)
        prediction_sources = build_prediction_input_sources(sources)
        roles = []
        for role in ROLE_ORDER:
            result = self.provider.generate_role_analysis(role, prediction_match, prediction_sources, lessons)
            roles.append(
                RoleAnalysis.from_payload(
                    role=role,
                    match=match,
                    payload=result.payload,
                    raw_text=result.raw_text,
                    parse_error=result.parse_error,
                )
            )

        summary_result = self.provider.generate_summary(
            match=prediction_match,
            role_payloads=[analysis.to_dict() for analysis in roles],
            sources=prediction_sources,
            lessons=lessons,
        )
        final = FinalPrediction.from_payload(
            match=match,
            payload=summary_result.payload,
            raw_text=summary_result.raw_text,
            parse_error=summary_result.parse_error,
        )
        return MatchAnalysisResult(match=match, role_analyses=roles, final_prediction=final)


def write_agent_markdown(base_dir: Path, result: MatchAnalysisResult) -> None:
    match_dir = base_dir / safe_name(result.match.match_id)
    for analysis in result.role_analyses:
        write_text(match_dir / f"{analysis.role}.md", render_role_markdown(result.match, analysis))
    write_text(match_dir / "summarizer.md", render_summary_markdown(result.match, result.final_prediction))


def build_prediction_input_match(match: MatchRecord) -> MatchRecord:
    """Return the match record used by worldcup_prediction.

    The prediction harness is allowed to use Opta and other public probabilities as
    evidence, but the final prediction is produced by the role workflow rather than
    copied from any single benchmark.
    """
    sanitized = deepcopy(match)
    sanitized.facts = dict(sanitized.facts)
    sanitized.facts["benchmark_note"] = "Opta 等外部概率可作为输入证据之一；最终结果由 worldcup_prediction 多角色综合给出。"
    return sanitized


def build_prediction_input_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sources


def render_role_markdown(match: MatchRecord, analysis: RoleAnalysis) -> str:
    citations = "\n".join(f"- `{item.source_id}`: {item.note}" for item in analysis.citations) or "- None"
    evidence = "\n".join(f"- {item}" for item in analysis.key_evidence) or "- None"
    risks = "\n".join(f"- {item}" for item in analysis.risks) or "- None"
    parse_note = f"\n\nParse warning: {analysis.parse_error}\n" if analysis.parse_error else ""
    return f"""# {ROLE_LABELS.get(analysis.role, analysis.role)} - {match.home_team} vs {match.away_team}

比赛：`{match.match_id}`

结论：{analysis.conclusion}

预测结果：{_outcome_label_zh(analysis.predicted_outcome, match.home_team, match.away_team)}

信心等级：{_confidence_zh(analysis.confidence)}

## 关键依据
{evidence}

## 风险点
{risks}

## 引用来源
{citations}
{parse_note}
"""


def render_summary_markdown(match: MatchRecord, prediction: FinalPrediction) -> str:
    role_lines = "\n".join(f"- `{role}`: {summary}" for role, summary in prediction.role_summaries.items()) or "- None"
    evidence = "\n".join(f"- {item}" for item in prediction.key_evidence) or "- None"
    agreements = "\n".join(f"- {item}" for item in prediction.agreements) or "- None"
    conflicts = "\n".join(f"- {item}" for item in prediction.conflicts) or "- None"
    citations = "\n".join(f"- `{item.source_id}`: {item.note}" for item in prediction.citations) or "- None"
    parse_note = f"\n\nParse warning: {prediction.parse_error}\n" if prediction.parse_error else ""
    return f"""# 汇总官 - {match.home_team} vs {match.away_team}

比赛：`{match.match_id}`

最终预测：{_outcome_label_zh(prediction.predicted_outcome, match.home_team, match.away_team)}

信心等级：{_confidence_zh(prediction.confidence)}

汇总取舍：{prediction.decision}

## 各角色观点
{role_lines}

## 一致意见
{agreements}

## 冲突意见
{conflicts}

## 关键依据
{evidence}

## 最大风险
{prediction.max_risk}

## 引用来源
{citations}
{parse_note}
"""


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _confidence_zh(confidence: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(confidence, confidence)


def _outcome_label_zh(outcome: str, home_team: str, away_team: str) -> str:
    if outcome == "HOME_WIN":
        return f"{home_team} 胜"
    if outcome == "AWAY_WIN":
        return f"{away_team} 胜"
    return "平局"
