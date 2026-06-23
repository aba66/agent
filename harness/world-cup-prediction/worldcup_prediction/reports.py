from __future__ import annotations

from .models import FinalPrediction, MatchRecord, SourceRecord


def render_daily_report(
    date: str,
    matches: list[MatchRecord],
    predictions: list[FinalPrediction],
    sources: list[SourceRecord],
    lessons_used: list[str],
) -> str:
    prediction_by_match = {prediction.match_id: prediction for prediction in predictions}
    lines = [
        f"# 世界杯预测日报 - {date}",
        "",
        "仅用于研究和娱乐预测，不构成投注建议。",
        "",
        "## 已加载历史教训",
    ]
    if lessons_used:
        lines.extend(f"- {lesson}" for lesson in lessons_used)
    else:
        lines.append("- 暂无历史教训。")

    lines.extend(["", "## 今日比赛（北京时间）"])
    for match in matches:
        lines.append(
            f"- `{match.match_id}` {match.kickoff_local}: {match.home_team} vs {match.away_team}，"
            f"{match.venue}，{match.stage}，状态={match.status}"
        )

    lines.extend(["", "## 预测结果"])
    for match in matches:
        prediction = prediction_by_match[match.match_id]
        lines.extend(
            [
                "",
                f"### {match.home_team} vs {match.away_team}",
                "",
                f"- worldcup_prediction 综合预测：{_outcome_label_zh(prediction.predicted_outcome, match.home_team, match.away_team)}",
                f"- Opta 单源预测：{_opta_summary(match)}",
                f"- 对比结论：{_comparison_summary(match, prediction)}",
                f"- 信心等级：{_confidence_zh(prediction.confidence)}",
                f"- 最大风险：{prediction.max_risk}",
                "- 关键依据：",
            ]
        )
        lines.extend(f"  - {item}" for item in prediction.key_evidence)
        lines.append("- 各角色一句话观点：")
        for role, summary in prediction.role_summaries.items():
            lines.append(f"  - `{role}`: {summary}")
        lines.append("- 汇总取舍：")
        lines.append(f"  - {prediction.decision}")

    lines.extend(["", "## 数据来源"])
    for source in sources:
        url = source.url or "n/a"
        lines.append(f"- `{source.source_id}` {source.name} ({source.kind}), fetched_at={source.fetched_at}, url={url}")
    lines.append("")
    return "\n".join(lines)


def _confidence_zh(confidence: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(confidence, confidence)


def _outcome_label_zh(outcome: str, home_team: str, away_team: str) -> str:
    if outcome == "HOME_WIN":
        return f"{home_team} 胜"
    if outcome == "AWAY_WIN":
        return f"{away_team} 胜"
    return "平局"


def _opta_summary(match: MatchRecord) -> str:
    prediction = match.facts.get("opta_prediction") or {}
    if not prediction:
        public = match.facts.get("public_predictions") or []
        prediction = public[0] if public else {}
    if not prediction:
        return "未提供"
    home = prediction.get("home_win")
    draw = prediction.get("draw")
    away = prediction.get("away_win")
    favorite = prediction.get("favorite")
    parts = []
    if favorite:
        parts.append(f"倾向 {favorite}")
    if home is not None and draw is not None and away is not None:
        parts.append(
            f"{match.home_team}胜 {float(home) * 100:.1f}%，平 {float(draw) * 100:.1f}%，{match.away_team}胜 {float(away) * 100:.1f}%"
        )
    return "；".join(parts) if parts else "未提供"


def _comparison_summary(match: MatchRecord, prediction: FinalPrediction) -> str:
    opta = match.facts.get("opta_prediction") or {}
    if not opta:
        return "未提供 Opta 基准，无法比较。"
    opta_pick = _opta_pick(opta)
    harness_pick = prediction.predicted_outcome
    if harness_pick == opta_pick:
        return f"一致，均为{_outcome_label_zh(harness_pick, match.home_team, match.away_team)}。"
    return (
        f"不一致：Harness 为{_outcome_label_zh(harness_pick, match.home_team, match.away_team)}，"
        f"Opta 为{_outcome_label_zh(opta_pick, match.home_team, match.away_team)}。"
    )


def _opta_pick(opta: dict) -> str:
    values = {
        "HOME_WIN": float(opta.get("home_win", 0)),
        "DRAW": float(opta.get("draw", 0)),
        "AWAY_WIN": float(opta.get("away_win", 0)),
    }
    return max(values, key=values.get)
