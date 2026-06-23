from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import MatchRecord, normalize_outcome
from .prompts import ROLE_LABELS, build_role_prompt, build_summary_prompt

DEFAULT_USER_AGENT = "worldcup-prediction/0.1"
DEFAULT_LOCAL_CONFIG_DIR = "~/.config/worldcup_prediction"
DEFAULT_LOCAL_CONFIG_PATH = f"{DEFAULT_LOCAL_CONFIG_DIR}/ddss.env"
DEFAULT_QWEN_CONFIG_PATH = f"{DEFAULT_LOCAL_CONFIG_DIR}/qwen.env"


@dataclass
class ProviderResult:
    payload: dict[str, Any]
    raw_text: str
    parse_error: str | None = None


class ProviderError(RuntimeError):
    pass


class BaseLLMProvider:
    name = "base"
    supports_daily_batch = False

    def generate_role_analysis(
        self,
        role: str,
        match: MatchRecord,
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        raise NotImplementedError

    def generate_summary(
        self,
        match: MatchRecord,
        role_payloads: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        raise NotImplementedError

    def generate_daily_batch(
        self,
        matches: list[MatchRecord],
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        raise NotImplementedError


class DeterministicProvider(BaseLLMProvider):
    name = "mock"

    def generate_role_analysis(
        self,
        role: str,
        match: MatchRecord,
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        payload = self._role_payload(role, match, lessons)
        return ProviderResult(payload=payload, raw_text=json.dumps(payload, ensure_ascii=False))

    def generate_summary(
        self,
        match: MatchRecord,
        role_payloads: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        votes = [normalize_outcome(item.get("predicted_outcome"), match.home_team, match.away_team) for item in role_payloads]
        vote_counts = Counter(votes)
        top_vote, top_count = vote_counts.most_common(1)[0]
        favorite = self._baseline_outcome(match)
        if top_count < 2:
            final = favorite
        else:
            final = top_vote
        if "逼平" in lessons or "draw" in lessons.lower():
            risk_vote = next((item for item in role_payloads if item.get("role") == "risk_officer"), None)
            if risk_vote and risk_vote.get("predicted_outcome") == "DRAW" and top_count <= 2:
                final = "DRAW"
        confidence = "high" if top_count >= 3 and final != "DRAW" else "medium"
        if top_count <= 1:
            confidence = "low"
        role_summaries = {
            str(item.get("role")): str(item.get("one_line") or item.get("conclusion", "")) for item in role_payloads
        }
        agreements = [f"{count} 个角色支持 {self._outcome_text(outcome, match)}" for outcome, count in vote_counts.items() if count > 1]
        conflicts = [f"{item.get('role')} 倾向 {self._outcome_text(str(item.get('predicted_outcome')), match)}" for item in role_payloads]
        key_evidence = []
        for item in role_payloads[:3]:
            key_evidence.extend([str(evidence) for evidence in item.get("key_evidence", [])[:1]])
        risks = []
        for item in role_payloads:
            risks.extend([str(risk) for risk in item.get("risks", [])[:1]])
        payload = {
            "match_id": match.match_id,
            "predicted_outcome": final,
            "confidence": confidence,
            "key_evidence": key_evidence[:4],
            "max_risk": risks[0] if risks else "临场首发或战术变化可能改变比赛走势。",
            "role_summaries": role_summaries,
            "agreements": agreements or ["没有形成清晰多数，汇总官结合基础实力和风险上下文做判断。"],
            "conflicts": conflicts,
            "decision": (
                f"最终选择{self._outcome_text(final, match)}。取舍依据是四个 harness 角色的独立判断、"
                "Opta/公开概率、球队状态、战术匹配和阵容风险；日报中会把该综合预测与 Opta 单源预测单独对比。"
                "本报告仅用于研究和娱乐预测，不构成投注建议。"
            ),
            "citations": self._citations(match),
        }
        return ProviderResult(payload=payload, raw_text=json.dumps(payload, ensure_ascii=False))

    def _role_payload(self, role: str, match: MatchRecord, lessons: str) -> dict[str, Any]:
        baseline = self._baseline_outcome(match)
        home = self._team_profile(match, match.home_team)
        away = self._team_profile(match, match.away_team)
        lesson_note = self._lesson_note(lessons)
        citations = self._citations(match)

        if role == "data_analyst":
            outcome = baseline
            evidence = [
                self._opta_sentence(match),
                self._form_sentence(match, home, away),
                self._goals_sentence(match, home, away),
                self._public_prediction_sentence(match),
            ]
            risks = ["赛前概率和状态快照可能滞后于临场阵容、天气或裁判尺度。"]
        elif role == "tactical_analyst":
            edge = str(match.facts.get("tactical_edge") or "").lower()
            if edge == "home":
                outcome = "HOME_WIN"
            elif edge == "away":
                outcome = "AWAY_WIN"
            elif edge == "draw":
                outcome = "DRAW"
            else:
                outcome = baseline
            evidence = [
                str(match.facts.get("tactical_notes") or "未提供战术备注，按双方风格匹配做保守判断。"),
                f"{match.home_team} 风格：{home.get('style', '未知')}；{match.away_team} 风格：{away.get('style', '未知')}。",
            ]
            risks = ["定位球、早早红牌或开局失球会快速改变赛前战术假设。"]
        elif role == "news_lineup_analyst":
            impact = self._availability_edge(match, home, away)
            outcome = impact or baseline
            evidence = [
                f"{match.home_team} 阵容新闻：{home.get('news', '暂无重大更新')}",
                f"{match.away_team} 阵容新闻：{away.get('news', '暂无重大更新')}",
            ]
            risks = ["首发名单可能与赛前预测不同，仍需警惕临场轮换。"]
        elif role == "risk_officer":
            outcome = self._contrarian_outcome(baseline, lessons)
            evidence = [
                "风险官刻意挑战热门方向，优先寻找平局或爆冷路径。",
                lesson_note or "暂无历史错题，默认提高平局和爆冷敏感度。",
            ]
            risks = [
                "热门球队在小组赛中可能开局保守。",
                "低位防守和快速反击会把比赛压缩成平局或小比分爆冷。"
            ]
        else:
            outcome = baseline
            evidence = ["角色兜底逻辑使用基础比赛事实。"]
            risks = ["未识别角色。"]

        confidence = self._confidence_for(role, match, outcome, baseline)
        conclusion = (
            f"{ROLE_LABELS.get(role, role)}倾向{self._outcome_text(outcome, match)}。"
            f"{lesson_note}" if lesson_note and role != "risk_officer" else
            f"{ROLE_LABELS.get(role, role)}倾向{self._outcome_text(outcome, match)}。"
        )
        return {
            "role": role,
            "match_id": match.match_id,
            "predicted_outcome": outcome,
            "conclusion": conclusion,
            "confidence": confidence,
            "key_evidence": [item for item in evidence if item],
            "risks": risks,
            "citations": citations,
            "one_line": f"{ROLE_LABELS.get(role, role)}：{self._outcome_text(outcome, match)}，信心{self._confidence_text(confidence)}。",
        }

    def _baseline_outcome(self, match: MatchRecord) -> str:
        home = self._team_profile(match, match.home_team)
        away = self._team_profile(match, match.away_team)
        home_score = self._team_score(home)
        away_score = self._team_score(away)
        public = match.facts.get("public_predictions") or []
        opta = match.facts.get("opta_prediction") or {}
        if opta:
            public = [opta] + public
        if public:
            first = public[0]
            home_score += float(first.get("home_win", 0)) * 12
            away_score += float(first.get("away_win", 0)) * 12
            draw_prob = float(first.get("draw", 0))
            if abs(home_score - away_score) < 2.5 and draw_prob >= 0.25:
                return "DRAW"
        diff = home_score - away_score
        if abs(diff) < 3.5:
            return "DRAW"
        return "HOME_WIN" if diff > 0 else "AWAY_WIN"

    @staticmethod
    def _team_profile(match: MatchRecord, team: str) -> dict[str, Any]:
        return dict((match.facts.get("team_profiles") or {}).get(team) or {})

    @staticmethod
    def _team_score(profile: dict[str, Any]) -> float:
        ranking = float(profile.get("ranking") or 50)
        form = profile.get("recent_form") or []
        form_score = sum({"W": 3, "D": 1, "L": 0}.get(str(item).upper()[:1], 0) for item in form)
        goals_for = float(profile.get("goals_for") or 0)
        goals_against = float(profile.get("goals_against") or 0)
        return (80 - ranking) * 0.5 + form_score * 1.2 + (goals_for - goals_against) * 0.9

    def _rank_sentence(self, match: MatchRecord, home: dict[str, Any], away: dict[str, Any]) -> str:
        return (
            f"Ranking snapshot: {match.home_team} #{home.get('ranking', 'n/a')}, "
            f"{match.away_team} #{away.get('ranking', 'n/a')}."
        )

    @staticmethod
    def _opta_sentence(match: MatchRecord) -> str:
        opta = match.facts.get("opta_prediction") or {}
        if not opta:
            return "未提供 Opta 赛前概率。"
        return (
            f"Opta 赛前概率：{match.home_team}胜 {float(opta.get('home_win', 0)) * 100:.1f}%，"
            f"平 {float(opta.get('draw', 0)) * 100:.1f}%，"
            f"{match.away_team}胜 {float(opta.get('away_win', 0)) * 100:.1f}%。"
        )

    def _form_sentence(self, match: MatchRecord, home: dict[str, Any], away: dict[str, Any]) -> str:
        return (
            f"近期状态：{match.home_team} {''.join(home.get('recent_form', [])) or 'n/a'}，"
            f"{match.away_team} {''.join(away.get('recent_form', [])) or 'n/a'}。"
        )

    def _goals_sentence(self, match: MatchRecord, home: dict[str, Any], away: dict[str, Any]) -> str:
        return (
            f"进失球快照：{match.home_team} {home.get('goals_for', 'n/a')} 进 / {home.get('goals_against', 'n/a')} 失，"
            f"{match.away_team} {away.get('goals_for', 'n/a')} 进 / {away.get('goals_against', 'n/a')} 失。"
        )

    @staticmethod
    def _public_prediction_sentence(match: MatchRecord) -> str:
        public = match.facts.get("public_predictions") or []
        if not public:
            return "未提供公开预测快照。"
        first = public[0]
        return (
            f"公开模型快照：主胜 {float(first.get('home_win', 0)) * 100:.1f}%，"
            f"平 {float(first.get('draw', 0)) * 100:.1f}%，"
            f"客胜 {float(first.get('away_win', 0)) * 100:.1f}%。"
        )

    @staticmethod
    def _availability_edge(match: MatchRecord, home: dict[str, Any], away: dict[str, Any]) -> str | None:
        home_absences = int(home.get("major_absences") or 0)
        away_absences = int(away.get("major_absences") or 0)
        if home_absences - away_absences >= 2:
            return "AWAY_WIN"
        if away_absences - home_absences >= 2:
            return "HOME_WIN"
        return None

    @staticmethod
    def _contrarian_outcome(baseline: str, lessons: str) -> str:
        if "逼平" in lessons or "draw" in lessons.lower() or "平局" in lessons:
            return "DRAW"
        if baseline == "HOME_WIN":
            return "DRAW"
        if baseline == "AWAY_WIN":
            return "DRAW"
        return "HOME_WIN"

    @staticmethod
    def _confidence_for(role: str, match: MatchRecord, outcome: str, baseline: str) -> str:
        if role == "risk_officer":
            return "medium"
        public = match.facts.get("public_predictions") or []
        if public and outcome == baseline:
            first = public[0]
            prob = max(float(first.get("home_win", 0)), float(first.get("draw", 0)), float(first.get("away_win", 0)))
            if prob >= 0.58:
                return "high"
        if outcome != baseline:
            return "low"
        return "medium"

    @staticmethod
    def _citations(match: MatchRecord) -> list[dict[str, str]]:
        source_ids = match.sources or ["sample_fixture"]
        return [{"source_id": source_id, "note": "来自已核对比赛记录。"} for source_id in source_ids]

    @staticmethod
    def _lesson_note(lessons: str) -> str:
        lines = [line.strip("- ").strip() for line in lessons.splitlines() if line.strip().startswith("-")]
        return f"已纳入历史教训：{lines[-1]}" if lines else ""

    @staticmethod
    def _outcome_text(outcome: str, match: MatchRecord) -> str:
        if outcome == "HOME_WIN":
            return f"{match.home_team}胜"
        if outcome == "AWAY_WIN":
            return f"{match.away_team}胜"
        return "平局"

    @staticmethod
    def _confidence_text(confidence: str) -> str:
        return {"high": "高", "medium": "中", "low": "低"}.get(confidence, confidence)


class OpenAICompatibleProvider(BaseLLMProvider):
    name = "openai-compatible"
    supports_daily_batch = True

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        wire_api: str = "chat_completions",
        timeout: int = 180,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.wire_api = wire_api
        self.timeout = timeout
        self.user_agent = user_agent

    def generate_role_analysis(
        self,
        role: str,
        match: MatchRecord,
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        prompt = build_role_prompt(role, match.to_dict(), sources, lessons)
        return self._complete_json(prompt)

    def generate_summary(
        self,
        match: MatchRecord,
        role_payloads: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        prompt = build_summary_prompt(match.to_dict(), role_payloads, sources, lessons)
        return self._complete_json(prompt)

    def generate_daily_batch(
        self,
        matches: list[MatchRecord],
        sources: list[dict[str, Any]],
        lessons: str,
    ) -> ProviderResult:
        from .prompts import build_daily_batch_prompt

        prompt = build_daily_batch_prompt([match.to_dict() for match in matches], sources, lessons)
        return self._complete_json(prompt)

    def _complete_json(self, prompt: str) -> ProviderResult:
        if self.wire_api == "responses":
            return self._complete_json_responses(prompt)
        return self._complete_json_chat(prompt)

    def _request(self, path: str, body: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            url=f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )

    def _complete_json_chat(self, prompt: str) -> ProviderResult:
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "/no_think You are a careful football research assistant. Return valid JSON only. Do not include reasoning.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 12000,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = self._request("/chat/completions", body)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", "ignore")
            raise ProviderError(f"HTTP {exc.code}: {error_text or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(str(exc)) from exc

        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed, parse_error = parse_json_object(text)
        return ProviderResult(payload=parsed, raw_text=text, parse_error=parse_error)

    def _complete_json_responses(self, prompt: str) -> ProviderResult:
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a careful football research assistant. Return valid JSON only.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }
        request = self._request("/responses", body)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", "ignore")
            raise ProviderError(f"HTTP {exc.code}: {error_text or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(str(exc)) from exc

        text = extract_responses_text(payload)
        parsed, parse_error = parse_json_object(text)
        return ProviderResult(payload=parsed, raw_text=text, parse_error=parse_error)


def parse_json_object(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}, None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}, "No JSON object found in model output."
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}, None
        except json.JSONDecodeError as exc:
            return {}, f"JSON parse failed: {exc}"


def extract_responses_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "".join(chunks)
    return json.dumps(payload, ensure_ascii=False)


def parse_env_lines(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def read_config_file(config_path: str | Path) -> dict[str, str]:
    try:
        lines = Path(config_path).expanduser().read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    return parse_env_lines(lines)


def read_local_config() -> dict[str, str]:
    override_path = os.getenv("WCP_CONFIG_ENV_PATH")
    if override_path:
        return read_config_file(override_path)

    values = read_config_file(DEFAULT_LOCAL_CONFIG_PATH)
    for key, value in read_config_file(DEFAULT_QWEN_CONFIG_PATH).items():
        values.setdefault(key, value)
    return values


def first_config_value(local_config: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    for name in names:
        value = local_config.get(name)
        if value:
            return value
    return None


def get_provider(name: str = "auto") -> BaseLLMProvider:
    normalized = (name or "auto").strip().lower()
    if normalized in {"mock", "sample", "deterministic"}:
        return DeterministicProvider()

    local_config = read_local_config()
    ddss_api_key = first_config_value(local_config, "DDSS_API_KEY")
    autodl_api_key = first_config_value(local_config, "AUTODL_API_KEY")
    openai_api_key = first_config_value(local_config, "OPENAI_API_KEY")
    wcp_openai_api_key = first_config_value(local_config, "WCP_OPENAI_API_KEY")
    if normalized == "auto":
        api_key = ddss_api_key or openai_api_key or wcp_openai_api_key or autodl_api_key
        backend = "ddss" if ddss_api_key else "openai" if (openai_api_key or wcp_openai_api_key) else "autodl"
    elif normalized == "ddss":
        api_key = ddss_api_key
        backend = "ddss"
    elif normalized in {"qwen", "autodl"}:
        api_key = autodl_api_key
        backend = "autodl"
    elif normalized in {"openai", "openai-compatible"}:
        api_key = openai_api_key or wcp_openai_api_key or ddss_api_key or autodl_api_key
        backend = "openai" if (openai_api_key or wcp_openai_api_key) else "ddss" if ddss_api_key else "autodl"
    else:
        raise ProviderError(f"Unknown LLM provider: {name}")

    if normalized in {"auto", "ddss", "qwen", "autodl", "openai", "openai-compatible"}:
        if not api_key:
            raise ProviderError(
                "真实 LLM 模式需要设置 AUTODL_API_KEY、DDSS_API_KEY、OPENAI_API_KEY 或 WCP_OPENAI_API_KEY；"
                "只有显式传入 --llm-provider mock 才会使用本地 deterministic mock。"
            )
        use_autodl = backend == "autodl"
        if use_autodl:
            model = first_config_value(local_config, "AUTODL_MODEL", "QWEN_MODEL") or "Qwen3.5-397B-A17B"
            base_url = first_config_value(local_config, "AUTODL_BASE_URL", "QWEN_BASE_URL") or "https://www.autodl.art/api/v1"
        elif backend == "ddss":
            model = first_config_value(local_config, "WCP_OPENAI_MODEL", "OPENAI_MODEL") or "gpt-5.5"
            base_url = first_config_value(local_config, "WCP_OPENAI_BASE_URL", "OPENAI_BASE_URL") or "https://code.ddsst.online/v1"
        else:
            model = first_config_value(local_config, "OPENAI_MODEL", "WCP_OPENAI_MODEL") or "gpt-5.5"
            base_url = first_config_value(local_config, "OPENAI_BASE_URL", "WCP_OPENAI_BASE_URL") or "https://api.openai.com/v1"
        default_wire_api = "responses" if base_url.rstrip("/") == "https://code.ddsst.online/v1" else "chat_completions"
        if use_autodl:
            wire_api = first_config_value(local_config, "AUTODL_WIRE_API", "QWEN_WIRE_API") or default_wire_api
        else:
            wire_api = first_config_value(local_config, "WCP_OPENAI_WIRE_API", "OPENAI_WIRE_API") or default_wire_api
        timeout = int(first_config_value(local_config, "WCP_OPENAI_TIMEOUT", "OPENAI_TIMEOUT") or "180")
        user_agent = first_config_value(local_config, "WCP_OPENAI_USER_AGENT", "OPENAI_USER_AGENT") or DEFAULT_USER_AGENT
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            wire_api=wire_api,
            timeout=timeout,
            user_agent=user_agent,
        )
