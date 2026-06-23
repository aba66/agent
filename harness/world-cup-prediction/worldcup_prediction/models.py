from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


OUTCOMES = {"HOME_WIN", "DRAW", "AWAY_WIN"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_confidence(value: str | None) -> str:
    normalized = (value or "medium").strip().lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    if normalized in {"h", "strong"}:
        return "high"
    if normalized in {"l", "weak"}:
        return "low"
    return "medium"


def normalize_outcome(value: str | None, home_team: str = "", away_team: str = "") -> str:
    raw = (value or "").strip()
    upper = raw.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "HOME": "HOME_WIN",
        "HOMEWIN": "HOME_WIN",
        "HOME_WIN": "HOME_WIN",
        "1": "HOME_WIN",
        "H": "HOME_WIN",
        "DRAW": "DRAW",
        "TIE": "DRAW",
        "X": "DRAW",
        "0": "DRAW",
        "AWAY": "AWAY_WIN",
        "AWAYWIN": "AWAY_WIN",
        "AWAY_WIN": "AWAY_WIN",
        "2": "AWAY_WIN",
        "A": "AWAY_WIN",
    }
    if upper in aliases:
        return aliases[upper]
    if raw and home_team and raw.casefold() == home_team.casefold():
        return "HOME_WIN"
    if raw and away_team and raw.casefold() == away_team.casefold():
        return "AWAY_WIN"
    return "DRAW"


def outcome_from_score(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "HOME_WIN"
    if home_score < away_score:
        return "AWAY_WIN"
    return "DRAW"


def outcome_label(outcome: str, home_team: str, away_team: str) -> str:
    normalized = normalize_outcome(outcome, home_team, away_team)
    if normalized == "HOME_WIN":
        return f"{home_team} win"
    if normalized == "AWAY_WIN":
        return f"{away_team} win"
    return "Draw"


@dataclass
class SourceRecord:
    source_id: str
    name: str
    kind: str
    url: str | None = None
    fetched_at: str = field(default_factory=utc_now_iso)
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceRecord":
        return cls(
            source_id=str(payload.get("source_id") or payload.get("id") or "source"),
            name=str(payload.get("name") or "Unnamed source"),
            kind=str(payload.get("kind") or "unknown"),
            url=payload.get("url"),
            fetched_at=str(payload.get("fetched_at") or utc_now_iso()),
            notes=str(payload.get("notes") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchRecord:
    match_id: str
    date: str
    kickoff_local: str
    timezone: str
    home_team: str
    away_team: str
    venue: str
    stage: str = "Group"
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None
    sources: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MatchRecord":
        score = payload.get("score") or {}
        return cls(
            match_id=str(payload["match_id"]),
            date=str(payload["date"]),
            kickoff_local=str(payload["kickoff_local"]),
            timezone=str(payload.get("timezone") or "Asia/Shanghai"),
            home_team=str(payload["home_team"]),
            away_team=str(payload["away_team"]),
            venue=str(payload.get("venue") or "Unknown venue"),
            stage=str(payload.get("stage") or "Group"),
            status=str(payload.get("status") or "scheduled"),
            home_score=payload.get("home_score", score.get("home")),
            away_score=payload.get("away_score", score.get("away")),
            sources=[str(item) for item in payload.get("sources", [])],
            facts=dict(payload.get("facts") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def actual_outcome(self) -> str | None:
        if self.home_score is None or self.away_score is None:
            return None
        return outcome_from_score(int(self.home_score), int(self.away_score))


@dataclass
class DataSnapshot:
    matches: list[MatchRecord]
    sources: list[SourceRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": [match.to_dict() for match in self.matches],
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass
class Citation:
    source_id: str
    note: str = ""

    @classmethod
    def from_any(cls, payload: Any) -> "Citation":
        if isinstance(payload, dict):
            return cls(
                source_id=str(payload.get("source_id") or payload.get("id") or "unknown"),
                note=str(payload.get("note") or payload.get("quote") or ""),
            )
        return cls(source_id=str(payload), note="")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoleAnalysis:
    match_id: str
    role: str
    predicted_outcome: str
    conclusion: str
    confidence: str
    key_evidence: list[str]
    risks: list[str]
    citations: list[Citation]
    one_line: str
    raw_text: str = ""
    parse_error: str | None = None

    @classmethod
    def from_payload(
        cls,
        role: str,
        match: MatchRecord,
        payload: dict[str, Any],
        raw_text: str = "",
        parse_error: str | None = None,
    ) -> "RoleAnalysis":
        conclusion = str(payload.get("conclusion") or payload.get("summary") or "")
        one_line = str(payload.get("one_line") or conclusion[:160] or f"{role} completed analysis.")
        return cls(
            match_id=match.match_id,
            role=role,
            predicted_outcome=normalize_outcome(
                str(payload.get("predicted_outcome") or payload.get("outcome") or ""),
                match.home_team,
                match.away_team,
            ),
            conclusion=conclusion,
            confidence=normalize_confidence(str(payload.get("confidence") or "medium")),
            key_evidence=[str(item) for item in payload.get("key_evidence", [])],
            risks=[str(item) for item in payload.get("risks", [])],
            citations=[Citation.from_any(item) for item in payload.get("citations", match.sources)],
            one_line=one_line,
            raw_text=raw_text,
            parse_error=parse_error,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        return payload


@dataclass
class FinalPrediction:
    match_id: str
    predicted_outcome: str
    confidence: str
    key_evidence: list[str]
    max_risk: str
    role_summaries: dict[str, str]
    agreements: list[str]
    conflicts: list[str]
    decision: str
    citations: list[Citation]
    raw_text: str = ""
    parse_error: str | None = None

    @classmethod
    def from_payload(
        cls,
        match: MatchRecord,
        payload: dict[str, Any],
        raw_text: str = "",
        parse_error: str | None = None,
    ) -> "FinalPrediction":
        return cls(
            match_id=match.match_id,
            predicted_outcome=normalize_outcome(
                str(payload.get("predicted_outcome") or payload.get("outcome") or ""),
                match.home_team,
                match.away_team,
            ),
            confidence=normalize_confidence(str(payload.get("confidence") or "medium")),
            key_evidence=[str(item) for item in payload.get("key_evidence", [])],
            max_risk=str(payload.get("max_risk") or "Unspecified risk."),
            role_summaries={str(k): str(v) for k, v in dict(payload.get("role_summaries", {})).items()},
            agreements=[str(item) for item in payload.get("agreements", [])],
            conflicts=[str(item) for item in payload.get("conflicts", [])],
            decision=str(payload.get("decision") or payload.get("conclusion") or ""),
            citations=[Citation.from_any(item) for item in payload.get("citations", match.sources)],
            raw_text=raw_text,
            parse_error=parse_error,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = [citation.to_dict() for citation in self.citations]
        return payload
