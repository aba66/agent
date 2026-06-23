from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_text
from .models import outcome_from_score, utc_now_iso


@dataclass
class ReviewSummary:
    date: str
    reviewed: int
    hits: int
    misses: int
    cumulative_reviewed: int
    cumulative_hits: int
    cumulative_accuracy: float
    entries: list[dict[str, Any]]


class LedgerManager:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.ledger_dir = project_root / "ledger"
        self.results_path = self.ledger_dir / "results_ledger.jsonl"
        self.lessons_path = self.ledger_dir / "lessons.md"

    def read_lessons(self) -> str:
        if not self.lessons_path.exists():
            return ""
        return self.lessons_path.read_text(encoding="utf-8")

    def lesson_lines(self, limit: int = 8) -> list[str]:
        lessons = []
        for line in self.read_lessons().splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                lessons.append(stripped.lstrip("- ").strip())
        return lessons[-limit:]

    def review(self, date: str, results_path: Path) -> ReviewSummary:
        predictions_path = self.project_root / "runs" / date / "predictions.json"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        predictions_payload = read_json(predictions_path)
        results_payload = read_json(results_path)
        results_by_id = self._normalize_results(results_payload)
        match_by_id = {item["match_id"]: item for item in predictions_payload.get("matches", [])}

        new_entries = []
        misses = []
        for prediction in predictions_payload.get("predictions", []):
            match_id = prediction["match_id"]
            if match_id not in results_by_id:
                continue
            result = results_by_id[match_id]
            actual = result.get("actual_outcome")
            if not actual:
                actual = outcome_from_score(int(result["home_score"]), int(result["away_score"]))
            hit = prediction["predicted_outcome"] == actual
            match = match_by_id.get(match_id, {})
            entry = {
                "run_date": date,
                "reviewed_at": utc_now_iso(),
                "match_id": match_id,
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "predicted_outcome": prediction["predicted_outcome"],
                "actual_outcome": actual,
                "home_score": result.get("home_score"),
                "away_score": result.get("away_score"),
                "hit": hit,
                "confidence": prediction.get("confidence"),
                "max_risk": prediction.get("max_risk"),
            }
            new_entries.append(entry)
            if not hit:
                misses.append(entry)

        existing = self._read_ledger_entries()
        existing = [entry for entry in existing if not (entry.get("run_date") == date and entry.get("match_id") in results_by_id)]
        all_entries = existing + new_entries
        self._write_ledger_entries(all_entries)
        self._append_lessons(date, new_entries, misses)

        cumulative_reviewed = len(all_entries)
        cumulative_hits = sum(1 for entry in all_entries if entry.get("hit"))
        accuracy = cumulative_hits / cumulative_reviewed if cumulative_reviewed else 0.0
        hits = sum(1 for entry in new_entries if entry["hit"])
        return ReviewSummary(
            date=date,
            reviewed=len(new_entries),
            hits=hits,
            misses=len(new_entries) - hits,
            cumulative_reviewed=cumulative_reviewed,
            cumulative_hits=cumulative_hits,
            cumulative_accuracy=accuracy,
            entries=new_entries,
        )

    def stats(self) -> dict[str, Any]:
        entries = self._read_ledger_entries()
        total = len(entries)
        hits = sum(1 for entry in entries if entry.get("hit"))
        by_confidence: dict[str, dict[str, int]] = {}
        for entry in entries:
            confidence = str(entry.get("confidence") or "unknown")
            bucket = by_confidence.setdefault(confidence, {"reviewed": 0, "hits": 0})
            bucket["reviewed"] += 1
            bucket["hits"] += 1 if entry.get("hit") else 0
        return {
            "reviewed": total,
            "hits": hits,
            "misses": total - hits,
            "accuracy": hits / total if total else 0.0,
            "by_confidence": by_confidence,
        }

    @staticmethod
    def _normalize_results(payload: Any) -> dict[str, dict[str, Any]]:
        if isinstance(payload, dict):
            rows = payload.get("results") or payload.get("matches") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        normalized = {}
        for row in rows:
            match_id = str(row["match_id"])
            item = dict(row)
            if "actual_outcome" not in item and "home_score" in item and "away_score" in item:
                item["actual_outcome"] = outcome_from_score(int(item["home_score"]), int(item["away_score"]))
            normalized[match_id] = item
        return normalized

    def _read_ledger_entries(self) -> list[dict[str, Any]]:
        if not self.results_path.exists():
            return []
        entries = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            import json

            entries.append(json.loads(line))
        return entries

    def _write_ledger_entries(self, entries: list[dict[str, Any]]) -> None:
        ensure_dir(self.ledger_dir)
        import json

        content = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
        write_text(self.results_path, content)

    def _append_lessons(self, date: str, entries: list[dict[str, Any]], misses: list[dict[str, Any]]) -> None:
        ensure_dir(self.ledger_dir)
        old = self.read_lessons()
        lines = [old.rstrip(), "", f"## {date} review", ""]
        if not entries:
            lines.append("- No matching results were supplied for this date.")
        elif not misses:
            lines.append("- All reviewed predictions hit. Keep the current evidence weighting, but continue checking lineup risk.")
        else:
            for miss in misses:
                home = miss.get("home_team") or "Home"
                away = miss.get("away_team") or "Away"
                lines.append(
                    "- "
                    + f"{home} vs {away}: predicted {miss['predicted_outcome']}, actual {miss['actual_outcome']}. "
                    + self._lesson_for_miss(miss)
                )
        lines.append("")
        write_text(self.lessons_path, "\n".join(line for line in lines if line is not None).lstrip())

    @staticmethod
    def _lesson_for_miss(miss: dict[str, Any]) -> str:
        if miss["actual_outcome"] == "DRAW":
            return "Lesson: draw probability was underestimated; raise weight for conservative favorites and low-block resilience."
        if miss["predicted_outcome"] == "DRAW":
            return "Lesson: draw hedge was too conservative; check whether one side has a decisive attacking or squad edge."
        return "Lesson: upset path was underweighted; inspect squad news, transition threat, and set-piece mismatch before next run."
