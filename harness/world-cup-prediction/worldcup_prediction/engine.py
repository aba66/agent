from __future__ import annotations

import logging
from pathlib import Path

from .agents import AgentOrchestrator, MatchAnalysisResult, build_prediction_input_match, build_prediction_input_sources, write_agent_markdown
from .data_sources import get_data_source
from .io_utils import ensure_dir, project_root_from_package, write_json, write_text
from .ledger import LedgerManager, ReviewSummary
from .models import FinalPrediction, RoleAnalysis, utc_now_iso
from .providers import get_provider
from .reports import render_daily_report


def run_predict(
    date: str,
    source: str = "sample",
    timezone_name: str = "Asia/Shanghai",
    llm_provider: str = "auto",
    project_root: Path | None = None,
    source_json: Path | None = None,
) -> Path:
    root = project_root or project_root_from_package()
    run_dir = root / "runs" / date
    ensure_dir(run_dir)
    logger = _setup_logger(run_dir / "run.log")
    logger.info("Starting prediction run date=%s source=%s timezone=%s", date, source, timezone_name)

    data_source = get_data_source(source, source_json=source_json)
    snapshot = data_source.load_matches(date, timezone_name)
    logger.info("Loaded %d match records from %s", len(snapshot.matches), data_source.name)
    if not snapshot.matches:
        logger.warning("No matches found for %s", date)

    ledger = LedgerManager(root)
    lessons_text = ledger.read_lessons()
    lessons_used = ledger.lesson_lines()
    logger.info("Loaded %d lesson lines", len(lessons_used))

    provider = get_provider(llm_provider)
    logger.info("Using provider=%s", provider.name)
    source_dicts = [source.to_dict() for source in snapshot.sources]

    if getattr(provider, "supports_daily_batch", False):
        logger.info("Using daily batch generation for %d matches", len(snapshot.matches))
        prediction_matches = [build_prediction_input_match(match) for match in snapshot.matches]
        prediction_sources = build_prediction_input_sources(source_dicts)
        batch_result = provider.generate_daily_batch(prediction_matches, prediction_sources, lessons_text)
        results = _batch_payload_to_results(snapshot.matches, batch_result.payload, batch_result.raw_text, batch_result.parse_error)
    else:
        orchestrator = AgentOrchestrator(provider)
        results = []
        for match in snapshot.matches:
            logger.info("Analyzing match_id=%s %s vs %s", match.match_id, match.home_team, match.away_team)
            result = orchestrator.analyze_match(match, source_dicts, lessons_text)
            results.append(result)

    for result in results:
        write_agent_markdown(run_dir / "agents", result)

    predictions = [result.final_prediction for result in results]
    write_json(run_dir / "matches.json", {"date": date, "matches": [match.to_dict() for match in snapshot.matches]})
    write_json(run_dir / "sources.json", {"generated_at": utc_now_iso(), "sources": source_dicts})
    write_json(
        run_dir / "predictions.json",
        {
            "date": date,
            "generated_at": utc_now_iso(),
            "provider": provider.name,
            "lessons_used": lessons_used,
            "matches": [match.to_dict() for match in snapshot.matches],
            "role_analyses": [
                analysis.to_dict()
                for result in results
                for analysis in result.role_analyses
            ],
            "predictions": [prediction.to_dict() for prediction in predictions],
        },
    )
    report = render_daily_report(date, snapshot.matches, predictions, snapshot.sources, lessons_used)
    write_text(run_dir / "daily_report.md", report)
    logger.info("Prediction run complete: %s", run_dir)
    return run_dir


def _batch_payload_to_results(matches, payload, raw_text: str, parse_error: str | None):
    matches_by_id = {match.match_id: match for match in matches}
    roles_by_match = {match.match_id: [] for match in matches}
    for item in payload.get("role_analyses", []):
        match_id = str(item.get("match_id") or "")
        match = matches_by_id.get(match_id)
        if not match:
            continue
        role = str(item.get("role") or "")
        roles_by_match[match_id].append(
            RoleAnalysis.from_payload(role=role, match=match, payload=item, raw_text=raw_text, parse_error=parse_error)
        )

    predictions_by_match = {}
    for item in payload.get("predictions", []):
        match_id = str(item.get("match_id") or "")
        match = matches_by_id.get(match_id)
        if not match:
            continue
        predictions_by_match[match_id] = FinalPrediction.from_payload(match=match, payload=item, raw_text=raw_text, parse_error=parse_error)

    results = []
    for match in matches:
        roles = roles_by_match.get(match.match_id, [])
        if len(roles) < 4:
            raise ValueError(f"Batch LLM output missing role analyses for {match.match_id}")
        if match.match_id not in predictions_by_match:
            raise ValueError(f"Batch LLM output missing final prediction for {match.match_id}")
        results.append(
            MatchAnalysisResult(
                match=match,
                role_analyses=roles,
                final_prediction=predictions_by_match[match.match_id],
            )
        )
    return results


def run_review(date: str, results_path: Path, project_root: Path | None = None) -> ReviewSummary:
    root = project_root or project_root_from_package()
    return LedgerManager(root).review(date, results_path)


def read_report(date: str, project_root: Path | None = None) -> str:
    root = project_root or project_root_from_package()
    path = root / "runs" / date / "daily_report.md"
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return path.read_text(encoding="utf-8")


def read_stats(project_root: Path | None = None) -> dict:
    root = project_root or project_root_from_package()
    return LedgerManager(root).stats()


def _setup_logger(path: Path) -> logging.Logger:
    ensure_dir(path.parent)
    logger = logging.getLogger(f"worldcup_prediction.{path}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
