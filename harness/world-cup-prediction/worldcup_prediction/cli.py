from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import read_report, read_stats, run_predict, run_review
from .io_utils import project_root_from_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a World Cup prediction harness.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root_from_package(),
        help="Project root containing runs/ and ledger/. Defaults to this package directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run fact-check -> multi-agent prediction.")
    predict.add_argument("--date", required=True, help="Run date in YYYY-MM-DD.")
    predict.add_argument("--source", default="sample", help="Data source name. Default: sample.")
    predict.add_argument("--source-json", type=Path, help="Optional local JSON data source.")
    predict.add_argument("--timezone", default="Asia/Shanghai", help="Local timezone. Default: Asia/Shanghai.")
    predict.add_argument(
        "--llm-provider",
        default="auto",
        choices=["auto", "ddss", "qwen", "autodl", "mock", "openai", "openai-compatible"],
        help="LLM provider. Default auto prefers DDSS_API_KEY; qwen/autodl uses Qwen3.5-397B-A17B by default.",
    )

    review = subparsers.add_parser("review", help="Record real results and update ledger/lessons.")
    review.add_argument("--date", required=True, help="Run date in YYYY-MM-DD.")
    review.add_argument("--results", type=Path, required=True, help="Path to results JSON.")

    report = subparsers.add_parser("report", help="Print a generated daily report.")
    report.add_argument("--date", required=True, help="Run date in YYYY-MM-DD.")

    subparsers.add_parser("stats", help="Print cumulative ledger statistics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.project_root

    if args.command == "predict":
        run_dir = run_predict(
            date=args.date,
            source=args.source,
            timezone_name=args.timezone,
            llm_provider=args.llm_provider,
            project_root=root,
            source_json=args.source_json,
        )
        print(f"Prediction run written to: {run_dir}")
        return 0

    if args.command == "review":
        summary = run_review(date=args.date, results_path=args.results, project_root=root)
        print(
            f"Reviewed {summary.reviewed} matches: {summary.hits} hit(s), {summary.misses} miss(es). "
            f"Cumulative accuracy: {summary.cumulative_hits}/{summary.cumulative_reviewed} "
            f"({summary.cumulative_accuracy:.1%})."
        )
        return 0

    if args.command == "report":
        print(read_report(args.date, project_root=root))
        return 0

    if args.command == "stats":
        print(json.dumps(read_stats(project_root=root), ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
