from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_analyst.models import UserContext
from test_analyst.workflow import TestAnalystWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the enhanced and advanced test-analyst workflow.")
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="Path to a CSV/Excel data file. Repeat this argument to analyze multiple files together.",
    )
    parser.add_argument("--template-file", help="Optional output template workbook path.")
    parser.add_argument("--request", help="Natural language requirement.")
    parser.add_argument("--request-file", help="Text file containing the requirement.")
    parser.add_argument("--sheet", help="Override the source sheet.")
    parser.add_argument("--metrics", nargs="*", help="Override metric names.")
    parser.add_argument("--stage", help="Manual stage value, such as EVT or DVT.")
    parser.add_argument("--remark", help="Manual remark value.")
    parser.add_argument("--revision-note", help="Optional note describing this revision.")
    parser.add_argument(
        "--decay-method",
        choices=["relative_to_first", "delta_from_first", "delta_from_previous"],
        help="Decay method override.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        help="Detail-mode grouping keys: stage / test_item / sn / test_node / source_file",
    )
    parser.add_argument(
        "--aggregate-by",
        nargs="*",
        default=[],
        help="Summary dimensions: test_item / sn / stage / test_node / source_file",
    )
    parser.add_argument(
        "--statistic",
        dest="requested_statistics",
        action="append",
        default=[],
        help="Summary statistics to output: min / max / mean / decay. Repeatable.",
    )
    parser.add_argument("--output-sheet-name", default="KPI Summary", help="Output Excel sheet name.")
    parser.add_argument("--template-sheet-name", help="Template sheet to copy and populate.")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Threshold rule, for example: ANSIContrast_Res_LvContRatio>=20. Repeatable.",
    )
    parser.add_argument("--without-chart", action="store_true", help="Disable chart generation.")
    parser.add_argument("--without-report", action="store_true", help="Disable markdown report generation.")
    parser.add_argument("--without-pdf", action="store_true", help="Disable PDF report generation.")
    parser.add_argument("--without-docx", action="store_true", help="Disable Docx report generation.")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output"), help="Artifact output directory.")
    parser.add_argument("--session-id", help="Reuse an existing session id.")
    parser.add_argument("--parent-task-id", help="Create a new revision based on an existing task id.")
    parser.add_argument("--session-title", default="命令行分析会话", help="Title used for history tracking.")
    parser.add_argument("--username", default="analyst", help="Recorded username for task history.")
    parser.add_argument("--role", default="analyst", help="Recorded role for task history.")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-api-key", help="OpenAI-compatible API key.")
    parser.add_argument("--llm-model", help="LLM model name.")
    parser.add_argument(
        "--llm-enable-thinking",
        dest="llm_enable_thinking",
        action="store_const",
        const=True,
        default=None,
        help="Explicitly enable provider-side thinking mode when supported.",
    )
    parser.add_argument(
        "--llm-disable-thinking",
        dest="llm_enable_thinking",
        action="store_const",
        const=False,
        help="Explicitly disable provider-side thinking mode when supported.",
    )
    return parser


def load_request_text(args: argparse.Namespace) -> str:
    if args.request:
        return args.request
    if args.request_file:
        return Path(args.request_file).read_text(encoding="utf-8")
    raise SystemExit("请使用 --request 或 --request-file 提供需求描述。")


def parse_thresholds(raw_rules: list[str]) -> dict[str, str]:
    rules: dict[str, str] = {}
    for raw in raw_rules:
        text = raw.strip()
        if not text:
            continue
        for operator in (">=", "<=", ">", "<", "="):
            if operator in text:
                key, value = text.split(operator, 1)
                rules[key.strip()] = f"{operator} {value.strip()}"
                break
    return rules


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    request_text = load_request_text(args)
    workflow = TestAnalystWorkflow(project_root=PROJECT_ROOT, output_root=args.output_root)
    overrides = {
        "source_sheet": args.sheet,
        "requested_metrics": args.metrics or [],
        "stage_value": args.stage,
        "remark_value": args.remark,
        "decay_method": args.decay_method,
        "aggregate_by_fields": args.aggregate_by or [],
        "requested_statistics": args.requested_statistics or [],
        "group_by": args.group_by or [],
        "output_sheet_name": args.output_sheet_name,
        "template_sheet_name": args.template_sheet_name,
        "need_chart": not args.without_chart,
        "need_report": not args.without_report,
        "need_pdf": not args.without_pdf,
        "need_docx": not args.without_docx,
        "threshold_rules": parse_thresholds(args.threshold),
        "revision_note": args.revision_note,
    }
    user_context = UserContext(username=args.username, role=args.role)
    result = workflow.run(
        data_files=args.files,
        request_text=request_text,
        template_file=args.template_file,
        overrides=overrides,
        session_id=args.session_id,
        parent_task_id=args.parent_task_id,
        user_context=user_context,
        session_title=args.session_title,
        llm_settings={
            "base_url": args.llm_base_url,
            "api_key": args.llm_api_key,
            "model": args.llm_model,
            "enable_thinking": args.llm_enable_thinking,
        },
    )

    print("\n=== Workflow Status ===")
    print(result.status)
    print(f"session_id: {result.session_id}")
    print(f"task_id: {result.task_id}")
    print(f"revision: {result.revision}")

    print("\n=== Analysis Plan ===")
    print(json.dumps(result.analysis_plan.model_dump(mode="json") if result.analysis_plan else {}, ensure_ascii=False, indent=2))

    print("\n=== Legacy Requirement View ===")
    print(json.dumps(result.requirement.model_dump(mode="json"), ensure_ascii=False, indent=2))

    print("\n=== Agent Trace ===")
    print(json.dumps([event.model_dump(mode="json") for event in result.trace_events], ensure_ascii=False, indent=2))

    print("\n=== A2A Handoffs ===")
    print(json.dumps([item.model_dump(mode="json") for item in result.a2a_handoffs], ensure_ascii=False, indent=2))

    print("\n=== MCP Calls ===")
    print(json.dumps([item.model_dump(mode="json") for item in result.mcp_calls], ensure_ascii=False, indent=2))

    if result.status == "needs_clarification":
        print("\n=== Clarifications ===")
        for question in result.clarifications:
            print(f"- {question}")
        return 1

    if result.status == "failed":
        print("\n=== Errors ===")
        for error in result.errors:
            print(f"- {error}")
        return 2

    assert result.analysis is not None
    print("\n=== Validation ===")
    for message in result.analysis.validation_messages:
        print(f"- {message}")

    print("\n=== Warnings ===")
    if result.analysis.warnings:
        for warning in result.analysis.warnings:
            print(f"- {warning}")
    else:
        print("- 无")

    print("\n=== Multi-file Notes ===")
    if result.analysis.comparison_notes:
        for note in result.analysis.comparison_notes:
            print(f"- {note}")
    else:
        print("- 无")

    print("\n=== Statistics ===")
    print(json.dumps(result.analysis.statistics, ensure_ascii=False, indent=2))

    print("\n=== Artifacts ===")
    print(json.dumps(result.analysis.artifacts.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
