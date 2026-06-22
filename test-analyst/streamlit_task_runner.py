from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_analyst.models import UserContext
from test_analyst.streamlit_payloads import serialize_workflow_result
from test_analyst.utils import write_json
from test_analyst.workflow import TestAnalystWorkflow


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: streamlit_task_runner.py <payload.json> <result.json>")

    payload_path = Path(sys.argv[1]).resolve()
    result_path = Path(sys.argv[2]).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    workflow = TestAnalystWorkflow(
        project_root=payload["project_root"],
        output_root=payload.get("output_root"),
    )
    result = workflow.run(
        data_files=payload.get("data_files"),
        request_text=payload["request_text"],
        template_file=payload.get("template_file"),
        overrides=payload.get("overrides"),
        session_id=payload.get("session_id"),
        parent_task_id=payload.get("parent_task_id"),
        user_context=UserContext(**payload["user_context"]),
        session_title=payload["session_title"],
        llm_settings=payload.get("llm_settings"),
    )
    write_json(result_path, serialize_workflow_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
