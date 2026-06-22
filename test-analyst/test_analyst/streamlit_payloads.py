from __future__ import annotations

from typing import Any

import pandas as pd

from .models import WorkflowResult


def serialize_workflow_result(result: WorkflowResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    analysis = payload.get("analysis")
    if analysis is not None:
        frame = result.analysis.summary_frame if result.analysis else None
        analysis["summary_frame"] = (
            {
                "columns": [str(column) for column in frame.columns],
                "records": frame.where(frame.notna(), None).to_dict(orient="records"),
            }
            if frame is not None
            else None
        )
    return payload


def deserialize_workflow_result(payload: dict[str, Any]) -> WorkflowResult:
    hydrated = dict(payload)
    analysis = hydrated.get("analysis")
    if isinstance(analysis, dict):
        frame_payload = analysis.get("summary_frame")
        if isinstance(frame_payload, dict):
            analysis["summary_frame"] = pd.DataFrame(
                frame_payload.get("records", []),
                columns=frame_payload.get("columns", []),
            )
        elif frame_payload is None:
            analysis["summary_frame"] = None
    return WorkflowResult.model_validate(hydrated)
