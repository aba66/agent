from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class AppBaseModel(BaseModel):
    """Base model used across the project."""

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {Path: str}


class SheetSummary(AppBaseModel):
    name: str
    rows: int
    columns: int
    column_names: list[str] = Field(default_factory=list)
    header_candidates: list[int] = Field(default_factory=list)
    preview_records: list[dict[str, Any]] = Field(default_factory=list)


class InputFileContext(AppBaseModel):
    file_path: Path
    file_type: str
    role: str = "data"
    sheet_summaries: list[SheetSummary] = Field(default_factory=list)
    default_sheet: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def sheet_names(self) -> list[str]:
        return [sheet.name for sheet in self.sheet_summaries]


class WorkbookContext(AppBaseModel):
    file_path: Path | None = None
    file_type: str = "multi"
    sheet_summaries: list[SheetSummary] = Field(default_factory=list)
    default_sheet: str | None = None
    warnings: list[str] = Field(default_factory=list)
    input_files: list[InputFileContext] = Field(default_factory=list)
    template_file: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def sheet_names(self) -> list[str]:
        if self.input_files:
            names: list[str] = []
            for file_context in self.input_files:
                if file_context.role != "data":
                    continue
                for sheet_name in file_context.sheet_names:
                    if sheet_name not in names:
                        names.append(sheet_name)
            return names
        return [sheet.name for sheet in self.sheet_summaries]


class RequirementSpec(AppBaseModel):
    raw_request: str
    intent_type: str = "new_request"
    requested_metrics: list[str] = Field(default_factory=list)
    metric_groups: dict[str, list[str]] = Field(default_factory=dict)
    metric_group_mode: str = "family_summary"
    source_sheet: str | None = None
    output_formats: list[str] = Field(default_factory=list)
    output_sheet_name: str = "KPI Summary"
    template_sheet_name: str | None = None
    stage_value: str | None = None
    remark_value: str | None = None
    template_hint: str | None = None
    need_chart: bool = False
    need_report: bool = True
    need_pdf: bool = True
    need_docx: bool = True
    strict_template: bool = True
    decay_method: str | None = None
    analysis_mode: str = "detail"
    aggregate_by_fields: list[str] = Field(default_factory=list)
    requested_statistics: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    threshold_rules: dict[str, Any] = Field(default_factory=dict)
    llm_notes: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    parsed_notes: list[str] = Field(default_factory=list)
    context_inheritance_notes: list[str] = Field(default_factory=list)
    revision_note: str | None = None


class MetricTargetPlan(AppBaseModel):
    label: str
    target_type: str = "metric_group"
    family: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    columns: list[str] = Field(default_factory=list)
    matched_column: str | None = None


class ReviewTargetPlan(AppBaseModel):
    dimension: str
    value: str
    metric_labels: list[str] = Field(default_factory=list)
    reason: str = ""


class AnalysisPlan(AppBaseModel):
    raw_request: str
    intent_type: str = "new_request"
    source_sheet: str | None = None
    group_dimensions: list[str] = Field(default_factory=list)
    metric_targets: list[MetricTargetPlan] = Field(default_factory=list)
    statistics: list[str] = Field(default_factory=list)
    decay_method: str | None = None
    output_granularity: str = "detail"
    output_formats: list[str] = Field(default_factory=list)
    output_sheet_name: str = "KPI Summary"
    template_sheet_name: str | None = None
    template_hint: str | None = None
    stage_value: str | None = None
    remark_value: str | None = None
    threshold_rules: dict[str, Any] = Field(default_factory=dict)
    review_targets: list[ReviewTargetPlan] = Field(default_factory=list)
    llm_notes: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    parsed_notes: list[str] = Field(default_factory=list)
    context_inheritance_notes: list[str] = Field(default_factory=list)
    revision_note: str | None = None


class MetricMatch(AppBaseModel):
    requested_name: str
    matched_column: str | None = None
    score: float = 0.0
    reason: str = ""
    status: str = "missing"


class AnalysisArtifacts(AppBaseModel):
    output_dir: Path | None = None
    excel_path: Path | None = None
    markdown_path: Path | None = None
    audit_log_path: Path | None = None
    audit_report_path: Path | None = None
    pdf_path: Path | None = None
    docx_path: Path | None = None
    chart_paths: list[Path] = Field(default_factory=list)


class LLMSettings(AppBaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1800
    enable_thinking: bool | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    def masked(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        if self.api_key:
            value["api_key"] = f"{self.api_key[:4]}***"
        return value


class SkillDefinition(AppBaseModel):
    name: str
    instruction: str = ""
    references: list[str] = Field(default_factory=list)
    path: Path | None = None


class MappingResolution(AppBaseModel):
    requested_name: str
    matched_column: str | None = None
    confidence: float = 0.0
    reason: str = ""


class MappingPlan(AppBaseModel):
    resolved_metrics: list[MappingResolution] = Field(default_factory=list)
    base_field_overrides: dict[str, str] = Field(default_factory=dict)
    clarification_questions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class MCPCallRecord(AppBaseModel):
    server: str
    action: str
    target: str
    status: str
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    summary: str = ""


class AgentCard(AppBaseModel):
    agent_id: str
    title: str
    description: str
    skills: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class A2AHandoff(AppBaseModel):
    from_agent: str
    to_agent: str
    task_type: str
    summary: str
    created_at: str
    payload_preview: dict[str, Any] = Field(default_factory=dict)


class AgentTraceEvent(AppBaseModel):
    agent: str
    step: str
    status: str
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    summary: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(AppBaseModel):
    source_sheet: str
    source_files: list[str] = Field(default_factory=list)
    base_field_mapping: dict[str, str] = Field(default_factory=dict)
    metric_matches: list[MetricMatch] = Field(default_factory=list)
    summary_frame: pd.DataFrame | None = None
    summary_preview: list[dict[str, Any]] = Field(default_factory=list)
    statistics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    validation_messages: list[str] = Field(default_factory=list)
    comparison_notes: list[str] = Field(default_factory=list)
    audit_records: list[dict[str, Any]] = Field(default_factory=list)
    recompute_reviews: list[dict[str, Any]] = Field(default_factory=list)
    narrative_summary: list[str] = Field(default_factory=list)
    llm_notes: list[str] = Field(default_factory=list)
    chart_figures: dict[str, Any] = Field(default_factory=dict)
    artifacts: AnalysisArtifacts = Field(default_factory=AnalysisArtifacts)
    trace_events: list[AgentTraceEvent] = Field(default_factory=list)


class ConversationContext(AppBaseModel):
    current_message: str
    active_task_id: str | None = None
    history_summary: list[str] = Field(default_factory=list)
    latest_confirmed_plan: dict[str, Any] = Field(default_factory=dict)
    latest_confirmed_requirement: dict[str, Any] = Field(default_factory=dict)
    inherited_fields: list[str] = Field(default_factory=list)
    pending_clarifications: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class UserContext(AppBaseModel):
    username: str = "guest"
    role: str = "viewer"


class TaskRecord(AppBaseModel):
    task_id: str
    session_id: str
    parent_task_id: str | None = None
    revision: int = 1
    title: str = ""
    username: str
    role: str
    status: str
    request_text: str
    input_files: list[str] = Field(default_factory=list)
    template_file: str | None = None
    requirement: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    clarifications: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str
    updated_at: str


class WorkflowResult(AppBaseModel):
    status: str
    workbook: WorkbookContext
    analysis_plan: AnalysisPlan | None = None
    requirement: RequirementSpec
    analysis: AnalysisResult | None = None
    conversation_context: ConversationContext | None = None
    clarifications: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    session_id: str | None = None
    task_id: str | None = None
    revision: int | None = None
    trace_events: list[AgentTraceEvent] = Field(default_factory=list)
    agent_cards: list[AgentCard] = Field(default_factory=list)
    a2a_handoffs: list[A2AHandoff] = Field(default_factory=list)
    mcp_calls: list[MCPCallRecord] = Field(default_factory=list)
    task_record: TaskRecord | None = None
