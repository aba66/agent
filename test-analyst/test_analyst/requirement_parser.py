from __future__ import annotations

import re
from typing import Any
from typing import Iterable

from .configuration import ConfigManager
from .models import AnalysisPlan, ConversationContext, MetricTargetPlan, RequirementSpec, ReviewTargetPlan, WorkbookContext
from .utils import normalize_text, unique_preserve_order

BASE_FIELD_KEYWORDS = {
    normalize_text("阶段"),
    normalize_text("测试项目"),
    normalize_text("测试项"),
    normalize_text("sn"),
    normalize_text("sn号"),
    normalize_text("测试节点"),
    normalize_text("备注"),
    normalize_text("remark"),
}

BASE_FIELD_ALIASES = {
    "stage": ["阶段", "phase", "stage"],
    "test_item": ["测试项目", "测试项", "test_item", "item", "项目"],
    "sn": ["sn", "sn号", "序列号", "serialnumber", "serial"],
    "test_node": ["测试节点", "节点", "testnode", "node", "station"],
    "remark": ["备注", "remark", "comment", "note"],
    "source_file": ["来源文件", "source_file", "file", "文件"],
}

DECAY_KEYWORDS = {
    "relative_to_first": ["衰减", "变化率", "相对首条", "相对首行", "相对初始", "relative"],
    "delta_from_first": ["差值", "绝对衰减", "绝对变化", "delta"],
    "delta_from_previous": ["环比", "前一条", "相邻差值", "previous"],
}

STATISTIC_KEYWORDS = {
    "min": ["最小值", "最小", "minimum", "min"],
    "max": ["最大值", "最大", "maximum", "max"],
    "mean": ["平均值", "均值", "平均", "avg", "mean"],
    "decay": ["衰减", "衰减率", "变化率", "delta", "relative"],
}


class RequirementOrchestratorAgent:
    """Hybrid candidate extractor and guardrail sanitizer for requirement understanding."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        self.config_manager = config_manager
        self.business_rules = config_manager.business_rules() if config_manager else {}

    def parse(
        self,
        request_text: str,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None = None,
    ) -> RequirementSpec:
        confirmed_context = conversation_context.latest_confirmed_requirement if conversation_context else {}
        output_formats = self._resolve_output_formats(request_text, confirmed_context)
        need_chart = "chart" in output_formats
        need_report = "markdown" in output_formats
        need_pdf = "pdf" in output_formats
        need_docx = "docx" in output_formats

        source_sheet = self._extract_source_sheet(request_text, workbook.sheet_names)
        available_columns = self._collect_available_columns(workbook, source_sheet)
        metric_groups = self._extract_metric_groups(request_text, available_columns)
        requested_statistics = self._extract_requested_statistics(request_text)
        metric_group_mode = self._extract_metric_group_mode(request_text, metric_groups)
        aggregate_by_fields = self._extract_aggregate_by_fields(request_text)
        requested_metrics = self._extract_metrics(
            request_text,
            workbook,
            target_sheet=source_sheet,
            available_columns=available_columns,
        )
        analysis_mode = self._extract_analysis_mode(
            request_text=request_text,
            metric_groups=metric_groups,
            metric_group_mode=metric_group_mode,
            requested_statistics=requested_statistics,
            aggregate_by_fields=aggregate_by_fields,
        )
        if analysis_mode in {"metric_group_summary", "metric_group_point_summary"} and metric_groups:
            requested_metrics = unique_preserve_order(list(metric_groups.keys()))
        template_hint = self._extract_template_hint(request_text)
        template_sheet_name = self._extract_template_sheet_name(request_text, workbook)
        stage_value = self._extract_stage(request_text)
        remark_value = self._extract_remark(request_text)
        decay_method = self._extract_decay_method(request_text)
        group_by = self._extract_group_by(
            request_text=request_text,
            requested_statistics=requested_statistics,
            aggregate_by_fields=aggregate_by_fields,
        )
        threshold_rules = self._extract_threshold_rules(request_text)

        spec = RequirementSpec(
            raw_request=request_text,
            intent_type=self._infer_intent_type(request_text, bool(confirmed_context)),
            requested_metrics=requested_metrics,
            metric_groups=metric_groups,
            metric_group_mode=metric_group_mode,
            source_sheet=source_sheet,
            output_formats=output_formats,
            output_sheet_name=self.business_rules.get("default_output_sheet_name", "KPI Summary"),
            template_sheet_name=template_sheet_name,
            stage_value=stage_value,
            remark_value=remark_value,
            template_hint=template_hint,
            need_chart=need_chart,
            need_report=need_report,
            need_pdf=need_pdf,
            need_docx=need_docx,
            decay_method=decay_method,
            analysis_mode=analysis_mode,
            aggregate_by_fields=aggregate_by_fields,
            requested_statistics=requested_statistics,
            group_by=group_by,
            threshold_rules=threshold_rules,
            parsed_notes=[],
        )
        spec = self._inherit_confirmed_context(spec, confirmed_context, workbook)
        spec = self._sanitize_requirement(spec, workbook, conversation_context)

        parsed_notes: list[str] = []
        if template_hint:
            parsed_notes.append(f"检测到参考模板：{template_hint}")
        if spec.stage_value:
            parsed_notes.append(f"检测到阶段值：{spec.stage_value}")
        if spec.group_by:
            parsed_notes.append(f"检测到分组线索：{', '.join(spec.group_by)}")
        if spec.analysis_mode in {"metric_group_summary", "grouped_summary"} and spec.aggregate_by_fields:
            parsed_notes.append(f"检测到分别统计维度：{', '.join(spec.aggregate_by_fields)}")
        if spec.intent_type != "new_request":
            parsed_notes.append(f"当前输入被识别为会话内的{spec.intent_type}。")

        spec.parsed_notes = unique_preserve_order(spec.parsed_notes + parsed_notes)
        spec.clarification_questions = self.build_clarifications(spec, workbook)
        return spec

    def compile_analysis_plan(
        self,
        request_text: str,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None = None,
        overrides: dict[str, object] | None = None,
    ) -> AnalysisPlan:
        spec = self.parse(request_text, workbook, conversation_context)
        plan = self.requirement_to_plan(spec)
        plan = self.finalize_analysis_plan(plan, workbook, conversation_context)
        if overrides:
            plan = self.apply_plan_overrides(plan, overrides)
            plan = self.finalize_analysis_plan(plan, workbook, conversation_context)
        return plan

    def apply_overrides(self, spec: RequirementSpec, overrides: dict[str, object]) -> RequirementSpec:
        updated = spec.copy(deep=True)
        if overrides.get("source_sheet"):
            updated.source_sheet = str(overrides["source_sheet"])
        if overrides.get("requested_metrics"):
            updated.requested_metrics = unique_preserve_order([str(item) for item in overrides["requested_metrics"] if str(item).strip()])
        if overrides.get("metric_groups"):
            updated.metric_groups = {
                str(key): [str(item) for item in value if str(item).strip()]
                for key, value in dict(overrides["metric_groups"]).items()
                if str(key).strip()
            }
        if overrides.get("metric_group_mode"):
            updated.metric_group_mode = str(overrides["metric_group_mode"]).strip() or updated.metric_group_mode
        if overrides.get("stage_value"):
            updated.stage_value = str(overrides["stage_value"]).strip() or None
        if overrides.get("remark_value"):
            updated.remark_value = str(overrides["remark_value"]).strip() or None
        if overrides.get("decay_method"):
            updated.decay_method = str(overrides["decay_method"]).strip() or None
        if overrides.get("analysis_mode"):
            updated.analysis_mode = str(overrides["analysis_mode"]).strip() or updated.analysis_mode
        if overrides.get("aggregate_by_fields"):
            updated.aggregate_by_fields = [str(item) for item in overrides["aggregate_by_fields"] if str(item).strip()]
        if overrides.get("requested_statistics"):
            updated.requested_statistics = [str(item) for item in overrides["requested_statistics"] if str(item).strip()]
        if overrides.get("group_by"):
            updated.group_by = [str(item) for item in overrides["group_by"] if str(item).strip()]
        if overrides.get("output_sheet_name"):
            updated.output_sheet_name = str(overrides["output_sheet_name"]).strip() or updated.output_sheet_name
        if overrides.get("template_sheet_name"):
            updated.template_sheet_name = str(overrides["template_sheet_name"]).strip() or None
        if "need_chart" in overrides:
            updated.need_chart = bool(overrides["need_chart"])
        if "need_report" in overrides:
            updated.need_report = bool(overrides["need_report"])
        if "need_pdf" in overrides:
            updated.need_pdf = bool(overrides["need_pdf"])
        if "need_docx" in overrides:
            updated.need_docx = bool(overrides["need_docx"])
        if overrides.get("threshold_rules"):
            updated.threshold_rules = overrides["threshold_rules"] or {}
        if overrides.get("revision_note"):
            updated.revision_note = str(overrides["revision_note"]).strip() or None
        updated.output_formats = unique_preserve_order(
            ["excel"]
            + (["chart"] if updated.need_chart else [])
            + (["markdown"] if updated.need_report else [])
            + (["pdf"] if updated.need_pdf else [])
            + (["docx"] if updated.need_docx else [])
        )
        updated.clarification_questions = self.build_clarifications(updated, None)
        return updated

    def apply_plan_overrides(self, plan: AnalysisPlan, overrides: dict[str, object]) -> AnalysisPlan:
        updated = plan.copy(deep=True)
        if overrides.get("source_sheet"):
            updated.source_sheet = str(overrides["source_sheet"]).strip()
        if overrides.get("metric_targets"):
            metric_targets: list[MetricTargetPlan] = []
            for item in overrides.get("metric_targets", []):
                if isinstance(item, MetricTargetPlan):
                    metric_targets.append(item.copy(deep=True))
                elif isinstance(item, dict):
                    metric_targets.append(
                        MetricTargetPlan(
                            label=str(item.get("label", "")).strip(),
                            target_type=str(item.get("target_type", "metric_group")).strip() or "metric_group",
                            family=str(item.get("family")).strip().upper() if item.get("family") else None,
                            start_index=int(item["start_index"]) if item.get("start_index") not in (None, "") else None,
                            end_index=int(item["end_index"]) if item.get("end_index") not in (None, "") else None,
                            columns=[str(column).strip() for column in item.get("columns", []) if str(column).strip()],
                            matched_column=str(item.get("matched_column")).strip() if item.get("matched_column") else None,
                        )
                    )
            if metric_targets:
                updated.metric_targets = metric_targets
        elif overrides.get("requested_metrics"):
            metric_targets: list[MetricTargetPlan] = []
            for raw_metric in overrides["requested_metrics"]:
                label = str(raw_metric).strip()
                if not label:
                    continue
                family, start, end = self._parse_metric_group_label(label)
                if family and start is not None and end is not None:
                    metric_targets.append(
                        MetricTargetPlan(
                            label=label,
                            target_type="metric_group",
                            family=family,
                            start_index=start,
                            end_index=end,
                        )
                    )
                else:
                    metric_targets.append(
                        MetricTargetPlan(
                            label=label,
                            target_type="metric_series",
                            matched_column=label,
                        )
                    )
            if metric_targets:
                updated.metric_targets = metric_targets
        if overrides.get("decay_method"):
            updated.decay_method = str(overrides["decay_method"]).strip() or None
        if overrides.get("aggregate_by_fields"):
            updated.group_dimensions = [str(item).strip() for item in overrides["aggregate_by_fields"] if str(item).strip()]
        if overrides.get("group_by"):
            updated.group_dimensions = [str(item).strip() for item in overrides["group_by"] if str(item).strip()]
        if overrides.get("requested_statistics"):
            updated.statistics = [str(item).strip() for item in overrides["requested_statistics"] if str(item).strip()]
        if overrides.get("output_granularity"):
            updated.output_granularity = str(overrides["output_granularity"]).strip() or updated.output_granularity
        if overrides.get("output_sheet_name"):
            updated.output_sheet_name = str(overrides["output_sheet_name"]).strip() or updated.output_sheet_name
        if overrides.get("template_sheet_name"):
            updated.template_sheet_name = str(overrides["template_sheet_name"]).strip() or None
        if overrides.get("template_hint"):
            updated.template_hint = str(overrides["template_hint"]).strip() or None
        if overrides.get("stage_value"):
            updated.stage_value = str(overrides["stage_value"]).strip() or None
        if overrides.get("remark_value"):
            updated.remark_value = str(overrides["remark_value"]).strip() or None
        if overrides.get("threshold_rules"):
            updated.threshold_rules = overrides["threshold_rules"] or {}
        if overrides.get("revision_note"):
            updated.revision_note = str(overrides["revision_note"]).strip() or None

        output_formats = list(updated.output_formats or [])
        if "need_chart" in overrides:
            want_chart = bool(overrides["need_chart"])
            output_formats = [item for item in output_formats if item != "chart"]
            if want_chart:
                output_formats.append("chart")
        if "need_report" in overrides:
            want_report = bool(overrides["need_report"])
            output_formats = [item for item in output_formats if item != "markdown"]
            if want_report:
                output_formats.append("markdown")
        if "need_pdf" in overrides:
            want_pdf = bool(overrides["need_pdf"])
            output_formats = [item for item in output_formats if item != "pdf"]
            if want_pdf:
                output_formats.append("pdf")
        if "need_docx" in overrides:
            want_docx = bool(overrides["need_docx"])
            output_formats = [item for item in output_formats if item != "docx"]
            if want_docx:
                output_formats.append("docx")
        updated.output_formats = self._normalize_plan_output_formats(output_formats)
        return updated

    def build_plan_clarifications(
        self,
        plan: AnalysisPlan,
        workbook: WorkbookContext | None,
    ) -> list[str]:
        questions: list[str] = []
        if not plan.metric_targets:
            questions.append("请补充要分析的指标组或测试项。")
        if not plan.source_sheet:
            if workbook and len(workbook.sheet_names) > 1:
                questions.append("当前文件包含多个 sheet，请明确指定要分析的 source sheet。")
        if plan.output_granularity in {"group_summary", "point_summary"} and not plan.group_dimensions:
            questions.append("当前是汇总输出，请明确分组维度，例如测试项目、SN、阶段或测试节点。")
        if plan.output_granularity in {"group_summary", "point_summary"} and not plan.statistics:
            questions.append("请明确要输出哪些统计项，当前支持：min、max、mean、decay。")
        if "decay" in plan.statistics and not plan.decay_method:
            questions.append("请明确衰减公式，当前首期支持：relative_to_first、delta_from_first、delta_from_previous。")
        if (plan.template_hint or (workbook and workbook.template_file)) and not plan.template_sheet_name:
            questions.append("检测到你提供了模板文件，请明确模板中要写入的目标 sheet 名称。")
        return unique_preserve_order(questions)

    def requirement_to_plan(self, spec: RequirementSpec) -> AnalysisPlan:
        metric_targets: list[MetricTargetPlan] = []
        seen_labels: set[str] = set()
        for label, columns in spec.metric_groups.items():
            family, start, end = self._parse_metric_group_label(label)
            metric_targets.append(
                MetricTargetPlan(
                    label=label,
                    target_type="metric_group",
                    family=family,
                    start_index=start,
                    end_index=end,
                    columns=[str(column).strip() for column in columns if str(column).strip()],
                )
            )
            seen_labels.add(label)

        for metric in spec.requested_metrics:
            label = str(metric).strip()
            if not label or label in seen_labels:
                continue
            family, start, end = self._parse_metric_group_label(label)
            if family and start is not None and end is not None:
                metric_targets.append(
                    MetricTargetPlan(
                        label=label,
                        target_type="metric_group",
                        family=family,
                        start_index=start,
                        end_index=end,
                    )
                )
            else:
                metric_targets.append(
                    MetricTargetPlan(
                        label=label,
                        target_type="metric_series",
                        matched_column=label,
                    )
                )
            seen_labels.add(label)

        return AnalysisPlan(
            raw_request=spec.raw_request,
            intent_type=spec.intent_type,
            source_sheet=spec.source_sheet,
            group_dimensions=list(spec.aggregate_by_fields or spec.group_by),
            metric_targets=metric_targets,
            statistics=list(spec.requested_statistics),
            decay_method=spec.decay_method,
            output_granularity=self._derive_output_granularity_from_requirement(spec),
            output_formats=list(spec.output_formats),
            output_sheet_name=spec.output_sheet_name,
            template_sheet_name=spec.template_sheet_name,
            template_hint=spec.template_hint,
            stage_value=spec.stage_value,
            remark_value=spec.remark_value,
            threshold_rules=dict(spec.threshold_rules),
            review_targets=self._extract_review_targets(spec.raw_request),
            llm_notes=list(spec.llm_notes),
            clarification_questions=list(spec.clarification_questions),
            parsed_notes=list(spec.parsed_notes),
            context_inheritance_notes=list(spec.context_inheritance_notes),
            revision_note=spec.revision_note,
        )

    def plan_to_requirement(self, plan: AnalysisPlan) -> RequirementSpec:
        metric_groups: dict[str, list[str]] = {}
        requested_metrics: list[str] = []
        for target in plan.metric_targets:
            requested_metrics.append(target.label)
            if target.target_type == "metric_group":
                metric_groups[target.label] = list(target.columns)

        output_formats = self._normalize_plan_output_formats(plan.output_formats)
        granularity = plan.output_granularity
        if granularity == "point_summary":
            analysis_mode = "metric_group_point_summary"
            metric_group_mode = "point_summary"
        elif metric_groups:
            analysis_mode = "metric_group_summary"
            metric_group_mode = "family_summary"
        elif plan.group_dimensions:
            analysis_mode = "grouped_summary"
            metric_group_mode = "family_summary"
        else:
            analysis_mode = "detail"
            metric_group_mode = "family_summary"

        return RequirementSpec(
            raw_request=plan.raw_request,
            intent_type=plan.intent_type,
            requested_metrics=requested_metrics,
            metric_groups=metric_groups,
            metric_group_mode=metric_group_mode,
            source_sheet=plan.source_sheet,
            output_formats=output_formats,
            output_sheet_name=plan.output_sheet_name,
            template_sheet_name=plan.template_sheet_name,
            stage_value=plan.stage_value,
            remark_value=plan.remark_value,
            template_hint=plan.template_hint,
            need_chart="chart" in output_formats,
            need_report="markdown" in output_formats,
            need_pdf="pdf" in output_formats,
            need_docx="docx" in output_formats,
            decay_method=plan.decay_method,
            analysis_mode=analysis_mode,
            aggregate_by_fields=list(plan.group_dimensions),
            requested_statistics=list(plan.statistics),
            group_by=list(plan.group_dimensions if analysis_mode == "detail" else []),
            threshold_rules=dict(plan.threshold_rules),
            llm_notes=list(plan.llm_notes),
            clarification_questions=list(plan.clarification_questions),
            parsed_notes=list(plan.parsed_notes),
            context_inheritance_notes=list(plan.context_inheritance_notes),
            revision_note=plan.revision_note,
        )

    def finalize_analysis_plan(
        self,
        plan: AnalysisPlan,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None = None,
    ) -> AnalysisPlan:
        inherited_payload = conversation_context.latest_confirmed_plan if conversation_context else {}
        inherited_plan = AnalysisPlan(raw_request=plan.raw_request)
        if isinstance(inherited_payload, dict) and inherited_payload:
            try:
                inherited_plan = AnalysisPlan(raw_request=plan.raw_request, **inherited_payload)
            except Exception:
                inherited_plan = AnalysisPlan(raw_request=plan.raw_request)

        updated = plan.copy(deep=True)
        updated.intent_type = self._canonicalize_intent_type(updated.intent_type, bool(inherited_payload))
        if not updated.source_sheet:
            updated.source_sheet = inherited_plan.source_sheet
        if len(workbook.sheet_names) == 1 and not updated.source_sheet:
            updated.source_sheet = workbook.default_sheet

        if not updated.group_dimensions:
            updated.group_dimensions = list(inherited_plan.group_dimensions)
        updated.group_dimensions = self._canonicalize_base_fields(updated.group_dimensions)

        if not updated.metric_targets:
            updated.metric_targets = list(inherited_plan.metric_targets)
        if not updated.statistics:
            updated.statistics = list(inherited_plan.statistics)
        updated.statistics = self._canonicalize_requested_statistics(updated.statistics)

        if not updated.decay_method:
            updated.decay_method = inherited_plan.decay_method
        if not updated.output_granularity or updated.output_granularity == "detail" and inherited_plan.output_granularity != "detail":
            updated.output_granularity = inherited_plan.output_granularity or updated.output_granularity
        if not updated.output_formats:
            updated.output_formats = list(inherited_plan.output_formats)
        if not updated.template_sheet_name:
            updated.template_sheet_name = inherited_plan.template_sheet_name
        if not updated.template_hint:
            updated.template_hint = inherited_plan.template_hint
        if not updated.stage_value:
            updated.stage_value = inherited_plan.stage_value
        if not updated.remark_value:
            updated.remark_value = inherited_plan.remark_value
        if not updated.threshold_rules:
            updated.threshold_rules = dict(inherited_plan.threshold_rules)
        if not updated.review_targets:
            updated.review_targets = self._extract_review_targets(updated.raw_request)

        available_columns = self._collect_available_columns(workbook, updated.source_sheet)
        updated.metric_targets = self._ground_metric_targets(updated.metric_targets, available_columns)
        updated.output_formats = self._normalize_plan_output_formats(updated.output_formats)
        if not updated.output_granularity:
            updated.output_granularity = self._derive_output_granularity_from_targets(updated.metric_targets, updated.group_dimensions)
        else:
            updated.output_granularity = self._canonicalize_output_granularity(
                updated.output_granularity,
                updated.metric_targets,
                updated.group_dimensions,
            )
        if updated.output_granularity == "group_summary" and not updated.group_dimensions:
            projected_requirement = self.plan_to_requirement(updated)
            if projected_requirement.aggregate_by_fields:
                updated.group_dimensions = list(projected_requirement.aggregate_by_fields)
        if updated.output_granularity in {"group_summary", "point_summary"} and not updated.statistics:
            projected_requirement = self.plan_to_requirement(updated)
            updated.statistics = self._canonicalize_requested_statistics(projected_requirement.requested_statistics)
        if not (updated.template_hint or workbook.template_file):
            updated.template_sheet_name = None
        projected_requirement = self.plan_to_requirement(updated)
        projected_requirement.parsed_notes = self._sanitize_parsed_notes(projected_requirement.parsed_notes, projected_requirement.metric_groups)
        updated.parsed_notes = list(projected_requirement.parsed_notes)
        updated.clarification_questions = self.build_plan_clarifications(updated, workbook)
        return updated

    def finalize_llm_requirement(
        self,
        spec: RequirementSpec,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None = None,
    ) -> RequirementSpec:
        confirmed_context = conversation_context.latest_confirmed_requirement if conversation_context else {}
        updated = spec.copy(deep=True)
        updated.intent_type = self._canonicalize_intent_type(updated.intent_type, bool(confirmed_context))
        updated = self._inherit_missing_context_for_llm(updated, confirmed_context, workbook)
        updated.requested_statistics = self._canonicalize_requested_statistics(updated.requested_statistics)

        if len(workbook.sheet_names) == 1 and not updated.source_sheet:
            updated.source_sheet = workbook.default_sheet

        available_columns = self._collect_available_columns(workbook, updated.source_sheet)
        updated.metric_groups = self._ground_metric_groups(updated.metric_groups, available_columns)
        updated.requested_metrics = self._ground_requested_metrics(
            requested_metrics=updated.requested_metrics,
            metric_groups=updated.metric_groups,
            available_columns=available_columns,
        )
        updated.aggregate_by_fields = self._canonicalize_base_fields(updated.aggregate_by_fields)
        updated.group_by = self._canonicalize_base_fields(updated.group_by)

        if updated.metric_group_mode not in {"family_summary", "point_summary"}:
            inherited_mode = confirmed_context.get("metric_group_mode")
            updated.metric_group_mode = inherited_mode if inherited_mode in {"family_summary", "point_summary"} else "family_summary"

        if updated.metric_groups and not updated.requested_metrics:
            updated.requested_metrics = unique_preserve_order(list(updated.metric_groups.keys()))

        updated.analysis_mode = self._canonicalize_analysis_mode(updated.analysis_mode, updated)

        if not (updated.template_hint or workbook.template_file):
            updated.template_sheet_name = None

        updated.output_formats = self._normalize_output_formats(updated)
        updated.need_chart = "chart" in updated.output_formats
        updated.need_report = "markdown" in updated.output_formats
        updated.need_pdf = "pdf" in updated.output_formats
        updated.need_docx = "docx" in updated.output_formats
        updated.parsed_notes = self._sanitize_parsed_notes(updated.parsed_notes, updated.metric_groups)
        updated.clarification_questions = self.build_clarifications(updated, workbook)
        return updated

    def build_clarifications(
        self,
        spec: RequirementSpec,
        workbook: WorkbookContext | None,
    ) -> list[str]:
        questions: list[str] = []
        if not spec.requested_metrics and not spec.metric_groups:
            questions.append("请补充要分析的测试项名称。")
        if not spec.source_sheet:
            if workbook and len(workbook.sheet_names) > 1:
                questions.append("当前文件包含多个 sheet，请明确指定要分析的 source sheet。")
            elif workbook and workbook.default_sheet:
                spec.source_sheet = workbook.default_sheet
        if self._requires_decay_method(spec) and not spec.decay_method:
            questions.append("请明确衰减公式，当前首期支持：relative_to_first、delta_from_first、delta_from_previous。")
        if spec.analysis_mode in {"metric_group_summary", "grouped_summary"} and not spec.aggregate_by_fields:
            questions.append("你希望按什么维度分别统计这些指标组，例如按测试项目、SN 或阶段？")
        if (spec.template_hint or (workbook and workbook.template_file)) and not spec.template_sheet_name:
            questions.append("检测到你提供了模板文件，请明确模板中要写入的目标 sheet 名称。")
        return self._sanitize_clarifications(spec, questions)

    def _resolve_output_formats(self, request_text: str, confirmed_context: dict[str, Any]) -> list[str]:
        default_formats = self.business_rules.get("default_report_formats", ["excel", "markdown", "pdf"])
        output_formats = list(default_formats) if default_formats else ["excel"]
        lowered = request_text.lower()
        if any(keyword in request_text for keyword in ["图", "图表", "曲线", "柱状图", "折线图"]) or "chart" in lowered:
            output_formats.append("chart")
        if any(keyword in request_text for keyword in ["报告", "总结", "汇报", "markdown"]) or "report" in lowered:
            output_formats.append("markdown")
        if any(keyword in request_text for keyword in ["pdf", "导出pdf", "输出pdf"]) or "pdf" in lowered:
            output_formats.append("pdf")
        if any(keyword in request_text for keyword in ["docx", "word"]) or "docx" in lowered or "word" in lowered:
            output_formats.append("docx")

        if confirmed_context:
            inherited_formats = confirmed_context.get("output_formats") or []
            output_formats.extend([str(item) for item in inherited_formats if str(item).strip()])

        return unique_preserve_order([fmt for fmt in output_formats if fmt])

    def _normalize_output_formats(self, spec: RequirementSpec) -> list[str]:
        output_formats = list(spec.output_formats or [])
        if spec.need_chart:
            output_formats.append("chart")
        if spec.need_report:
            output_formats.append("markdown")
        if spec.need_pdf:
            output_formats.append("pdf")
        if spec.need_docx:
            output_formats.append("docx")
        if not output_formats:
            output_formats = list(self.business_rules.get("default_report_formats", ["excel"]))
        return unique_preserve_order(["excel", *[fmt for fmt in output_formats if fmt and fmt != "excel"]])

    def _normalize_plan_output_formats(self, output_formats: list[str]) -> list[str]:
        formats = [str(item).strip() for item in output_formats or [] if str(item).strip()]
        if not formats:
            formats = list(self.business_rules.get("default_report_formats", ["excel"]))
        return unique_preserve_order(["excel", *[fmt for fmt in formats if fmt and fmt != "excel"]])

    def _derive_output_granularity_from_requirement(self, spec: RequirementSpec) -> str:
        if spec.analysis_mode == "metric_group_point_summary":
            return "point_summary"
        if spec.analysis_mode in {"metric_group_summary", "grouped_summary"}:
            return "group_summary"
        return "detail"

    def _derive_output_granularity_from_targets(
        self,
        metric_targets: list[MetricTargetPlan],
        group_dimensions: list[str],
    ) -> str:
        if any(target.target_type == "metric_group" for target in metric_targets):
            return "group_summary" if group_dimensions else "detail"
        if group_dimensions:
            return "group_summary"
        return "detail"

    def _canonicalize_output_granularity(
        self,
        output_granularity: str | None,
        metric_targets: list[MetricTargetPlan],
        group_dimensions: list[str],
    ) -> str:
        text = normalize_text(str(output_granularity or ""))
        mapping = {
            "groupsummary": "group_summary",
            "summary": "group_summary",
            "pointsummary": "point_summary",
            "detail": "detail",
        }
        if text in mapping:
            return mapping[text]
        return self._derive_output_granularity_from_targets(metric_targets, group_dimensions)

    def _requirement_dict_to_plan(self, payload: dict[str, Any], raw_request: str) -> AnalysisPlan:
        if not isinstance(payload, dict):
            return AnalysisPlan(raw_request=raw_request)
        try:
            spec = RequirementSpec(raw_request=raw_request, **payload)
        except Exception:
            spec = RequirementSpec(raw_request=raw_request)
            for field in spec.model_fields:
                if field == "raw_request":
                    continue
                value = payload.get(field)
                if value not in (None, ""):
                    setattr(spec, field, value)
        return self.requirement_to_plan(spec)

    def _inherit_missing_context_for_llm(
        self,
        spec: RequirementSpec,
        confirmed_context: dict[str, Any],
        workbook: WorkbookContext,
    ) -> RequirementSpec:
        updated = spec.copy(deep=True)
        if not confirmed_context:
            if len(workbook.sheet_names) == 1 and not updated.source_sheet:
                updated.source_sheet = workbook.default_sheet
            return updated

        for field_name in (
            "source_sheet",
            "stage_value",
            "remark_value",
            "template_hint",
            "template_sheet_name",
            "decay_method",
            "metric_group_mode",
            "analysis_mode",
            "threshold_rules",
        ):
            current_value = getattr(updated, field_name, None)
            inherited_value = confirmed_context.get(field_name)
            if current_value in (None, "", [], {}) and inherited_value not in (None, "", [], {}):
                if field_name == "source_sheet" and inherited_value not in workbook.sheet_names:
                    continue
                setattr(updated, field_name, inherited_value)

        for list_field in ("aggregate_by_fields", "requested_statistics", "group_by"):
            current_value = getattr(updated, list_field, None) or []
            inherited_value = confirmed_context.get(list_field) or []
            if not current_value and inherited_value:
                setattr(updated, list_field, [str(item).strip() for item in inherited_value if str(item).strip()])

        if not updated.requested_metrics and confirmed_context.get("requested_metrics"):
            updated.requested_metrics = [
                str(item).strip()
                for item in confirmed_context.get("requested_metrics", [])
                if str(item).strip()
            ]
        if not updated.metric_groups and confirmed_context.get("metric_groups"):
            updated.metric_groups = self._coerce_metric_groups(confirmed_context.get("metric_groups"))
        return updated

    def _inherit_confirmed_context(
        self,
        spec: RequirementSpec,
        confirmed_context: dict[str, Any],
        workbook: WorkbookContext,
    ) -> RequirementSpec:
        inherited_notes: list[str] = []
        if not confirmed_context:
            if len(workbook.sheet_names) == 1 and not spec.source_sheet:
                spec.source_sheet = workbook.default_sheet
            return spec

        inherited_metric_groups = self._coerce_metric_groups(confirmed_context.get("metric_groups"))
        if (
            spec.intent_type != "new_request"
            and spec.metric_groups
            and inherited_metric_groups
            and self._should_merge_metric_groups(spec.raw_request)
        ):
            spec.metric_groups = self._merge_metric_groups(inherited_metric_groups, spec.metric_groups)
            spec.requested_metrics = unique_preserve_order(
                [*confirmed_context.get("requested_metrics", []), *list(spec.metric_groups.keys()), *spec.requested_metrics]
            )
            inherited_notes.append("沿用并扩展已确认的字段家族")

        inherited_aggregate_fields = [
            str(item).strip()
            for item in confirmed_context.get("aggregate_by_fields", [])
            if str(item).strip()
        ]
        if (
            spec.intent_type != "new_request"
            and spec.aggregate_by_fields
            and inherited_aggregate_fields
            and self._should_merge_aggregate_fields(spec.raw_request)
        ):
            spec.aggregate_by_fields = unique_preserve_order([*inherited_aggregate_fields, *spec.aggregate_by_fields])
            inherited_notes.append("沿用并扩展已确认的分别统计维度")

        inheritable_fields = {
            "source_sheet": "source sheet",
            "requested_metrics": "测试项",
            "metric_groups": "字段家族",
            "metric_group_mode": "指标组输出模式",
            "stage_value": "阶段值",
            "remark_value": "备注",
            "template_hint": "模板提示",
            "template_sheet_name": "模板 sheet",
            "decay_method": "衰减公式",
            "analysis_mode": "分析模式",
            "aggregate_by_fields": "分别统计维度",
            "requested_statistics": "统计指标",
            "group_by": "分组维度",
            "threshold_rules": "阈值规则",
        }
        for field_name, label in inheritable_fields.items():
            current_value = getattr(spec, field_name, None)
            if field_name == "requested_statistics" and current_value not in (None, "", [], {}):
                inherited_value = confirmed_context.get(field_name)
                if (
                    spec.intent_type != "new_request"
                    and inherited_value not in (None, "", [], {})
                    and set(current_value).issubset(set(inherited_value))
                    and not self._has_explicit_statistic_override(spec.raw_request)
                ):
                    setattr(spec, field_name, unique_preserve_order([*inherited_value, *current_value]))
                    inherited_notes.append(f"沿用已确认的{label}")
                continue
            if current_value not in (None, "", [], {}):
                continue
            inherited_value = confirmed_context.get(field_name)
            if inherited_value in (None, "", [], {}):
                continue
            if field_name == "source_sheet" and inherited_value not in workbook.sheet_names:
                continue
            setattr(spec, field_name, inherited_value)
            inherited_notes.append(f"沿用已确认的{label}")

        if spec.output_sheet_name == "KPI Summary" and confirmed_context.get("output_sheet_name"):
            spec.output_sheet_name = str(confirmed_context["output_sheet_name"])
        if not spec.output_formats and confirmed_context.get("output_formats"):
            spec.output_formats = [str(item) for item in confirmed_context["output_formats"] if str(item).strip()]

        spec.context_inheritance_notes = unique_preserve_order(spec.context_inheritance_notes + inherited_notes)
        spec.need_chart = "chart" in spec.output_formats
        spec.need_report = "markdown" in spec.output_formats
        spec.need_pdf = "pdf" in spec.output_formats
        spec.need_docx = "docx" in spec.output_formats

        if len(workbook.sheet_names) == 1 and not spec.source_sheet:
            spec.source_sheet = workbook.default_sheet

        return spec

    def _infer_intent_type(self, request_text: str, has_confirmed_context: bool) -> str:
        if not has_confirmed_context:
            return "new_request"

        lowered = request_text.lower()
        if any(keyword in request_text for keyword in ["不对", "有误", "重算", "改成", "修正", "修订", "补充", "继续", "沿用", "还是"]):
            return "revision"
        if any(keyword in request_text for keyword in ["不是", "不要", "别", "取消"]) or "instead" in lowered:
            return "correction"
        if any(keyword in request_text for keyword in ["再补", "另外", "顺便", "继续补"]) or "also" in lowered:
            return "supplement"
        return "follow_up"

    def _extract_source_sheet(self, request_text: str, sheet_names: Iterable[str]) -> str | None:
        normalized_lookup = {normalize_text(sheet_name): sheet_name for sheet_name in sheet_names if sheet_name}
        match = re.search(r"([A-Za-z0-9_\u4e00-\u9fff-]+)\s*(?:这个|该)?\s*(?:sheet|工作表)", request_text, re.IGNORECASE)
        if match:
            candidate = normalize_text(match.group(1))
            if candidate in normalized_lookup:
                return normalized_lookup[candidate]
            for normalized_sheet, sheet_name in sorted(normalized_lookup.items(), key=lambda item: len(item[0]), reverse=True):
                if normalized_sheet and normalized_sheet in candidate:
                    return sheet_name

        direct_match = re.search(r"(?:从|在|使用|读取)?\s*([A-Za-z0-9_\u4e00-\u9fff-]+)\s*(?:中|里|内)\s*(?:提取|读取|定位|统计|分析)?", request_text, re.IGNORECASE)
        if direct_match:
            candidate = normalize_text(direct_match.group(1))
            if candidate in normalized_lookup:
                return normalized_lookup[candidate]

        pattern = re.search(r"(?:来自|从|使用)\s*[“\"']?([^”\"'\n]+?)(?:[”\"']?\s*(?:sheet|工作表))", request_text, re.IGNORECASE)
        if pattern:
            candidate = pattern.group(1).strip()
            normalized_candidate = normalize_text(candidate)
            best_match = None
            best_length = 0
            for sheet_name in sheet_names:
                normalized_sheet = normalize_text(sheet_name)
                if normalized_sheet == normalized_candidate:
                    return sheet_name
                if normalized_sheet and normalized_sheet in normalized_candidate and len(normalized_sheet) > best_length:
                    best_match = sheet_name
                    best_length = len(normalized_sheet)
            if best_match:
                return best_match

        for sheet_name in sheet_names:
            if not sheet_name:
                continue
            if re.search(rf"(?<![A-Za-z0-9_\u4e00-\u9fff]){re.escape(sheet_name)}(?![A-Za-z0-9_\u4e00-\u9fff])", request_text):
                return sheet_name
        normalized_request = normalize_text(request_text)
        for normalized_sheet, sheet_name in sorted(normalized_lookup.items(), key=lambda item: len(item[0]), reverse=True):
            if len(normalized_sheet) < 3 and not re.search(r"[\u4e00-\u9fff]", sheet_name):
                continue
            if normalized_sheet and normalized_sheet in normalized_request:
                return sheet_name
        return None

    def _extract_metrics(
        self,
        request_text: str,
        workbook: WorkbookContext,
        target_sheet: str | None = None,
        available_columns: list[str] | None = None,
    ) -> list[str]:
        metrics: list[str] = []
        normalized_request = normalize_text(request_text)
        sheet_names = {sheet_name.lower() for sheet_name in workbook.sheet_names}

        available_columns = available_columns or self._collect_available_columns(workbook, target_sheet)

        for column in available_columns:
            normalized_column = normalize_text(column)
            if not normalized_column or normalized_column in BASE_FIELD_KEYWORDS:
                continue
            if self._is_explicit_metric_mention(request_text, normalized_request, column, normalized_column):
                metrics.append(column)

        regex_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_./-]{3,}\b", request_text)
        for token in regex_tokens:
            normalized_token = normalize_text(token)
            if normalized_token in BASE_FIELD_KEYWORDS:
                continue
            if token.lower() in DECAY_KEYWORDS:
                continue
            if token.lower() in sheet_names:
                continue
            if token.lower().endswith((".xlsx", ".xls", ".csv", ".md")):
                continue
            if any(char == "_" for char in token) or any(char.isdigit() for char in token):
                metrics.append(token)

        quoted = re.findall(r"[`“\"']([^`”\"']{2,})[`”\"']", request_text)
        for token in quoted:
            if (
                normalize_text(token) not in BASE_FIELD_KEYWORDS
                and token.lower() not in sheet_names
                and not token.lower().endswith((".xlsx", ".xls", ".csv", ".md"))
            ):
                metrics.append(token)

        return unique_preserve_order([metric.strip() for metric in metrics if metric.strip()])

    def _collect_available_columns(self, workbook: WorkbookContext, target_sheet: str | None) -> list[str]:
        available_columns: list[str] = []
        candidate_summaries = list(workbook.sheet_summaries)
        if workbook.input_files:
            for file_context in workbook.input_files:
                if file_context.role != "data":
                    continue
                candidate_summaries.extend(file_context.sheet_summaries)

        if target_sheet:
            for summary in candidate_summaries:
                if summary.name == target_sheet:
                    available_columns.extend([column for column in summary.column_names if column])
        else:
            for summary in candidate_summaries:
                available_columns.extend([column for column in summary.column_names if column])
        return unique_preserve_order(available_columns)

    def _extract_metric_groups(self, request_text: str, available_columns: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        normalized_text = self._normalize_metric_range_mentions(request_text)
        available_by_prefix = self._collect_metric_family_columns(available_columns)

        range_pattern = re.compile(
            r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*)_?(\d+)\s*(?:~|～|-|到|至)\s*(?:([A-Za-z][A-Za-z0-9]*)_?)?(\d+)(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        for match in range_pattern.finditer(normalized_text):
            prefix = match.group(1).upper()
            end_prefix = (match.group(3) or prefix).upper()
            if prefix != end_prefix:
                continue
            start = int(match.group(2))
            end = int(match.group(4))
            label, columns = self._build_metric_group_selection(prefix, start, end, available_by_prefix)
            if columns:
                groups[label] = columns

        if not groups:
            adjacent_pattern = re.compile(
                r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*)_?(\d+)\s*(?:([A-Za-z][A-Za-z0-9]*)_?)(\d+)(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            for match in adjacent_pattern.finditer(normalized_text):
                prefix = match.group(1).upper()
                end_prefix = match.group(3).upper()
                if prefix != end_prefix:
                    continue
                label, columns = self._build_metric_group_selection(prefix, int(match.group(2)), int(match.group(4)), available_by_prefix)
                if columns:
                    groups[label] = columns

        mentioned_families = self._extract_metric_family_mentions(normalized_text)
        for family in mentioned_families:
            if any(label.startswith(f"{family}[") or label == family for label in groups):
                continue
            family_columns = available_by_prefix.get(family, [])
            if family_columns:
                label = self._build_metric_group_label(family, family_columns)
                groups[label] = family_columns

        return groups

    def _extract_metric_group_mode(self, request_text: str, metric_groups: dict[str, list[str]]) -> str:
        if not metric_groups:
            return "family_summary"

        lowered = request_text.lower()
        if any(keyword in request_text for keyword in ["所有测点", "所有点位", "每个测点", "每个点位", "点位明细", "测点明细"]):
            return "point_summary"
        if any(keyword in request_text for keyword in ["逐点", "按点位", "按测点", "逐个点位", "逐个测点"]):
            return "point_summary"
        if "明细" in request_text or "detail" in lowered:
            return "point_summary"
        return "family_summary"

    def _extract_requested_statistics(self, request_text: str) -> list[str]:
        lowered = request_text.lower()
        results: list[str] = []
        for statistic, keywords in STATISTIC_KEYWORDS.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                results.append(statistic)
        return results

    def _extract_analysis_mode(
        self,
        *,
        request_text: str,
        metric_groups: dict[str, list[str]],
        metric_group_mode: str,
        requested_statistics: list[str],
        aggregate_by_fields: list[str],
    ) -> str:
        lowered = request_text.lower()
        has_summary_cue = any(keyword in request_text for keyword in ["分别统计", "汇总", "统计", "分组", "分别输出"])
        has_summary_cue = has_summary_cue or any(keyword in lowered for keyword in ["summary", "aggregate", "group by"])

        if metric_groups:
            if metric_group_mode == "point_summary" and (requested_statistics or aggregate_by_fields or has_summary_cue):
                return "metric_group_point_summary"
            if requested_statistics or aggregate_by_fields or has_summary_cue:
                return "metric_group_summary"
            return "detail"

        if aggregate_by_fields and (requested_statistics or has_summary_cue):
            return "grouped_summary"
        return "detail"

    def _extract_aggregate_by_fields(self, request_text: str) -> list[str]:
        fields: list[str] = []
        if self._has_dimension_cue(request_text, BASE_FIELD_ALIASES["test_item"]):
            fields.append("test_item")
        if self._has_dimension_cue(request_text, BASE_FIELD_ALIASES["sn"]):
            fields.append("sn")
        if self._has_dimension_cue(request_text, BASE_FIELD_ALIASES["stage"]):
            fields.append("stage")
        if self._has_dimension_cue(request_text, BASE_FIELD_ALIASES["test_node"]):
            fields.append("test_node")
        if self._has_dimension_cue(request_text, BASE_FIELD_ALIASES["source_file"]):
            fields.append("source_file")
        return unique_preserve_order(fields)

    def _requires_decay_method(self, spec: RequirementSpec) -> bool:
        if spec.decay_method:
            return False
        if "decay" in spec.requested_statistics:
            return True
        lowered = spec.raw_request.lower()
        return any(keyword.lower() in lowered for values in DECAY_KEYWORDS.values() for keyword in values)

    def _is_explicit_metric_mention(
        self,
        request_text: str,
        normalized_request: str,
        column: str,
        normalized_column: str,
    ) -> bool:
        if not normalized_column:
            return False
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])",
            request_text,
        ):
            return True
        if "_" not in column and column in request_text:
            return True
        return normalized_column in normalized_request and "_" not in column

    def _extract_template_hint(self, request_text: str) -> str | None:
        matches = re.findall(r"([A-Za-z0-9_.-]+\.xlsx)", request_text)
        if matches:
            for candidate in matches:
                if "output" in candidate.lower():
                    return candidate
            return matches[-1]
        return None

    def _extract_template_sheet_name(self, request_text: str, workbook: WorkbookContext | None = None) -> str | None:
        has_template_context = bool(self._extract_template_hint(request_text) or (workbook and workbook.template_file))
        if not has_template_context:
            return None
        match = re.search(r"([A-Za-z0-9 _-]+)\s+(?:这一个sheet|sheet)", request_text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower().endswith(".xlsx"):
                return None
            return candidate
        if workbook:
            template_sheets = [
                summary.name
                for file_context in workbook.input_files
                if file_context.role == "template"
                for summary in file_context.sheet_summaries
            ]
            default_sheet_name = self.business_rules.get("template", {}).get("default_sheet_name")
            if default_sheet_name in template_sheets:
                return default_sheet_name
            if len(template_sheets) == 1:
                return template_sheets[0]
        return self.business_rules.get("template", {}).get("default_sheet_name")

    def _extract_stage(self, request_text: str) -> str | None:
        match = re.search(
            r"(?:阶段(?:先按|按|填充为|设置为|为|是)?|phase(?: is| to)?)"
            r"[：:\s]*\b(EVT|DVT|PVT|MP)\b",
            request_text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        return None

    def _extract_remark(self, request_text: str) -> str | None:
        match = re.search(r"(?:remark|备注)[:：]\s*([^\n]+)", request_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_decay_method(self, request_text: str) -> str | None:
        lowered = request_text.lower()
        if "delta_from_previous" in lowered:
            return "delta_from_previous"
        if "delta_from_first" in lowered:
            return "delta_from_first"
        if "relative_to_first" in lowered:
            return "relative_to_first"

        priority_order = ["delta_from_previous", "delta_from_first", "relative_to_first"]
        for method in priority_order:
            keywords = DECAY_KEYWORDS[method]
            explicit_keywords = [keyword for keyword in keywords if keyword not in {"衰减"}]
            if any(keyword.lower() in lowered for keyword in explicit_keywords):
                return method

        for method, keywords in DECAY_KEYWORDS.items():
            if any(keyword.lower() in lowered for keyword in keywords if keyword != "衰减"):
                return method
        return None

    def _extract_group_by(
        self,
        *,
        request_text: str,
        requested_statistics: list[str],
        aggregate_by_fields: list[str],
    ) -> list[str]:
        candidates = []
        lowered = request_text.lower()
        if aggregate_by_fields:
            return []
        if re.search(r"(相对|针对|基于)\s*每个\s*sn", lowered) or re.search(r"每个\s*sn\s*首条", lowered) or "按sn" in lowered:
            candidates.append("sn")
        if self._has_decay_grouping_cue(request_text, BASE_FIELD_ALIASES["test_node"]):
            candidates.append("test_node")
        if self._has_decay_grouping_cue(request_text, BASE_FIELD_ALIASES["stage"]):
            candidates.append("stage")
        if self._has_decay_grouping_cue(request_text, BASE_FIELD_ALIASES["test_item"]):
            candidates.append("test_item")
        if not candidates and "decay" in requested_statistics and "sn" in lowered:
            candidates.append("sn")
        return unique_preserve_order(candidates)

    def _extract_threshold_rules(self, request_text: str) -> dict[str, float | str]:
        rules: dict[str, float | str] = {}
        matches = re.findall(r"([A-Za-z0-9_./-]+)\s*(?:>=|<=|>|<|=)\s*([0-9.]+)", request_text)
        for metric, value in matches:
            rules[metric] = value
        return rules

    def _sanitize_requirement(
        self,
        spec: RequirementSpec,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None,
    ) -> RequirementSpec:
        message = conversation_context.current_message if conversation_context else spec.raw_request
        normalized_message = normalize_text(message)

        if len(workbook.sheet_names) == 1 and not spec.source_sheet:
            spec.source_sheet = workbook.default_sheet

        spec.aggregate_by_fields = self._canonicalize_base_fields(spec.aggregate_by_fields)
        spec.group_by = self._canonicalize_base_fields(spec.group_by)
        if not (spec.template_hint or workbook.template_file):
            spec.template_sheet_name = None

        if spec.metric_groups and spec.metric_group_mode == "point_summary":
            spec.analysis_mode = "metric_group_point_summary"
        elif spec.metric_groups:
            if spec.requested_statistics or spec.aggregate_by_fields:
                spec.analysis_mode = "metric_group_summary"
            elif spec.analysis_mode not in {"metric_group_summary", "metric_group_point_summary"}:
                spec.analysis_mode = "detail"

        if (
            spec.metric_groups
            and conversation_context
            and conversation_context.latest_confirmed_requirement.get("metric_group_mode") == "point_summary"
            and spec.metric_group_mode == "family_summary"
            and not self._has_explicit_family_summary_cue(message)
        ):
            spec.metric_group_mode = "point_summary"
            spec.analysis_mode = "metric_group_point_summary"
            spec.context_inheritance_notes = unique_preserve_order(
                spec.context_inheritance_notes + ["沿用已确认的点位级指标组输出模式"]
            )

        if "不需要包含子光学" in message or "不要包含子光学" in message or "只要处理我提及的数据列" in message:
            spec.parsed_notes = unique_preserve_order(spec.parsed_notes + ["已排除 SubOptic 子光学列。"])

        if spec.metric_groups:
            filtered_groups: dict[str, list[str]] = {}
            for family, columns in spec.metric_groups.items():
                filtered_groups[family] = [
                    column
                    for column in columns
                    if "suboptic" not in normalize_text(column)
                ]
            spec.metric_groups = filtered_groups

        if spec.metric_groups and not spec.requested_metrics:
            spec.requested_metrics = unique_preserve_order(list(spec.metric_groups.keys()))

        if spec.metric_groups and "suboptic" not in normalized_message:
            spec.parsed_notes = unique_preserve_order(spec.parsed_notes + ["对明确指定的 MTFH/MTFV 点位默认不包含 SubOptic 系列。"])

        spec.parsed_notes = self._sanitize_parsed_notes(spec.parsed_notes, spec.metric_groups)
        return spec

    def looks_like_metric_series_request(self, requested_metrics: list[str]) -> bool:
        families: dict[str, set[int]] = {}
        for metric in requested_metrics or []:
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)_(\d+)", str(metric).strip(), re.IGNORECASE)
            if not match:
                continue
            family = match.group(1).upper()
            families.setdefault(family, set()).add(int(match.group(2)))
        return any(len(indexes) >= 3 for indexes in families.values())

    def _ground_metric_targets(
        self,
        metric_targets: list[MetricTargetPlan],
        available_columns: list[str],
    ) -> list[MetricTargetPlan]:
        available_by_family = self._collect_metric_family_columns(available_columns)
        available_column_set = {str(column).strip() for column in available_columns if str(column).strip()}
        grounded: list[MetricTargetPlan] = []
        for target in metric_targets or []:
            current = target.copy(deep=True)
            if current.target_type == "metric_group":
                explicit_columns = [column for column in current.columns if column in available_column_set]
                family = current.family
                start = current.start_index
                end = current.end_index
                if not family:
                    family, start, end = self._parse_metric_group_label(current.label)
                if explicit_columns:
                    current.columns = unique_preserve_order(explicit_columns)
                    if not family:
                        family, start, end = self._parse_metric_group_label(current.label)
                    current.family = family
                    current.start_index = start
                    current.end_index = end
                elif family and start is not None and end is not None:
                    _, columns = self._build_metric_group_selection(family, start, end, available_by_family)
                    current.columns = columns
                    current.family = family
                    current.start_index = start
                    current.end_index = end
                elif family:
                    current.columns = available_by_family.get(family, [])
                    current.family = family
                else:
                    continue
                if current.columns:
                    current.label = self._build_metric_group_label(current.family or family or current.label, current.columns, current.start_index, current.end_index)
                    grounded.append(current)
                continue

            if current.matched_column and current.matched_column in available_column_set:
                grounded.append(current)
                continue
            if current.label in available_column_set:
                current.matched_column = current.label
                grounded.append(current)
                continue
            normalized_label = normalize_text(current.label)
            matched = next((column for column in available_columns if normalize_text(column) == normalized_label), None)
            if matched:
                current.matched_column = matched
                grounded.append(current)
        return grounded

    def _ground_metric_groups(
        self,
        metric_groups: dict[str, list[str]],
        available_columns: list[str],
    ) -> dict[str, list[str]]:
        grounded: dict[str, list[str]] = {}
        if not metric_groups:
            return grounded

        available_by_family = self._collect_metric_family_columns(available_columns)
        available_column_set = {str(column).strip() for column in available_columns if str(column).strip()}
        for raw_label, raw_columns in self._coerce_metric_groups(metric_groups).items():
            explicit_columns = [column for column in raw_columns if column in available_column_set]
            if explicit_columns:
                grounded[raw_label] = unique_preserve_order(explicit_columns)
                continue

            family, start, end = self._parse_metric_group_label(raw_label)
            if not family:
                continue
            if start is not None and end is not None:
                label, columns = self._build_metric_group_selection(family, start, end, available_by_family)
            else:
                columns = available_by_family.get(family, [])
                label = self._build_metric_group_label(family, columns)
            if columns:
                grounded[label] = columns
        return grounded

    def _ground_requested_metrics(
        self,
        *,
        requested_metrics: list[str],
        metric_groups: dict[str, list[str]],
        available_columns: list[str],
    ) -> list[str]:
        if metric_groups:
            return unique_preserve_order(list(metric_groups.keys()))

        available_column_set = {str(column).strip() for column in available_columns if str(column).strip()}
        metrics: list[str] = []
        for metric in requested_metrics or []:
            text = str(metric).strip()
            if not text:
                continue
            if text in metrics:
                continue
            family, _, _ = self._parse_metric_group_label(text)
            if family:
                metrics.append(text)
                continue
            if text in available_column_set:
                metrics.append(text)
        return unique_preserve_order(metrics)

    def _derive_analysis_mode_from_structure(self, spec: RequirementSpec) -> str:
        if spec.metric_groups:
            if spec.metric_group_mode == "point_summary" and (spec.requested_statistics or spec.aggregate_by_fields):
                return "metric_group_point_summary"
            if spec.requested_statistics or spec.aggregate_by_fields:
                return "metric_group_summary"
            return "detail"
        if spec.aggregate_by_fields and spec.requested_statistics:
            return "grouped_summary"
        return "detail"

    def _canonicalize_intent_type(self, intent_type: str | None, has_confirmed_context: bool) -> str:
        text = str(intent_type or "").strip().lower()
        if text in {"new_request", "follow_up", "revision", "supplement", "correction"}:
            return text
        if text in {"new", "create"}:
            return "new_request"
        if text in {"revise", "update", "modify"}:
            return "revision"
        if text in {"followup", "follow-up"}:
            return "follow_up"
        return "follow_up" if has_confirmed_context else "new_request"

    def _canonicalize_requested_statistics(self, requested_statistics: list[str]) -> list[str]:
        mapping = {
            "minimum": "min",
            "lowest": "min",
            "min": "min",
            "maximum": "max",
            "highest": "max",
            "max": "max",
            "average": "mean",
            "avg": "mean",
            "mean": "mean",
            "decay": "decay",
            "decay_rate": "decay",
            "decayrate": "decay",
            "decayratio": "decay",
            "rate_of_change": "decay",
            "change_rate": "decay",
            "relative_to_first": "decay",
        }
        normalized: list[str] = []
        for statistic in requested_statistics or []:
            key = normalize_text(str(statistic))
            normalized.append(mapping.get(key, str(statistic).strip()))
        return unique_preserve_order([item for item in normalized if item])

    def _canonicalize_analysis_mode(self, analysis_mode: str | None, spec: RequirementSpec) -> str:
        text = normalize_text(str(analysis_mode or ""))
        mapping = {
            "metricgroupsummary": "metric_group_summary",
            "metricgroupaggregation": "metric_group_summary",
            "groupedsummary": "grouped_summary",
            "groupedaggregation": "grouped_summary",
            "aggregation": "grouped_summary",
            "summary": "grouped_summary",
            "detail": "detail",
            "metricgrouppointsummary": "metric_group_point_summary",
        }
        canonical = mapping.get(text)
        if canonical:
            return canonical
        return self._derive_analysis_mode_from_structure(spec)

    def _parse_metric_group_label(self, label: str) -> tuple[str | None, int | None, int | None]:
        text = str(label).strip()
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)(?:\[(\d+)-(\d+)\])?", text, re.IGNORECASE)
        if not match:
            return None, None, None
        family = match.group(1).upper()
        start = int(match.group(2)) if match.group(2) else None
        end = int(match.group(3)) if match.group(3) else None
        return family, start, end

    def _extract_review_targets(self, request_text: str) -> list[ReviewTargetPlan]:
        targets: list[ReviewTargetPlan] = []
        challenge_keywords = ["不对", "有误", "检查", "核对", "复核", "仔细修改"]
        if not any(keyword in request_text for keyword in challenge_keywords):
            return []

        challenge_text = request_text
        for keyword in challenge_keywords:
            if keyword in request_text:
                challenge_text = request_text.split(keyword, 1)[1]
                break

        sn_matches = re.findall(r"(?:SN(?:号)?(?:为|是)?[:：]?\s*)([A-Za-z0-9-]{8,})", challenge_text, re.IGNORECASE)
        for value in sn_matches:
            targets.append(ReviewTargetPlan(dimension="sn", value=value, reason="user_challenge"))
        node_matches = re.findall(r"(?:测试节点|节点)(?:为|是)?[:：]?\s*([A-Za-z0-9._-]{1,32})", challenge_text, re.IGNORECASE)
        for value in node_matches:
            targets.append(ReviewTargetPlan(dimension="test_node", value=value, reason="user_challenge"))
        item_matches = re.findall(r"(?:测试项目|测试项)(?:为|是)[:：]?\s*([^\s，。,；;]+)", challenge_text, re.IGNORECASE)
        for value in item_matches:
            targets.append(ReviewTargetPlan(dimension="test_item", value=value, reason="user_challenge"))
        return targets

    def _normalize_metric_range_mentions(self, request_text: str) -> str:
        normalized = request_text
        normalized = re.sub(
            r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*)_?(\d+)\s*(\1)_?(\d+)(?![A-Za-z0-9])",
            r"\1_\2~\3_\4",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"([A-Za-z])(\d)", r"\1_\2", normalized)
        normalized = re.sub(r"(\d)([A-Za-z])", r"\1 \2", normalized)
        return normalized

    def _collect_metric_family_columns(self, available_columns: list[str]) -> dict[str, list[str]]:
        families: dict[str, list[str]] = {}
        for column in available_columns:
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)_(\d+)", str(column).strip(), re.IGNORECASE)
            if not match:
                continue
            family = match.group(1).upper()
            families.setdefault(family, []).append(column)
        for family, columns in list(families.items()):
            families[family] = sorted(columns, key=self._metric_column_sort_key)
        return families

    def _extract_metric_family_mentions(self, request_text: str) -> list[str]:
        families: list[str] = []
        for family in ("MTFH", "MTFV", "SMTF", "TMTF", "CTFH", "CTFV"):
            if re.search(rf"\b{re.escape(family)}\b", request_text, re.IGNORECASE) or family.lower() in request_text.lower():
                families.append(family)
        return unique_preserve_order(families)

    def _build_metric_group_selection(
        self,
        family: str,
        start: int,
        end: int,
        available_by_prefix: dict[str, list[str]],
    ) -> tuple[str, list[str]]:
        if end < start:
            start, end = end, start
        columns = [
            column
            for column in available_by_prefix.get(family, [])
            if start <= self._metric_column_index(column) <= end
        ]
        return self._build_metric_group_label(family, columns, start=start, end=end), columns

    def _build_metric_group_label(
        self,
        family: str,
        columns: list[str],
        start: int | None = None,
        end: int | None = None,
    ) -> str:
        if start is None or end is None:
            indexes = [self._metric_column_index(column) for column in columns if self._metric_column_index(column) >= 0]
            if indexes:
                start = min(indexes)
                end = max(indexes)
        if start is not None and end is not None:
            return f"{family}[{start}-{end}]"
        return family

    def _metric_column_index(self, column: str) -> int:
        match = re.search(r"_(\d+)$", str(column).strip())
        return int(match.group(1)) if match else -1

    def _metric_column_sort_key(self, column: str) -> tuple[str, int]:
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)_(\d+)", str(column).strip(), re.IGNORECASE)
        if not match:
            return (str(column), -1)
        return (match.group(1).upper(), int(match.group(2)))

    def _coerce_metric_groups(self, value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key).strip(): [str(item).strip() for item in items if str(item).strip()]
            for key, items in value.items()
            if str(key).strip()
        }

    def _merge_metric_groups(
        self,
        inherited_groups: dict[str, list[str]],
        current_groups: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {key: list(value) for key, value in inherited_groups.items()}
        for key, value in current_groups.items():
            if key in merged:
                merged[key] = unique_preserve_order([*merged[key], *value])
            else:
                merged[key] = list(value)
        return merged

    def _should_merge_metric_groups(self, request_text: str) -> bool:
        lowered = request_text.lower()
        reset_keywords = ["只保留", "仅保留", "只要", "仅要", "移除", "删除", "去掉", "不要", "改成只"]
        if any(keyword in request_text or keyword in lowered for keyword in reset_keywords):
            return False
        additive_keywords = ["还有", "再加", "加上", "补充", "另外", "以及", "并且"]
        return any(keyword in request_text or keyword in lowered for keyword in additive_keywords) or True

    def _should_merge_aggregate_fields(self, request_text: str) -> bool:
        lowered = request_text.lower()
        return not any(keyword in request_text or keyword in lowered for keyword in ["只按", "仅按", "改成按", "只保留"])

    def _has_dimension_cue(self, request_text: str, aliases: list[str]) -> bool:
        for alias in aliases:
            if re.search(rf"(每个|按)\s*{re.escape(alias)}\s*(?:首条|首行|初始)", request_text, re.IGNORECASE):
                continue
            token = re.escape(alias)
            patterns = [
                rf"每个\s*{token}",
                rf"按\s*{token}",
                rf"分\s*{token}",
                rf"各\s*{token}",
                rf"(?:加上|加一个|增加|带上|附上|包含|加上它的)\s*{token}",
                rf"{token}\s*(?:也要|也加上|也带上|一起带上)",
                rf"每个\s*{token}.{{0,20}}(?:最小值|最大值|平均值|衰减|变化率|统计|汇总|比较|对比)",
                rf"{token}\s*(?:分别统计|分组统计|维度|汇总|统计|对比|比较)",
                rf"按\s*[^，。,\n]*\b{token}\b",
                rf"\b{token}\b\s*(?:和|与|及|以及)\s*[^，。,\n]*(?:输出|汇总|统计|分组)",
            ]
            if any(re.search(pattern, request_text, re.IGNORECASE) for pattern in patterns):
                return True
        return False

    def _has_decay_grouping_cue(self, request_text: str, aliases: list[str]) -> bool:
        lowered = request_text.lower()
        if "首条记录" not in request_text and "首行" not in request_text and "相对初始" not in request_text and "relative_to_first" not in lowered:
            return False
        return any(re.search(rf"(每个|按)\s*{re.escape(alias)}", request_text, re.IGNORECASE) for alias in aliases)

    def _has_explicit_statistic_override(self, request_text: str) -> bool:
        lowered = request_text.lower()
        return any(
            keyword in request_text or keyword in lowered
            for keyword in ["只看", "只保留", "仅看", "仅保留", "只统计", "仅统计", "只要衰减", "只输出衰减"]
        )

    def _canonicalize_base_fields(self, fields: list[str]) -> list[str]:
        if not fields:
            return []

        alias_lookup: dict[str, str] = {}
        for canonical, aliases in BASE_FIELD_ALIASES.items():
            alias_lookup[normalize_text(canonical)] = canonical
            for alias in aliases:
                alias_lookup[normalize_text(alias)] = canonical

        normalized_fields: list[str] = []
        for field in fields:
            text = str(field).strip()
            if not text:
                continue
            normalized_fields.append(alias_lookup.get(normalize_text(text), text))
        return unique_preserve_order(normalized_fields)

    def _sanitize_clarifications(self, spec: RequirementSpec, questions: list[str]) -> list[str]:
        filtered: list[str] = []
        for question in questions:
            lowered = question.lower()
            if spec.source_sheet and ("sheet" in lowered or "工作表" in question):
                continue
            if spec.metric_groups and any(token in lowered for token in ["suboptic", "子光学"]):
                continue
            if spec.metric_groups and spec.metric_group_mode == "point_summary" and any(
                token in question for token in ["测点", "点位", "明细", "平均值", "最小值", "最大值"]
            ):
                continue
            filtered.append(question)
        return unique_preserve_order(filtered)

    def _sanitize_parsed_notes(self, notes: list[str], metric_groups: dict[str, list[str]]) -> list[str]:
        if not notes:
            return []

        filtered: list[str] = []
        normalized_families = {normalize_text(name) for name, columns in metric_groups.items() if columns}
        for note in notes:
            normalized_note = normalize_text(note)
            if any(keyword in normalized_note for keyword in ["未检测到", "不存在", "未发现", "有误"]):
                if any(family in normalized_note for family in normalized_families):
                    continue
            filtered.append(note)
        return unique_preserve_order(filtered)

    def _has_explicit_family_summary_cue(self, message: str) -> bool:
        lowered = message.lower()
        return any(
            keyword in message or keyword in lowered
            for keyword in ["按 MTFH 和 MTFV 汇总", "指标组汇总", "family summary", "整体汇总", "不需要每个测点"]
        )
