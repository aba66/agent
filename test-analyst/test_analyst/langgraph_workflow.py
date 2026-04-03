from __future__ import annotations

import json
import re
from datetime import datetime
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .a2a_runtime import A2AAgentRegistry
from .analysis import DataAnalysisAgent, ValidationAgent
from .configuration import ConfigManager
from .file_loader import DataParserAgent
from .llm_client import TestAnalystLLMClient
from .mcp_servers import LocalMCPRegistry, build_local_mcp_registry
from .models import (
    A2AHandoff,
    AgentCard,
    AnalysisPlan,
    AgentTraceEvent,
    AnalysisResult,
    ConversationContext,
    LLMSettings,
    MCPCallRecord,
    MetricTargetPlan,
    MappingPlan,
    MappingResolution,
    RequirementSpec,
    WorkbookContext,
)
from .reporting import ReportGeneratorAgent
from .requirement_parser import RequirementOrchestratorAgent
from .skills_runtime import SkillRegistry
from .utils import jsonify, merge_unique_texts


class GraphState(TypedDict, total=False):
    data_files: list[str]
    template_file: str | None
    request_text: str
    overrides: dict[str, Any]
    conversation_context: ConversationContext
    workbook: WorkbookContext
    data_frames: dict[str, Any]
    analysis_plan: AnalysisPlan
    mapping_plan: MappingPlan
    analysis: AnalysisResult
    clarifications: list[str]
    errors: list[str]
    next_agent: str
    status: str
    trace_events: list[AgentTraceEvent]
    a2a_handoffs: list[A2AHandoff]
    mcp_calls: list[MCPCallRecord]


class LangGraphTestAnalystRuntime:
    """Real multi-agent runtime built on LangGraph."""

    def __init__(
        self,
        *,
        config_manager: ConfigManager,
        data_parser: DataParserAgent,
        requirement_parser: RequirementOrchestratorAgent,
        analysis_agent: DataAnalysisAgent,
        validation_agent: ValidationAgent,
        report_agent: ReportGeneratorAgent,
        llm_settings: LLMSettings,
    ) -> None:
        self.config_manager = config_manager
        self.data_parser = data_parser
        self.requirement_parser = requirement_parser
        self.analysis_agent = analysis_agent
        self.validation_agent = validation_agent
        self.report_agent = report_agent
        self.llm_client = TestAnalystLLMClient(llm_settings)
        self.skill_registry = SkillRegistry(config_manager.skill_root())
        self.mcp_registry: LocalMCPRegistry = build_local_mcp_registry(
            data_parser=data_parser,
            report_agent=report_agent,
            config_manager=config_manager,
        )
        self.agent_registry = A2AAgentRegistry(self._build_agent_cards())
        self.graph = self._build_graph()

    @property
    def agent_cards(self) -> list[AgentCard]:
        return self.agent_registry.cards

    def invoke(self, state: GraphState) -> GraphState:
        initial_state: GraphState = {
            "data_files": state.get("data_files", []),
            "template_file": state.get("template_file"),
            "request_text": state.get("request_text", ""),
            "overrides": state.get("overrides", {}),
            "conversation_context": state.get("conversation_context"),
            "clarifications": [],
            "errors": [],
            "trace_events": [],
            "a2a_handoffs": [],
            "mcp_calls": [],
            "status": "running",
        }
        return self.graph.invoke(initial_state)

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("supervisor_agent", self._wrap_node("supervisor_agent", self._supervisor_logic))
        graph.add_node("data_parser_agent", self._wrap_node("data_parser_agent", self._data_parser_logic))
        graph.add_node(
            "requirement_orchestrator_agent",
            self._wrap_node("requirement_orchestrator_agent", self._requirement_logic),
        )
        graph.add_node("metric_mapping_agent", self._wrap_node("metric_mapping_agent", self._mapping_logic))
        graph.add_node("data_analysis_agent", self._wrap_node("data_analysis_agent", self._analysis_logic))
        graph.add_node("validation_agent", self._wrap_node("validation_agent", self._validation_logic))
        graph.add_node("report_generator_agent", self._wrap_node("report_generator_agent", self._report_logic))

        graph.add_edge(START, "supervisor_agent")
        graph.add_conditional_edges(
            "supervisor_agent",
            self._route_next_agent,
            {
                "data_parser_agent": "data_parser_agent",
                "requirement_orchestrator_agent": "requirement_orchestrator_agent",
                "metric_mapping_agent": "metric_mapping_agent",
                "data_analysis_agent": "data_analysis_agent",
                "validation_agent": "validation_agent",
                "report_generator_agent": "report_generator_agent",
                "end": END,
            },
        )
        graph.add_edge("data_parser_agent", "supervisor_agent")
        graph.add_edge("requirement_orchestrator_agent", "supervisor_agent")
        graph.add_edge("metric_mapping_agent", "supervisor_agent")
        graph.add_edge("data_analysis_agent", "supervisor_agent")
        graph.add_edge("validation_agent", "supervisor_agent")
        graph.add_edge("report_generator_agent", "supervisor_agent")
        return graph.compile()

    def _wrap_node(self, agent_name: str, logic):
        def runner(state: GraphState) -> GraphState:
            started_at = datetime.utcnow().isoformat()
            timer = perf_counter()
            trace_events = list(state.get("trace_events", []))
            try:
                updates = logic(state) or {}
                summary = updates.pop("_summary", "执行成功")
                trace_events.append(
                    AgentTraceEvent(
                        agent=agent_name,
                        step=logic.__name__,
                        status="completed",
                        started_at=started_at,
                        finished_at=datetime.utcnow().isoformat(),
                        duration_ms=int((perf_counter() - timer) * 1000),
                        summary=summary,
                    )
                )
                updates["trace_events"] = trace_events
                return updates
            except Exception as exc:
                trace_events.append(
                    AgentTraceEvent(
                        agent=agent_name,
                        step=logic.__name__,
                        status="failed",
                        started_at=started_at,
                        finished_at=datetime.utcnow().isoformat(),
                        duration_ms=int((perf_counter() - timer) * 1000),
                        summary=str(exc),
                    )
                )
                return {
                    "status": "failed",
                    "errors": merge_unique_texts(state.get("errors", []), [str(exc)]),
                    "trace_events": trace_events,
                    "next_agent": "end",
                }

        return runner

    def _supervisor_logic(self, state: GraphState) -> GraphState:
        next_agent = "end"
        status = state.get("status", "running")

        if state.get("errors"):
            status = "failed"
        elif not state.get("workbook"):
            next_agent = "data_parser_agent"
        elif not state.get("analysis_plan"):
            next_agent = "requirement_orchestrator_agent"
        elif state.get("clarifications"):
            status = "needs_clarification"
        elif not state.get("mapping_plan"):
            next_agent = "metric_mapping_agent"
        elif not state.get("analysis"):
            next_agent = "data_analysis_agent"
        elif state["analysis"] and not state["analysis"].validation_messages:
            next_agent = "validation_agent"
        elif state["analysis"] and not state["analysis"].artifacts.excel_path:
            next_agent = "report_generator_agent"
        else:
            status = "completed"

        updates: GraphState = {"next_agent": next_agent, "status": status, "_summary": f"下一步：{next_agent}"}
        if next_agent != "end":
            handoffs = list(state.get("a2a_handoffs", []))
            handoffs.append(
                self.agent_registry.handoff(
                    from_agent="supervisor_agent",
                    to_agent=next_agent,
                    task_type="delegate-task",
                    summary=f"将当前任务交给 {next_agent}",
                    payload_preview={"status": status},
                )
            )
            updates["a2a_handoffs"] = handoffs
        return updates

    def _route_next_agent(self, state: GraphState) -> str:
        return state.get("next_agent", "end")

    def _data_parser_logic(self, state: GraphState) -> GraphState:
        result, mcp_calls = self._call_mcp_tool(
            state,
            "file-intake-mcp",
            "inspect_inputs",
            data_files=state["data_files"],
            template_file=state.get("template_file"),
        )
        workbook, data_frames, _ = result
        return {
            "workbook": workbook,
            "data_frames": data_frames,
            "mcp_calls": mcp_calls,
            "_summary": f"已解析 {len(workbook.input_files)} 个输入文件",
        }

    def _requirement_logic(self, state: GraphState) -> GraphState:
        workbook = state["workbook"]
        request_text = state["request_text"]
        conversation_context = state.get("conversation_context")
        if not self.llm_client.is_configured:
            raise RuntimeError("无可用 LLM API，数据分析系统无法使用。请先配置有效的 LLM_API_KEY 和 LLM_MODEL。")
        transcript_turns = self._extract_embedded_user_turns(request_text)
        if transcript_turns and not (conversation_context and conversation_context.active_task_id):
            synthesized_context = self._synthesize_context_from_transcript(transcript_turns, workbook)
            if synthesized_context is not None:
                conversation_context = synthesized_context
                request_text = transcript_turns[-1]
        analysis_plan = self._compile_analysis_plan_with_llm(
            request_text=request_text,
            workbook=workbook,
            conversation_context=conversation_context,
            overrides=state.get("overrides"),
        )

        llm_notes = ["需求理解由 LLM 直接编译为唯一 AnalysisPlan，未启用规则兜底主链。"]
        analysis_plan.llm_notes = merge_unique_texts(analysis_plan.llm_notes, llm_notes)
        if transcript_turns:
            analysis_plan.llm_notes = merge_unique_texts(
                analysis_plan.llm_notes,
                ["检测到嵌入式聊天记录，已提取最后一条用户输入并复用前文计划上下文。"],
            )
        clarifications = merge_unique_texts(
            analysis_plan.clarification_questions,
            self.requirement_parser.build_plan_clarifications(analysis_plan, workbook),
        )
        analysis_plan.clarification_questions = clarifications
        return {
            "analysis_plan": analysis_plan,
            "conversation_context": conversation_context,
            "clarifications": clarifications,
            "_summary": f"需求解析完成，计划指标数：{len(analysis_plan.metric_targets)}",
        }

    def _mapping_logic(self, state: GraphState) -> GraphState:
        analysis_plan = state["analysis_plan"]
        workbook = state["workbook"]
        candidate_columns = self._collect_candidate_columns(workbook, analysis_plan.source_sheet)
        field_aliases, mcp_calls = self._call_mcp_tool(state, "rules-config-mcp", "get_field_aliases")

        mapping_plan = MappingPlan()
        if all(target.target_type == "metric_group" and target.columns for target in analysis_plan.metric_targets):
            mapping_plan.notes = ["已识别明确的指标组列，跳过字段映射智能体的二次歧义判断。"]
            clarifications = merge_unique_texts(
                state.get("clarifications", []),
                self.requirement_parser.build_plan_clarifications(analysis_plan, workbook),
            )
            return {
                "analysis_plan": analysis_plan,
                "mapping_plan": mapping_plan,
                "clarifications": clarifications,
                "mcp_calls": mcp_calls,
                "_summary": "指标组列已明确，跳过字段映射",
            }

        unresolved_labels = [target.label for target in analysis_plan.metric_targets if target.target_type == "metric_series" and not target.matched_column]
        if unresolved_labels:
            skill_bundle = self.skill_registry.render_bundle(["metric-mapping"])
            payload = {
                "requested_metrics": unresolved_labels,
                "group_dimensions": analysis_plan.group_dimensions,
                "source_sheet": analysis_plan.source_sheet,
                "candidate_columns": candidate_columns[:200],
                "base_fields": list(field_aliases.get("base_fields", {}).keys()) if isinstance(field_aliases, dict) else [],
            }
            llm_result = self._safe_llm_json(
                agent_name="metric_mapping_agent",
                system_prompt=(
                    "你是测试数据分析系统中的字段映射智能体。"
                    "请只在候选列中选择匹配列，并返回严格 JSON。"
                    "JSON 字段包括：resolved_metrics, base_field_overrides, clarification_questions, notes。"
                    f"\n\n{skill_bundle}"
                ),
                user_prompt=json.dumps(jsonify(payload), ensure_ascii=False, indent=2),
            )
            if llm_result:
                mapping_plan = self._build_mapping_plan(llm_result, candidate_columns)
                mapping_plan.notes = merge_unique_texts(mapping_plan.notes, ["字段映射智能体已调用 LLM。"])
            else:
                reason = self.llm_client.last_error or "未知错误"
                raise RuntimeError(f"字段映射智能体 LLM 调用失败，无法继续执行。原因：{reason}")
        else:
            mapping_plan.notes = ["当前轮没有未解析的单指标请求，跳过字段映射智能体。"]

        analysis_plan = self._apply_mapping_plan_to_analysis_plan(analysis_plan, mapping_plan, candidate_columns)
        clarifications = merge_unique_texts(state.get("clarifications", []), mapping_plan.clarification_questions)
        clarifications = merge_unique_texts(clarifications, self.requirement_parser.build_plan_clarifications(analysis_plan, workbook))
        return {
            "analysis_plan": analysis_plan,
            "mapping_plan": mapping_plan,
            "clarifications": clarifications,
            "mcp_calls": mcp_calls,
            "_summary": f"映射候选列数：{len(candidate_columns)}",
        }

    def _analysis_logic(self, state: GraphState) -> GraphState:
        analysis = self.analysis_agent.analyze(
            state["workbook"],
            state["data_frames"],
            state["analysis_plan"],
        )
        if state.get("mapping_plan"):
            analysis.llm_notes = merge_unique_texts(analysis.llm_notes, state["mapping_plan"].notes)
        return {"analysis": analysis, "_summary": "数据分析完成"}

    def _validation_logic(self, state: GraphState) -> GraphState:
        analysis = state["analysis"]
        analysis_plan = state["analysis_plan"]
        analysis.validation_messages = self.validation_agent.validate(analysis_plan, analysis)

        skill_bundle = self.skill_registry.render_bundle(["result-audit", "report-summary"])
        payload = {
            "analysis_plan": analysis_plan.model_dump(mode="json"),
            "metric_matches": [item.model_dump(mode="json") for item in analysis.metric_matches],
            "statistics": analysis.statistics,
            "validation_messages": analysis.validation_messages,
            "warnings": analysis.warnings,
        }
        llm_result = self._safe_llm_json(
            agent_name="validation_agent",
            system_prompt=(
                "你是测试数据分析系统中的结果校验智能体。"
                "请根据分析结果输出严格 JSON。"
                "JSON 字段包括：narrative_summary, risk_flags, notes。"
                f"\n\n{skill_bundle}"
            ),
            user_prompt=json.dumps(jsonify(payload), ensure_ascii=False, indent=2),
        )
        if llm_result:
            analysis.narrative_summary = merge_unique_texts(
                analysis.narrative_summary,
                llm_result.get("narrative_summary", []),
            )
            analysis.warnings = merge_unique_texts(analysis.warnings, llm_result.get("risk_flags", []))
            analysis.llm_notes = merge_unique_texts(analysis.llm_notes, llm_result.get("notes", []))
        else:
            reason = self.llm_client.last_error or "未知错误"
            raise RuntimeError(f"结果校验智能体 LLM 调用失败，无法继续执行。原因：{reason}")

        return {"analysis": analysis, "_summary": "结果校验完成"}

    def _compile_analysis_plan_with_llm(
        self,
        *,
        request_text: str,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None,
        overrides: dict[str, Any] | None,
    ) -> AnalysisPlan:
        analysis_plan = self._build_analysis_plan_seed(
            request_text=request_text,
            workbook=workbook,
            conversation_context=conversation_context,
        )
        skill_bundle = self.skill_registry.render_bundle(["requirements-clarify", "decay-rule-reasoning"])
        payload = {
            "current_user_message": request_text,
            "conversation_context": self._compact_conversation_context(conversation_context),
            "file_summary": self._compact_workbook_summary(workbook),
            "plan_hints": self._build_analysis_plan_hint(workbook, conversation_context),
        }
        llm_result = self._safe_llm_json(
            agent_name="requirement_orchestrator_agent",
            system_prompt=(
                "你是测试数据分析系统中的需求编排智能体。"
                "你的职责是把当前用户消息和会话上下文编译成唯一的、完整展开的 AnalysisPlan。"
                "AnalysisPlan 是唯一真状态，后续执行层会严格按照这个计划进行 pandas 计算，"
                "不要输出并行冗余状态，不要输出 delta，不要让执行层再去猜模式。"
                "如果当前是会话内追问，你必须返回“本轮执行后的完整新计划”，而不是只返回变化部分。"
                "请把这次输入默认理解为当前会话的增量修订，除非用户明确表示要新建任务、替换数据源、或只保留新的范围。"
                "如果用户在后续消息中新增了指标范围或分组维度，默认与已确认计划合并，而不是覆盖。"
                "像“每个测试项目的每个测试节点”“加上SN号”“带上SN”“包含序列号”这类表达，"
                "都应理解为 group_dimensions 的补充。"
                "像“MTFH_1~MTFH_17”“MTFH1-17”“MTFH_1MTFH_17”“MTFH7到10”这类表达，"
                "应理解为独立的 metric_targets，使用 target_type=metric_group，并优先返回 family/start_index/end_index。"
                "不同范围必须保留为不同 metric_targets，不能合并成单个 MTFH 或 MTFV。"
                "如果用户明确点名某个 SN、测试项目或测试节点来质疑结果，请在 review_targets 中返回该复核对象。"
                "plan_hints 只是文件结构和已确认上下文提示，不能覆盖用户真实意图。"
                "请只输出一个 JSON 对象，字段包括："
                "intent_type, source_sheet, group_dimensions, metric_targets, statistics, decay_method, "
                "output_granularity, output_formats, output_sheet_name, template_sheet_name, template_hint, "
                "stage_value, remark_value, threshold_rules, review_targets, clarification_questions, "
                "parsed_notes, context_inheritance_notes。"
                "其中 metric_targets 是数组；每项字段包括："
                "label, target_type, family, start_index, end_index, columns, matched_column。"
                "statistics 只允许从 min/max/mean/decay 中选择。"
                "output_granularity 只允许 detail/group_summary/point_summary。"
                "如果用户要按测试项目、测试节点、SN 聚合统计指标范围，通常应输出 group_summary。"
                "只有用户明确要求逐点位逐列展开时才输出 point_summary。"
                f"\n\n{skill_bundle}"
            ),
            user_prompt=json.dumps(jsonify(payload), ensure_ascii=False, indent=2),
        )
        if not llm_result:
            reason = self.llm_client.last_error or "未知错误"
            raise RuntimeError(f"需求编排智能体 LLM 调用失败，无法继续执行。原因：{reason}")

        analysis_plan = self._merge_analysis_plan(analysis_plan, llm_result)
        analysis_plan = self.requirement_parser.finalize_analysis_plan(analysis_plan, workbook, conversation_context)
        if overrides:
            analysis_plan = self.requirement_parser.apply_plan_overrides(analysis_plan, overrides)
            analysis_plan = self.requirement_parser.finalize_analysis_plan(analysis_plan, workbook, conversation_context)
        return analysis_plan

    def _report_logic(self, state: GraphState) -> GraphState:
        analysis = state["analysis"]
        analysis_plan = state["analysis_plan"]
        artifacts, mcp_calls = self._call_mcp_tool(
            state,
            "report-artifact-mcp",
            "generate_reports",
            workbook=state["workbook"],
            analysis_plan=analysis_plan,
            analysis_result=analysis,
            template_file=state.get("template_file"),
        )
        analysis.artifacts = artifacts
        return {"analysis": analysis, "mcp_calls": mcp_calls, "_summary": "已生成交付物"}

    def _safe_llm_json(self, *, agent_name: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            return self._coerce_dict_payload(
                self.llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    agent_name=agent_name,
                )
            )
        except Exception:
            return {}

    def _build_analysis_plan_seed(
        self,
        *,
        request_text: str,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None,
    ) -> AnalysisPlan:
        default_formats = self.config_manager.business_rules().get("default_report_formats", ["excel"])
        output_formats = list(default_formats) if default_formats else ["excel"]
        output_sheet_name = self.config_manager.business_rules().get("default_output_sheet_name", "KPI Summary")
        plan = AnalysisPlan(
            raw_request=request_text,
            output_formats=output_formats,
            output_sheet_name=output_sheet_name,
        )
        if len(workbook.sheet_names) == 1:
            plan.source_sheet = workbook.default_sheet
        confirmed_context = conversation_context.latest_confirmed_plan if conversation_context else {}
        if confirmed_context:
            plan.intent_type = "follow_up"
        return plan

    def _build_analysis_plan_hint(
        self,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None,
    ) -> dict[str, Any]:
        hint = {
            "available_sheets": workbook.sheet_names,
            "default_sheet": workbook.default_sheet,
            "sheet_columns": {
                sheet.name: sheet.column_names[:80]
                for sheet in workbook.sheet_summaries
                if sheet.column_names
            },
        }
        confirmed_context = conversation_context.latest_confirmed_plan if conversation_context else {}
        if confirmed_context:
            hint["latest_confirmed_plan"] = confirmed_context
        return hint

    def _analysis_plan_needs_fallback_backfill(self, analysis_plan: AnalysisPlan) -> bool:
        if not analysis_plan.source_sheet:
            return True
        if not analysis_plan.metric_targets:
            return True
        if analysis_plan.output_granularity in {"group_summary", "point_summary"} and not analysis_plan.statistics:
            return True
        if (
            analysis_plan.output_granularity == "group_summary"
            and analysis_plan.group_dimensions
            and not any(target.target_type == "metric_group" for target in analysis_plan.metric_targets)
            and self.requirement_parser.looks_like_metric_series_request([target.label for target in analysis_plan.metric_targets])
        ):
            return True
        if "decay" in analysis_plan.statistics and not analysis_plan.decay_method:
            return True
        return False

    def _backfill_analysis_plan_from_fallback(
        self,
        *,
        analysis_plan: AnalysisPlan,
        fallback_plan: AnalysisPlan,
    ) -> AnalysisPlan:
        updated = analysis_plan.copy(deep=True)
        for field_name in (
            "source_sheet",
            "decay_method",
            "output_granularity",
            "template_sheet_name",
            "template_hint",
            "stage_value",
            "remark_value",
            "threshold_rules",
        ):
            current_value = getattr(updated, field_name, None)
            fallback_value = getattr(fallback_plan, field_name, None)
            if current_value in (None, "", [], {}) and fallback_value not in (None, "", [], {}):
                setattr(updated, field_name, fallback_value)

        if not updated.group_dimensions:
            updated.group_dimensions = list(fallback_plan.group_dimensions)
        else:
            updated.group_dimensions = merge_unique_texts(fallback_plan.group_dimensions, updated.group_dimensions)
        if not updated.statistics:
            updated.statistics = list(fallback_plan.statistics)
        else:
            updated.statistics = merge_unique_texts(fallback_plan.statistics, updated.statistics)
        if not updated.metric_targets:
            updated.metric_targets = list(fallback_plan.metric_targets)
        else:
            existing_labels = {target.label for target in updated.metric_targets}
            updated.metric_targets.extend([target for target in fallback_plan.metric_targets if target.label not in existing_labels])
        updated.review_targets = list(updated.review_targets or fallback_plan.review_targets)
        updated.parsed_notes = merge_unique_texts(fallback_plan.parsed_notes, updated.parsed_notes)
        updated.context_inheritance_notes = merge_unique_texts(
            fallback_plan.context_inheritance_notes,
            updated.context_inheritance_notes,
        )
        updated.clarification_questions = merge_unique_texts(
            fallback_plan.clarification_questions,
            updated.clarification_questions,
        )
        updated.output_formats = merge_unique_texts(fallback_plan.output_formats, updated.output_formats)
        return updated

    def _merge_analysis_plan(self, base: AnalysisPlan, payload: dict[str, Any]) -> AnalysisPlan:
        payload = self._coerce_dict_payload(payload)
        updated = base.copy(deep=True)
        if payload.get("intent_type"):
            updated.intent_type = str(payload["intent_type"]).strip()
        if payload.get("source_sheet"):
            updated.source_sheet = str(payload["source_sheet"]).strip()
        if payload.get("group_dimensions"):
            updated.group_dimensions = self._coerce_text_list(payload["group_dimensions"])
        if payload.get("statistics"):
            updated.statistics = self._coerce_text_list(payload["statistics"])
        if payload.get("decay_method"):
            updated.decay_method = str(payload["decay_method"]).strip()
        if payload.get("output_granularity"):
            updated.output_granularity = str(payload["output_granularity"]).strip()
        if payload.get("output_formats"):
            updated.output_formats = self._coerce_text_list(payload["output_formats"])
        if payload.get("output_sheet_name"):
            updated.output_sheet_name = str(payload["output_sheet_name"]).strip()
        if payload.get("template_sheet_name"):
            updated.template_sheet_name = str(payload["template_sheet_name"]).strip()
        if payload.get("template_hint"):
            updated.template_hint = str(payload["template_hint"]).strip()
        if payload.get("stage_value"):
            updated.stage_value = str(payload["stage_value"]).strip()
        if payload.get("remark_value"):
            updated.remark_value = str(payload["remark_value"]).strip()
        if payload.get("threshold_rules"):
            updated.threshold_rules = self._coerce_dict_value(payload["threshold_rules"])
        if payload.get("metric_targets"):
            targets: list[Any] = payload["metric_targets"] if isinstance(payload["metric_targets"], list) else []
            updated.metric_targets = []
            for item in targets:
                if not isinstance(item, dict):
                    continue
                updated.metric_targets.append(
                    MetricTargetPlan(
                        label=str(item.get("label", "")).strip(),
                        target_type=str(item.get("target_type", "metric_group")).strip() or "metric_group",
                        family=str(item.get("family")).strip().upper() if item.get("family") else None,
                        start_index=self._coerce_int(item.get("start_index")),
                        end_index=self._coerce_int(item.get("end_index")),
                        columns=self._coerce_text_list(item.get("columns", [])),
                        matched_column=str(item.get("matched_column")).strip() if item.get("matched_column") else None,
                    )
                )
        if payload.get("review_targets"):
            review_targets: list[Any] = payload["review_targets"] if isinstance(payload["review_targets"], list) else []
            updated.review_targets = []
            for item in review_targets:
                if not isinstance(item, dict):
                    continue
                dimension = str(item.get("dimension", "")).strip()
                value = str(item.get("value", "")).strip()
                if not dimension or not value:
                    continue
                updated.review_targets.append(
                    ReviewTargetPlan(
                        dimension=dimension,
                        value=value,
                        metric_labels=self._coerce_text_list(item.get("metric_labels", [])),
                        reason=str(item.get("reason", "")).strip(),
                    )
                )
        updated.parsed_notes = merge_unique_texts(updated.parsed_notes, self._coerce_text_list(payload.get("parsed_notes", [])))
        updated.context_inheritance_notes = merge_unique_texts(
            updated.context_inheritance_notes,
            self._coerce_text_list(payload.get("context_inheritance_notes", [])),
        )
        updated.clarification_questions = merge_unique_texts(
            updated.clarification_questions,
            self._coerce_text_list(payload.get("clarification_questions", [])),
        )
        return updated

    def _apply_mapping_plan_to_analysis_plan(
        self,
        analysis_plan: AnalysisPlan,
        mapping_plan: MappingPlan,
        candidate_columns: list[str],
    ) -> AnalysisPlan:
        updated = analysis_plan.copy(deep=True)
        resolved_lookup = {
            item.requested_name: item.matched_column
            for item in mapping_plan.resolved_metrics
            if item.requested_name and item.matched_column in candidate_columns
        }
        for target in updated.metric_targets:
            if target.target_type == "metric_series" and not target.matched_column:
                matched = resolved_lookup.get(target.label)
                if matched:
                    target.matched_column = matched
        return updated

    def _coerce_dict_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                loaded = json.loads(text)
            except Exception:
                return {}
            return loaded if isinstance(loaded, dict) else {}
        return {}

    def _coerce_text_list(self, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            items = value
        elif isinstance(value, tuple):
            items = list(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    loaded = json.loads(text)
                    if isinstance(loaded, list):
                        items = loaded
                    else:
                        items = [text]
                except Exception:
                    items = [part.strip() for part in text.splitlines() if part.strip()]
            else:
                chunks = [part.strip() for part in text.replace("，", ",").split(",")]
                if len(chunks) <= 1:
                    items = [part.strip() for part in text.splitlines() if part.strip()]
                else:
                    items = [part for part in chunks if part]
                if not items:
                    items = [text]
        else:
            items = [value]

        result: list[str] = []
        for item in items:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _coerce_dict_value(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                loaded = json.loads(text)
            except Exception:
                return {}
            return loaded if isinstance(loaded, dict) else {}
        return {}

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return default

    def _coerce_resolved_metrics(self, value: Any) -> list[dict[str, Any]]:
        if value in (None, ""):
            return []
        items: list[Any]
        if isinstance(value, dict):
            items = [value]
        elif isinstance(value, list):
            items = value
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                loaded = json.loads(text)
            except Exception:
                return []
            if isinstance(loaded, dict):
                items = [loaded]
            elif isinstance(loaded, list):
                items = loaded
            else:
                return []
        else:
            return []

        result: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, str):
                parsed = self._coerce_dict_payload(item)
                if parsed:
                    result.append(parsed)
        return result

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _call_mcp_tool(self, state: GraphState, server_name: str, tool_name: str, **kwargs):
        started_at = datetime.utcnow().isoformat()
        timer = perf_counter()
        mcp_calls = list(state.get("mcp_calls", []))
        result = self.mcp_registry.call_tool(server_name, tool_name, **kwargs)
        mcp_calls.append(
            MCPCallRecord(
                server=server_name,
                action="tool",
                target=tool_name,
                status="completed",
                started_at=started_at,
                finished_at=datetime.utcnow().isoformat(),
                duration_ms=int((perf_counter() - timer) * 1000),
                summary=f"调用参数：{', '.join(kwargs.keys())}",
            )
        )
        return result, mcp_calls

    def _compact_workbook_summary(self, workbook: WorkbookContext) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for item in workbook.input_files:
            summary.append(
                {
                    "file_name": item.file_path.name,
                    "role": item.role,
                    "sheet_summaries": [
                        {
                            "name": sheet.name,
                            "rows": sheet.rows,
                            "columns": sheet.columns,
                            "column_names": sheet.column_names[:40],
                            "header_candidates": sheet.header_candidates[:5],
                        }
                        for sheet in item.sheet_summaries
                    ],
                }
            )
        return summary

    def _compact_conversation_context(self, context: ConversationContext | None) -> dict[str, Any]:
        if context is None:
            return {}
        return {
            "current_message": context.current_message,
            "history_summary": context.history_summary[-6:],
            "latest_confirmed_plan": context.latest_confirmed_plan,
            "latest_confirmed_requirement": context.latest_confirmed_requirement,
            "inherited_fields": context.inherited_fields,
            "pending_clarifications": context.pending_clarifications,
            "notes": context.notes,
        }

    def _compact_requirement_hint(self, requirement: RequirementSpec) -> dict[str, Any]:
        return {
            key: value
            for key, value in requirement.model_dump(mode="json").items()
            if value not in (None, "", [], {})
            and key
            in {
                "intent_type",
                "source_sheet",
                "requested_metrics",
                "metric_groups",
                "metric_group_mode",
                "aggregate_by_fields",
                "requested_statistics",
                "analysis_mode",
                "stage_value",
                "remark_value",
                "template_hint",
                "template_sheet_name",
                "decay_method",
                "group_by",
                "threshold_rules",
                "clarification_questions",
                "parsed_notes",
                "context_inheritance_notes",
            }
        }

    def _build_requirement_hint(
        self,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None,
    ) -> dict[str, Any]:
        hint = {
            "available_sheets": workbook.sheet_names,
            "default_sheet": workbook.default_sheet,
            "sheet_columns": {
                sheet.name: sheet.column_names[:80]
                for sheet in workbook.sheet_summaries
                if sheet.column_names
            },
        }
        confirmed_context = conversation_context.latest_confirmed_requirement if conversation_context else {}
        if confirmed_context:
            hint["latest_confirmed_requirement"] = {
                key: value
                for key, value in confirmed_context.items()
                if value not in (None, "", [], {})
                and key
                in {
                    "source_sheet",
                    "requested_metrics",
                    "metric_groups",
                    "metric_group_mode",
                    "aggregate_by_fields",
                    "requested_statistics",
                    "analysis_mode",
                    "decay_method",
                    "group_by",
                    "template_sheet_name",
                }
            }
        return hint

    def _collect_candidate_columns(self, workbook: WorkbookContext, source_sheet: str | None) -> list[str]:
        columns: list[str] = []
        for item in workbook.input_files:
            if item.role != "data":
                continue
            for sheet in item.sheet_summaries:
                if source_sheet and sheet.name != source_sheet:
                    continue
                for column in sheet.column_names:
                    if column and column not in columns:
                        columns.append(column)
        return columns

    def _extract_embedded_user_turns(self, request_text: str) -> list[str]:
        if "用户输入" not in request_text:
            return []
        matches = re.findall(
            r"用户输入[:：]\s*(.*?)(?=\n\s*(?:已完成本轮分析[:：]|用户输入[:：])|\Z)",
            request_text,
            flags=re.DOTALL,
        )
        turns = [" ".join(item.split()) for item in matches if item and item.strip()]
        return [item for item in turns if item]

    def _synthesize_context_from_transcript(
        self,
        transcript_turns: list[str],
        workbook: WorkbookContext,
    ) -> ConversationContext | None:
        if len(transcript_turns) < 2:
            return None
        if not self.llm_client.is_configured:
            return None

        confirmed_plan: AnalysisPlan | None = None
        history_summary: list[str] = []
        for index, turn in enumerate(transcript_turns[:-1], start=1):
            prior_context = None
            if confirmed_plan is not None:
                confirmed_requirement = self.requirement_parser.plan_to_requirement(confirmed_plan)
                prior_context = ConversationContext(
                    current_message=turn,
                    latest_confirmed_plan=confirmed_plan.model_dump(mode="json"),
                    latest_confirmed_requirement=confirmed_requirement.model_dump(mode="json"),
                    history_summary=list(history_summary),
                )
            confirmed_plan = self._compile_analysis_plan_with_llm(
                request_text=turn,
                workbook=workbook,
                conversation_context=prior_context,
                overrides=None,
            )
            history_summary.append(
                " | ".join(
                    [
                        f"转录轮次 {index}",
                        f"用户请求：{turn[:120]}",
                        (
                            "指标="
                            + ", ".join(target.label for target in confirmed_plan.metric_targets[:4])
                            if confirmed_plan.metric_targets
                            else "指标=未识别"
                        ),
                    ]
                )
            )

        if confirmed_plan is None:
            return None

        confirmed_requirement = self.requirement_parser.plan_to_requirement(confirmed_plan)
        inherited_fields = [
            key
            for key in [
                "source_sheet",
                "group_dimensions",
                "metric_targets",
                "statistics",
                "decay_method",
                "output_granularity",
            ]
            if getattr(confirmed_plan, key, None) not in (None, "", [], {})
        ]
        return ConversationContext(
            current_message=transcript_turns[-1],
            history_summary=history_summary,
            latest_confirmed_plan=confirmed_plan.model_dump(mode="json"),
            latest_confirmed_requirement=confirmed_requirement.model_dump(mode="json"),
            inherited_fields=inherited_fields,
            notes=["检测到用户粘贴了聊天记录，系统已自动抽取前文会话上下文。"],
        )

    def _build_requirement_seed(
        self,
        *,
        request_text: str,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None,
    ) -> RequirementSpec:
        default_formats = self.config_manager.business_rules().get("default_report_formats", ["excel"])
        output_formats = list(default_formats) if default_formats else ["excel"]
        output_sheet_name = self.config_manager.business_rules().get("default_output_sheet_name", "KPI Summary")
        seed = RequirementSpec(
            raw_request=request_text,
            output_formats=output_formats,
            output_sheet_name=output_sheet_name,
            need_chart="chart" in output_formats,
            need_report="markdown" in output_formats,
            need_pdf="pdf" in output_formats,
            need_docx="docx" in output_formats,
        )
        if len(workbook.sheet_names) == 1:
            seed.source_sheet = workbook.default_sheet
        confirmed_context = conversation_context.latest_confirmed_requirement if conversation_context else {}
        if confirmed_context:
            seed.intent_type = "follow_up"
        return seed

    def _requirement_needs_fallback_backfill(self, requirement: RequirementSpec) -> bool:
        if not requirement.source_sheet:
            return True
        if not requirement.requested_metrics and not requirement.metric_groups:
            return True
        if (
            requirement.requested_statistics
            and requirement.aggregate_by_fields
            and not requirement.metric_groups
            and self.requirement_parser.looks_like_metric_series_request(requirement.requested_metrics)
        ):
            return True
        if requirement.analysis_mode in {"metric_group_summary", "metric_group_point_summary", "grouped_summary"}:
            if not requirement.requested_statistics:
                return True
        if requirement.metric_groups and requirement.analysis_mode in {"metric_group_summary", "metric_group_point_summary"}:
            if not requirement.requested_metrics:
                return True
        if "decay" in requirement.requested_statistics and not requirement.decay_method:
            return True
        return False

    def _backfill_requirement_from_fallback(
        self,
        *,
        requirement: RequirementSpec,
        fallback_requirement: RequirementSpec,
        workbook: WorkbookContext,
        conversation_context: ConversationContext | None,
    ) -> RequirementSpec:
        updated = requirement.copy(deep=True)
        fallback_fields = [
            "source_sheet",
            "metric_group_mode",
            "stage_value",
            "remark_value",
            "template_hint",
            "template_sheet_name",
            "decay_method",
            "analysis_mode",
            "aggregate_by_fields",
            "requested_statistics",
            "group_by",
            "threshold_rules",
        ]
        for field_name in fallback_fields:
            current_value = getattr(updated, field_name, None)
            if current_value not in (None, "", [], {}):
                continue
            fallback_value = getattr(fallback_requirement, field_name, None)
            if fallback_value in (None, "", [], {}):
                continue
            setattr(updated, field_name, fallback_value)

        if fallback_requirement.metric_groups:
            if updated.metric_groups:
                updated.metric_groups = self.requirement_parser._merge_metric_groups(
                    fallback_requirement.metric_groups,
                    updated.metric_groups,
                )
            else:
                updated.metric_groups = fallback_requirement.metric_groups

        for list_field in ("aggregate_by_fields", "requested_statistics", "group_by"):
            fallback_value = getattr(fallback_requirement, list_field, None) or []
            current_value = getattr(updated, list_field, None) or []
            if fallback_value:
                setattr(updated, list_field, merge_unique_texts(fallback_value, current_value))

        if fallback_requirement.requested_metrics:
            updated.requested_metrics = merge_unique_texts(
                fallback_requirement.requested_metrics,
                updated.requested_metrics,
            )

        if updated.metric_groups and updated.analysis_mode in {"metric_group_summary", "metric_group_point_summary"}:
            updated.requested_metrics = merge_unique_texts(list(updated.metric_groups.keys()), updated.requested_metrics)

        updated.parsed_notes = merge_unique_texts(updated.parsed_notes, fallback_requirement.parsed_notes)
        updated.context_inheritance_notes = merge_unique_texts(
            updated.context_inheritance_notes,
            fallback_requirement.context_inheritance_notes,
        )
        updated.clarification_questions = merge_unique_texts(
            updated.clarification_questions,
            fallback_requirement.clarification_questions,
        )
        if not updated.output_formats:
            updated.output_formats = list(fallback_requirement.output_formats or ["excel"])
        updated.need_chart = "chart" in updated.output_formats
        updated.need_report = "markdown" in updated.output_formats
        updated.need_pdf = "pdf" in updated.output_formats
        updated.need_docx = "docx" in updated.output_formats
        return self.requirement_parser._sanitize_requirement(updated, workbook, conversation_context)

    def _merge_requirement(self, base: RequirementSpec, payload: dict[str, Any]) -> RequirementSpec:
        payload = self._coerce_dict_payload(payload)
        updated = base.copy(deep=True)
        if payload.get("intent_type"):
            updated.intent_type = str(payload["intent_type"]).strip()
        if payload.get("source_sheet"):
            updated.source_sheet = str(payload["source_sheet"]).strip()
        if payload.get("requested_metrics"):
            updated.requested_metrics = self._coerce_text_list(payload["requested_metrics"])
        if payload.get("metric_groups"):
            updated.metric_groups = {
                str(key).strip(): self._coerce_text_list(value)
                for key, value in self._coerce_dict_value(payload["metric_groups"]).items()
                if str(key).strip()
            }
        if payload.get("metric_group_mode"):
            updated.metric_group_mode = str(payload["metric_group_mode"]).strip()
        if payload.get("aggregate_by_fields"):
            updated.aggregate_by_fields = self._coerce_text_list(payload["aggregate_by_fields"])
        if payload.get("requested_statistics"):
            updated.requested_statistics = self._coerce_text_list(payload["requested_statistics"])
        if payload.get("analysis_mode"):
            updated.analysis_mode = str(payload["analysis_mode"]).strip()
        if "stage_value" in payload and payload.get("stage_value") not in ("", None):
            updated.stage_value = str(payload["stage_value"]).strip()
        if "remark_value" in payload and payload.get("remark_value") not in ("", None):
            updated.remark_value = str(payload["remark_value"]).strip()
        if payload.get("template_hint"):
            updated.template_hint = str(payload["template_hint"]).strip()
        if payload.get("template_sheet_name"):
            updated.template_sheet_name = str(payload["template_sheet_name"]).strip()
        if payload.get("decay_method"):
            updated.decay_method = str(payload["decay_method"]).strip()
        if payload.get("group_by"):
            updated.group_by = self._coerce_text_list(payload["group_by"])
        if payload.get("threshold_rules"):
            updated.threshold_rules = self._coerce_dict_value(payload["threshold_rules"])
        if "need_chart" in payload:
            updated.need_chart = self._coerce_bool(payload["need_chart"], updated.need_chart)
        if "need_report" in payload:
            updated.need_report = self._coerce_bool(payload["need_report"], updated.need_report)
        if "need_pdf" in payload:
            updated.need_pdf = self._coerce_bool(payload["need_pdf"], updated.need_pdf)
        if "need_docx" in payload:
            updated.need_docx = self._coerce_bool(payload["need_docx"], updated.need_docx)
        updated.output_formats = merge_unique_texts(
            ["excel"],
            ["chart"] if updated.need_chart else [],
            ["markdown"] if updated.need_report else [],
            ["pdf"] if updated.need_pdf else [],
            ["docx"] if updated.need_docx else [],
        )
        updated.parsed_notes = merge_unique_texts(updated.parsed_notes, self._coerce_text_list(payload.get("parsed_notes", [])))
        updated.context_inheritance_notes = merge_unique_texts(
            updated.context_inheritance_notes,
            self._coerce_text_list(payload.get("context_inheritance_notes", [])),
        )
        updated.clarification_questions = merge_unique_texts(
            updated.clarification_questions,
            self._coerce_text_list(payload.get("clarification_questions", [])),
        )
        return updated

    def _build_mapping_plan(self, payload: dict[str, Any], candidate_columns: list[str]) -> MappingPlan:
        payload = self._coerce_dict_payload(payload)
        resolved_metrics: list[MappingResolution] = []
        for item in self._coerce_resolved_metrics(payload.get("resolved_metrics", [])):
            matched_column = item.get("matched_column")
            if matched_column and matched_column not in candidate_columns:
                matched_column = None
            resolved_metrics.append(
                MappingResolution(
                    requested_name=str(item.get("requested_name", "")).strip(),
                    matched_column=matched_column,
                    confidence=self._coerce_float(item.get("confidence", 0.0) or 0.0),
                    reason=str(item.get("reason", "")).strip(),
                )
            )
        base_field_overrides = {
            str(key).strip(): str(value).strip()
            for key, value in self._coerce_dict_value(payload.get("base_field_overrides") or {}).items()
            if str(value).strip() in candidate_columns
        }
        return MappingPlan(
            resolved_metrics=resolved_metrics,
            base_field_overrides=base_field_overrides,
            clarification_questions=self._coerce_text_list(payload.get("clarification_questions", [])),
            notes=self._coerce_text_list(payload.get("notes", [])),
        )

    def _coerce_int(self, value: Any) -> int | None:
        if value in (None, "", []):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_agent_cards(self) -> list[AgentCard]:
        return [
            AgentCard(
                agent_id="supervisor_agent",
                title="需求编排监督智能体",
                description="管理 LangGraph 中的节点跳转、handoff 和会话状态。",
                skills=["requirements-clarify"],
                input_schema={"state": "workflow state"},
                output_schema={"next_agent": "string"},
            ),
            AgentCard(
                agent_id="data_parser_agent",
                title="数据解析智能体",
                description="通过 MCP 工具读取文件、识别 sheet 与字段结构。",
                skills=[],
                input_schema={"data_files": "list[path]"},
                output_schema={"workbook": "WorkbookContext"},
            ),
            AgentCard(
                agent_id="requirement_orchestrator_agent",
                title="需求编排智能体",
                description="理解自然语言需求、提炼结构化需求和澄清问题。",
                skills=["requirements-clarify", "decay-rule-reasoning"],
                input_schema={"request_text": "string", "workbook": "WorkbookContext"},
                output_schema={"requirement": "RequirementSpec"},
            ),
            AgentCard(
                agent_id="metric_mapping_agent",
                title="字段映射智能体",
                description="结合技能与候选列，判断测试项和基础字段的映射。",
                skills=["metric-mapping"],
                input_schema={"requirement": "RequirementSpec", "candidate_columns": "list[string]"},
                output_schema={"mapping_plan": "MappingPlan"},
            ),
            AgentCard(
                agent_id="data_analysis_agent",
                title="数据分析智能体",
                description="执行确定性数据筛选、聚合、衰减和统计分析。",
                skills=[],
                input_schema={"data_frames": "dict", "requirement": "RequirementSpec"},
                output_schema={"analysis": "AnalysisResult"},
            ),
            AgentCard(
                agent_id="validation_agent",
                title="结果校验智能体",
                description="进行校验、自检和结果总结。",
                skills=["result-audit", "report-summary"],
                input_schema={"analysis": "AnalysisResult"},
                output_schema={"validation_messages": "list[string]"},
            ),
            AgentCard(
                agent_id="report_generator_agent",
                title="报告生成智能体",
                description="通过 MCP 工具生成 Excel、图表和多格式报告。",
                skills=["template-fill", "report-summary"],
                input_schema={"analysis": "AnalysisResult", "template_file": "path"},
                output_schema={"artifacts": "AnalysisArtifacts"},
            ),
        ]
