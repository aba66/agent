from __future__ import annotations
from pathlib import Path

from .analysis import DataAnalysisAgent, ValidationAgent
from .configuration import ConfigManager
from .file_loader import DataParserAgent
from .langgraph_workflow import LangGraphTestAnalystRuntime
from .models import AnalysisPlan, ConversationContext, LLMSettings, RequirementSpec, UserContext, WorkflowResult
from .reporting import ReportGeneratorAgent
from .requirement_parser import RequirementOrchestratorAgent
from .storage import TaskStore
from .utils import jsonify, merge_unique_texts


class TestAnalystWorkflow:
    """Public workflow facade backed by a LangGraph multi-agent runtime."""

    def __init__(self, project_root: str | Path, output_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.output_root = Path(output_root).resolve() if output_root else self.project_root / "output"

        self.config_manager = ConfigManager(self.project_root)
        self.data_parser = DataParserAgent()
        self.requirement_agent = RequirementOrchestratorAgent(config_manager=self.config_manager)
        self.analysis_agent = DataAnalysisAgent(config_manager=self.config_manager)
        self.validation_agent = ValidationAgent()
        self.report_agent = ReportGeneratorAgent(output_root=self.output_root, config_manager=self.config_manager)
        self.task_store = TaskStore(self.project_root / "data" / "task_history.db")

    def inspect_inputs(
        self,
        data_files: list[str | Path],
        template_file: str | Path | None = None,
    ):
        return self.data_parser.load_many(data_files, template_file=template_file)

    def _build_runtime(self, llm_settings: LLMSettings) -> LangGraphTestAnalystRuntime:
        return LangGraphTestAnalystRuntime(
            config_manager=self.config_manager,
            data_parser=self.data_parser,
            requirement_parser=self.requirement_agent,
            analysis_agent=self.analysis_agent,
            validation_agent=self.validation_agent,
            report_agent=self.report_agent,
            llm_settings=llm_settings,
        )

    def run(
        self,
        data_files: str | Path | list[str | Path] | None,
        request_text: str,
        template_file: str | Path | None = None,
        overrides: dict[str, object] | None = None,
        session_id: str | None = None,
        parent_task_id: str | None = None,
        user_context: UserContext | None = None,
        session_title: str = "默认分析会话",
        llm_settings: dict[str, object] | None = None,
    ) -> WorkflowResult:
        user = user_context or UserContext(username="analyst", role="analyst")
        raw_files = [] if data_files is None else ([data_files] if isinstance(data_files, (str, Path)) else list(data_files))

        session_id = self.task_store.get_or_create_session(session_id, session_title, user)
        context_task = self._resolve_context_task(session_id=session_id, parent_task_id=parent_task_id)
        file_list = [Path(path) for path in raw_files] if raw_files else self._inherit_input_files(context_task)
        if not file_list:
            raise ValueError("未提供输入文件，且当前会话/父版本中也没有可继承的数据文件。")

        resolved_template = Path(template_file) if template_file else self._inherit_template_file(context_task)
        conversation_context = self._build_conversation_context(context_task=context_task, request_text=request_text)
        effective_request_text = request_text
        effective_overrides = self._merge_overrides_with_context(overrides or {}, context_task)
        runtime = self._build_runtime(self.config_manager.llm_settings(llm_settings))

        task_id, revision = self.task_store.start_task(
            session_id=session_id,
            parent_task_id=parent_task_id,
            title=session_title,
            user=user,
            request_text=request_text,
            input_files=[str(Path(path)) for path in file_list],
            template_file=str(resolved_template) if resolved_template else None,
        )

        final_state = runtime.invoke(
            {
                "data_files": [str(Path(path)) for path in file_list],
                "template_file": str(resolved_template) if resolved_template else None,
                "request_text": effective_request_text,
                "overrides": effective_overrides,
                "conversation_context": conversation_context,
            }
        )

        analysis_plan = final_state.get("analysis_plan") or AnalysisPlan(raw_request=request_text)
        requirement = self.requirement_agent.plan_to_requirement(analysis_plan)
        trace_events = final_state.get("trace_events", [])
        a2a_handoffs = final_state.get("a2a_handoffs", [])
        mcp_calls = final_state.get("mcp_calls", [])
        conversation_context = final_state.get("conversation_context", conversation_context)
        status = final_state.get("status", "failed")
        workbook = final_state.get("workbook")
        if workbook is None:
            workbook, _, _ = self.inspect_inputs(file_list, template_file=resolved_template)

        inherited_notes = self._build_inherited_context_notes(context_task)
        if inherited_notes:
            requirement.context_inheritance_notes = merge_unique_texts(
                requirement.context_inheritance_notes,
                inherited_notes,
            )

        analysis = final_state.get("analysis")
        if analysis is not None:
            analysis.trace_events = trace_events

        if status == "needs_clarification":
            task_record = self.task_store.finish_task(
                task_id=task_id,
                status="needs_clarification",
                requirement=requirement.model_dump(mode="json"),
                artifacts={
                    "analysis_plan": analysis_plan.model_dump(mode="json"),
                    "current_request_text": request_text,
                    "effective_request_text": effective_request_text,
                    "context_task_id": context_task.task_id if context_task else None,
                    "conversation_context": jsonify(conversation_context),
                    "a2a_handoffs": jsonify([item.model_dump(mode="json") for item in a2a_handoffs]),
                    "mcp_calls": jsonify([item.model_dump(mode="json") for item in mcp_calls]),
                },
                clarifications=final_state.get("clarifications", []),
                errors=[],
                trace_events=[item.model_dump(mode="json") for item in trace_events],
            )
            return WorkflowResult(
                status="needs_clarification",
                workbook=workbook,
                analysis_plan=analysis_plan,
                requirement=requirement,
                conversation_context=conversation_context,
                clarifications=final_state.get("clarifications", []),
                session_id=session_id,
                task_id=task_id,
                revision=revision,
                trace_events=trace_events,
                agent_cards=runtime.agent_cards,
                a2a_handoffs=a2a_handoffs,
                mcp_calls=mcp_calls,
                task_record=task_record,
            )

        if status == "completed" and analysis is not None:
            task_record = self.task_store.finish_task(
                task_id=task_id,
                status="completed",
                requirement=requirement.model_dump(mode="json"),
                artifacts={
                    "analysis_plan": analysis_plan.model_dump(mode="json"),
                    **jsonify(analysis.artifacts.model_dump(mode="json")),
                    "current_request_text": request_text,
                    "effective_request_text": effective_request_text,
                    "context_task_id": context_task.task_id if context_task else None,
                    "conversation_context": jsonify(conversation_context),
                    "a2a_handoffs": jsonify([item.model_dump(mode="json") for item in a2a_handoffs]),
                    "mcp_calls": jsonify([item.model_dump(mode="json") for item in mcp_calls]),
                },
                clarifications=[],
                errors=[],
                trace_events=[item.model_dump(mode="json") for item in trace_events],
            )
            return WorkflowResult(
                status="completed",
                workbook=workbook,
                analysis_plan=analysis_plan,
                requirement=requirement,
                analysis=analysis,
                conversation_context=conversation_context,
                session_id=session_id,
                task_id=task_id,
                revision=revision,
                trace_events=trace_events,
                agent_cards=runtime.agent_cards,
                a2a_handoffs=a2a_handoffs,
                mcp_calls=mcp_calls,
                task_record=task_record,
            )

        errors = final_state.get("errors", []) or ["多智能体工作流执行失败。"]
        task_record = self.task_store.finish_task(
            task_id=task_id,
            status="failed",
            requirement=requirement.model_dump(mode="json"),
            artifacts={
                "analysis_plan": analysis_plan.model_dump(mode="json"),
                "current_request_text": request_text,
                "effective_request_text": effective_request_text,
                "context_task_id": context_task.task_id if context_task else None,
                "conversation_context": jsonify(conversation_context),
                "a2a_handoffs": jsonify([item.model_dump(mode="json") for item in a2a_handoffs]),
                "mcp_calls": jsonify([item.model_dump(mode="json") for item in mcp_calls]),
            },
            clarifications=[],
            errors=errors,
            trace_events=[item.model_dump(mode="json") for item in trace_events],
        )
        return WorkflowResult(
            status="failed",
            workbook=workbook,
            analysis_plan=analysis_plan,
            requirement=requirement,
            conversation_context=conversation_context,
            errors=errors,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            trace_events=trace_events,
            agent_cards=runtime.agent_cards,
            a2a_handoffs=a2a_handoffs,
            mcp_calls=mcp_calls,
            task_record=task_record,
        )

    def _resolve_context_task(self, session_id: str, parent_task_id: str | None):
        if parent_task_id:
            return self.task_store.get_task(parent_task_id)
        return self.task_store.get_latest_task(session_id)

    def _inherit_input_files(self, context_task) -> list[Path]:
        if not context_task:
            return []
        return [Path(path) for path in context_task.input_files]

    def _inherit_template_file(self, context_task) -> Path | None:
        if not context_task or not context_task.template_file:
            return None
        return Path(context_task.template_file)

    def _build_conversation_context(self, context_task, request_text: str) -> ConversationContext:
        if not context_task:
            return ConversationContext(current_message=request_text)

        lineage = self.task_store.get_task_lineage(context_task.task_id)
        latest_requirement = context_task.requirement or {}
        latest_plan = context_task.artifacts.get("analysis_plan") or {}
        if not latest_plan and latest_requirement:
            try:
                latest_plan = self.requirement_agent.requirement_to_plan(
                    RequirementSpec(raw_request=context_task.request_text, **latest_requirement)
                ).model_dump(mode="json")
            except Exception:
                latest_plan = {}
        inherited_fields = [
            key
            for key in [
                "source_sheet",
                "group_dimensions",
                "metric_targets",
                "stage_value",
                "remark_value",
                "decay_method",
                "statistics",
                "output_granularity",
                "template_sheet_name",
                "threshold_rules",
            ]
            if latest_plan.get(key) not in (None, "", [], {})
        ]
        history_summary: list[str] = []
        for task in lineage[-6:]:
            parts = [f"r{task.revision}", task.status]
            if task.request_text:
                parts.append(f"用户请求：{task.request_text.strip()[:120]}")
            plan_payload = task.artifacts.get("analysis_plan") or {}
            if not plan_payload and task.requirement:
                try:
                    plan_payload = self.requirement_agent.requirement_to_plan(
                        RequirementSpec(raw_request=task.request_text, **task.requirement)
                    ).model_dump(mode="json")
                except Exception:
                    plan_payload = {}
            if plan_payload.get("source_sheet"):
                parts.append(f"sheet={plan_payload['source_sheet']}")
            if plan_payload.get("metric_targets"):
                labels = [item.get("label", "") for item in plan_payload["metric_targets"] if item.get("label")]
                if labels:
                    parts.append(f"指标组={', '.join(labels[:4])}")
            if plan_payload.get("group_dimensions"):
                parts.append(f"分组维度={', '.join(plan_payload['group_dimensions'])}")
            if task.clarifications:
                parts.append(f"待澄清={'; '.join(task.clarifications[:2])}")
            history_summary.append(" | ".join(parts))

        notes = [f"当前输入将作为会话任务 {context_task.task_id[:8]} 的增量修订来理解。"]
        if context_task.input_files:
            notes.append(f"沿用输入文件：{', '.join(Path(path).name for path in context_task.input_files)}")

        return ConversationContext(
            current_message=request_text,
            active_task_id=context_task.task_id,
            history_summary=history_summary,
            latest_confirmed_plan=latest_plan,
            latest_confirmed_requirement=latest_requirement,
            inherited_fields=inherited_fields,
            pending_clarifications=context_task.clarifications,
            notes=notes,
        )

    def _merge_overrides_with_context(self, overrides: dict[str, object], context_task) -> dict[str, object]:
        return dict(overrides)

    def _build_inherited_context_notes(self, context_task) -> list[str]:
        if not context_task:
            return []
        notes = [f"已继承历史会话上下文，基于任务 {context_task.task_id[:8]} 继续分析。"]
        if context_task.input_files:
            notes.append(f"沿用输入文件：{', '.join(Path(path).name for path in context_task.input_files)}")
        plan_payload = context_task.artifacts.get("analysis_plan") or {}
        if plan_payload.get("source_sheet"):
            notes.append(f"沿用 source sheet：{plan_payload['source_sheet']}")
        if plan_payload.get("metric_targets"):
            labels = [item.get("label", "") for item in plan_payload["metric_targets"] if item.get("label")]
            if labels:
                notes.append(f"沿用指标计划：{', '.join(labels[:6])}")
        if plan_payload.get("output_granularity"):
            notes.append(f"沿用输出粒度：{plan_payload['output_granularity']}")
        return notes
