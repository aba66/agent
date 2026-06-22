from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_analyst.configuration import ConfigManager
from test_analyst.permissions import PermissionManager
from test_analyst.streamlit_payloads import deserialize_workflow_result
from test_analyst.utils import write_json
from test_analyst.workflow import TestAnalystWorkflow


st.set_page_config(page_title="Test Analyst Enhanced", layout="wide")
st.title("对话式测试分析助手")
st.caption("以 LLM 会话理解为核心，结合本地工具执行测试数据解析、统计、导出和审计。")

workflow = TestAnalystWorkflow(project_root=PROJECT_ROOT)
config_manager = ConfigManager(PROJECT_ROOT)
permission_manager = PermissionManager(config_manager)

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "active_task_id" not in st.session_state:
    st.session_state.active_task_id = None
if "draft_session_title" not in st.session_state:
    st.session_state.draft_session_title = "新对话"
if "draft_session_title_input" not in st.session_state:
    st.session_state.draft_session_title_input = st.session_state.draft_session_title
if "attached_data_files" not in st.session_state:
    st.session_state.attached_data_files = []
if "attached_template_file" not in st.session_state:
    st.session_state.attached_template_file = None
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "session_selector" not in st.session_state:
    st.session_state.session_selector = "__new__"
if "session_selector_input" not in st.session_state:
    st.session_state.session_selector_input = st.session_state.session_selector
if "llm_base_url" not in st.session_state:
    st.session_state.llm_base_url = config_manager.llm_settings().base_url or ""
if "llm_model" not in st.session_state:
    st.session_state.llm_model = config_manager.llm_settings().model or ""
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = config_manager.llm_settings().api_key or ""
if "llm_thinking_mode" not in st.session_state:
    st.session_state.llm_thinking_mode = "auto"
if "active_run" not in st.session_state:
    st.session_state.active_run = None
if "run_feedback" not in st.session_state:
    st.session_state.run_feedback = None


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    temp_dir = PROJECT_ROOT / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    payload = uploaded_file.getvalue()
    digest = hashlib.sha1(payload).hexdigest()[:12]
    path = temp_dir / f"{Path(uploaded_file.name).stem}-{digest}{suffix}"
    if not path.exists():
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        temp_path.replace(path)
    return path


def to_streamlit_safe_dataframe(data) -> pd.DataFrame:
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.columns = [str(column) for column in frame.columns]
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            frame[column] = series.astype("string").fillna("")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            frame[column] = series.where(series.notna(), "").astype("string")
    return frame


def parse_threshold_rules(raw_text: str) -> dict[str, str]:
    rules: dict[str, str] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            rules[key.strip()] = value.strip()
    return rules


def summarize_task_response(task) -> str:
    if task.status == "needs_clarification":
        if task.clarifications:
            return "需要澄清：\n" + "\n".join(f"- {item}" for item in task.clarifications)
        return "需要进一步澄清后再执行。"
    if task.status == "canceled":
        return "本轮处理已终止，系统不会继续分析这条需求。"
    if task.status == "failed":
        if task.errors:
            return "执行失败：\n" + "\n".join(f"- {item}" for item in task.errors)
        return "执行失败。"

    plan = (task.artifacts or {}).get("analysis_plan") or {}
    notes: list[str] = []
    if plan.get("source_sheet"):
        notes.append(f"数据源：{plan['source_sheet']}")
    if plan.get("metric_targets"):
        labels = [item.get("label", "") for item in plan.get("metric_targets", []) if item.get("label")]
        if labels:
            notes.append("指标组：" + ", ".join(labels[:6]))
    if plan.get("group_dimensions"):
        notes.append("分组维度：" + ", ".join(plan["group_dimensions"]))
    if plan.get("statistics"):
        notes.append("统计项：" + ", ".join(plan["statistics"]))
    if plan.get("decay_method"):
        notes.append(f"衰减公式：{plan['decay_method']}")
    if notes:
        return "已完成本轮分析：\n" + "\n".join(f"- {item}" for item in notes[:5])
    return "已完成本轮分析并生成结果。"


def render_conversation_history(task_records) -> None:
    if not task_records:
        return

    for task in sorted(task_records, key=lambda item: item.revision):
        with st.chat_message("user"):
            st.write(task.request_text)
        with st.chat_message("assistant"):
            st.caption(f"r{task.revision} | {task.status} | {task.updated_at}")
            st.markdown(summarize_task_response(task))


def render_login() -> None:
    st.sidebar.header("用户登录")
    if "user_context" not in st.session_state:
        st.session_state.user_context = None

    if st.session_state.user_context:
        user = st.session_state.user_context
        st.sidebar.success(f"已登录：{user.username} ({user.role})")
        if st.sidebar.button("退出登录"):
            st.session_state.user_context = None
            st.rerun()
        return

    username = st.sidebar.text_input("用户名", value="analyst")
    password = st.sidebar.text_input("密码", type="password", value="analyst123")
    if st.sidebar.button("登录"):
        user = permission_manager.authenticate(username, password)
        if user:
            st.session_state.user_context = user
            st.rerun()
        else:
            st.sidebar.error("用户名或密码错误。")


def sync_draft_session_title_from_input() -> None:
    st.session_state.draft_session_title = st.session_state.draft_session_title_input


def sync_session_selector_from_input() -> None:
    st.session_state.session_selector = st.session_state.session_selector_input

    st.sidebar.caption("演示账号：viewer/viewer123，analyst/analyst123，admin/admin123")


render_login()
user = st.session_state.user_context

def get_session_map() -> dict[str, str | None]:
    sessions = workflow.task_store.list_sessions(limit=30)
    mapping = {"新对话": None}
    for row in sessions:
        mapping[f"{row['title']} | {row['session_id'][:8]}"] = row["session_id"]
    return mapping


def load_active_task_records() -> tuple[list, object | None, dict[str, object] | None]:
    session_records = workflow.task_store.list_sessions(limit=30)
    session_meta = next((row for row in session_records if row["session_id"] == st.session_state.active_session_id), None)
    task_records = (
        workflow.task_store.list_tasks(session_id=st.session_state.active_session_id, limit=50)
        if st.session_state.active_session_id
        else []
    )
    latest_task = task_records[0] if task_records else None
    return task_records, latest_task, session_meta


def derive_session_title(prompt: str) -> str:
    compact = " ".join(prompt.split())
    return compact[:32] or "新对话"


def build_overrides(
    source_sheet_override: str,
    metric_override_text: str,
    stage_value: str,
    remark_value: str,
    decay_method: str,
    aggregate_by_fields: list[str],
    requested_statistics: list[str],
    group_by: list[str],
    output_sheet_name: str,
    template_sheet_name: str,
    threshold_rules_text: str,
    revision_note: str,
    need_chart: bool,
    need_report: bool,
    need_pdf: bool,
    need_docx: bool,
) -> dict[str, object]:
    return {
        "source_sheet": source_sheet_override or None,
        "requested_metrics": [item.strip() for item in metric_override_text.split(",") if item.strip()],
        "stage_value": stage_value or None,
        "remark_value": remark_value or None,
        "decay_method": decay_method or None,
        "aggregate_by_fields": aggregate_by_fields,
        "requested_statistics": requested_statistics,
        "group_by": group_by,
        "output_sheet_name": output_sheet_name,
        "template_sheet_name": template_sheet_name or None,
        "need_chart": need_chart,
        "need_report": need_report,
        "need_pdf": need_pdf,
        "need_docx": need_docx,
        "threshold_rules": parse_threshold_rules(threshold_rules_text),
        "revision_note": revision_note or None,
    }


def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_llm_settings() -> dict[str, object]:
    return {
        "base_url": st.session_state.llm_base_url or None,
        "model": st.session_state.llm_model or None,
        "api_key": st.session_state.llm_api_key or None,
        "enable_thinking": {
            "auto": None,
            "enabled": True,
            "disabled": False,
        }[st.session_state.llm_thinking_mode],
    }


def finalize_background_run() -> None:
    active_run = st.session_state.active_run
    if not active_run:
        return

    pid = int(active_run["pid"])
    result_path = Path(active_run["result_path"])
    if is_process_alive(pid):
        return

    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = deserialize_workflow_result(payload)
        st.session_state.latest_result = result
        st.session_state.active_session_id = result.session_id
        st.session_state.active_task_id = result.task_id
        st.session_state.draft_session_title = active_run["session_title"]
        st.session_state.session_selector = next(
            (label for label, session_id in get_session_map().items() if session_id == result.session_id),
            "新对话",
        )
        st.session_state.run_feedback = {"type": "success", "message": "后台分析已完成。"}
    else:
        st.session_state.run_feedback = {"type": "warning", "message": "后台任务未产出结果，可能已被终止。"}
    st.session_state.active_run = None


def start_background_run(
    *,
    effective_data_files: list[str],
    effective_template_file: str | None,
    prompt: str,
    overrides: dict[str, object],
    latest_session_task,
    session_title: str,
    user,
) -> None:
    session_id = workflow.task_store.get_or_create_session(
        st.session_state.active_session_id,
        session_title,
        user,
    )
    run_id = uuid.uuid4().hex
    temp_dir = PROJECT_ROOT / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    payload_path = temp_dir / f"streamlit-run-{run_id}.json"
    result_path = temp_dir / f"streamlit-result-{run_id}.json"
    payload = {
        "project_root": str(PROJECT_ROOT),
        "output_root": str(PROJECT_ROOT / "output"),
        "data_files": effective_data_files,
        "request_text": prompt,
        "template_file": effective_template_file,
        "overrides": overrides,
        "session_id": session_id,
        "parent_task_id": latest_session_task.task_id if latest_session_task else None,
        "user_context": user.model_dump(mode="json"),
        "session_title": session_title,
        "llm_settings": build_llm_settings(),
    }
    write_json(payload_path, payload)
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "streamlit_task_runner.py"), str(payload_path), str(result_path)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state.active_session_id = session_id
    st.session_state.active_run = {
        "pid": process.pid,
        "payload_path": str(payload_path),
        "result_path": str(result_path),
        "session_id": session_id,
        "session_title": session_title,
        "started_at": time.time(),
        "request_text": prompt,
    }
    st.session_state.run_feedback = {"type": "info", "message": "已启动后台分析，你可以随时停止并重新提交需求。"}


def cancel_background_run() -> None:
    active_run = st.session_state.active_run
    if not active_run:
        return

    pid = int(active_run["pid"])
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except OSError:
            break
        time.sleep(0.1)
        if not is_process_alive(pid):
            break

    workflow.task_store.cancel_running_tasks(active_run["session_id"])
    st.session_state.active_run = None
    st.session_state.run_feedback = {"type": "warning", "message": "已终止当前处理，本轮需求不会继续被分析。"}


def render_result_details(result) -> None:
    if not result:
        return

    with st.expander("查看本轮结构化结果", expanded=result.status != "completed"):
        st.write(f"session_id: `{result.session_id}` | task_id: `{result.task_id}` | revision: `{result.revision}`")
        if result.analysis_plan:
            st.caption("当前执行 AnalysisPlan")
            st.json(result.analysis_plan.model_dump(mode="json"))
        st.caption("兼容视图 RequirementSpec")
        st.json(result.requirement.model_dump(mode="json"))
        if result.conversation_context:
            st.caption("本轮使用的会话上下文")
            st.json(result.conversation_context.model_dump(mode="json"))

    if result.status == "needs_clarification":
        st.warning("当前信息还不足以执行，请继续补充后直接发送下一条消息。")
        for question in result.clarifications:
            st.write(f"- {question}")
        return

    if result.status == "failed":
        st.error("执行失败")
        for error in result.errors:
            st.write(f"- {error}")
        if result.trace_events:
            with st.expander("失败前 Agent Trace", expanded=False):
                st.dataframe(
                    to_streamlit_safe_dataframe([event.model_dump(mode="json") for event in result.trace_events]),
                    width="stretch",
                )
        return

    assert result.analysis is not None
    if result.analysis.validation_messages:
        st.markdown("**校验结果**")
        for message in result.analysis.validation_messages:
            st.write(f"- {message}")

    if result.analysis.warnings:
        st.markdown("**风险与告警**")
        for warning in result.analysis.warnings:
            st.write(f"- {warning}")

    if result.analysis.comparison_notes:
        st.markdown("**多文件说明**")
        for note in result.analysis.comparison_notes:
            st.write(f"- {note}")

    st.markdown("**结果预览**")
    st.dataframe(to_streamlit_safe_dataframe(result.analysis.summary_frame), width="stretch")

    if result.analysis.statistics:
        with st.expander("统计摘要", expanded=False):
            st.json(result.analysis.statistics)

    if result.analysis.chart_figures:
        with st.expander("图表", expanded=False):
            for _, figure in result.analysis.chart_figures.items():
                st.plotly_chart(figure)

    with st.expander("Trace / Agent / MCP", expanded=False):
        st.dataframe(
            to_streamlit_safe_dataframe([event.model_dump(mode="json") for event in result.trace_events]),
            width="stretch",
        )
        if result.agent_cards:
            st.dataframe(
                to_streamlit_safe_dataframe([item.model_dump(mode="json") for item in result.agent_cards]),
                width="stretch",
            )
        if result.a2a_handoffs:
            st.dataframe(
                to_streamlit_safe_dataframe([item.model_dump(mode="json") for item in result.a2a_handoffs]),
                width="stretch",
            )
        if result.mcp_calls:
            st.dataframe(
                to_streamlit_safe_dataframe([item.model_dump(mode="json") for item in result.mcp_calls]),
                width="stretch",
            )

    artifacts = result.analysis.artifacts
    st.markdown("**下载产物**")
    for label, path in [
        ("下载 Excel 结果", artifacts.excel_path),
        ("下载 Markdown 报告", artifacts.markdown_path),
        ("下载 PDF 报告", artifacts.pdf_path),
        ("下载 Docx 报告", artifacts.docx_path),
        ("下载审计日志", artifacts.audit_log_path),
        ("下载审计报告", artifacts.audit_report_path),
    ]:
        if path and Path(path).exists():
            data = Path(path).read_bytes() if Path(path).suffix != ".md" else Path(path).read_text(encoding="utf-8")
            st.download_button(label=label, data=data, file_name=Path(path).name)


finalize_background_run()

llm_defaults = config_manager.llm_settings().masked()
llm_default_model = config_manager.llm_settings().model or ""
llm_default_api_key = config_manager.llm_settings().api_key or ""
thinking_option_labels = {
    "auto": "自动",
    "enabled": "开启",
    "disabled": "关闭",
}
thinking_default_mode = "auto"
if llm_defaults.get("enable_thinking") is True:
    thinking_default_mode = "enabled"
elif llm_defaults.get("enable_thinking") is False:
    thinking_default_mode = "disabled"
if st.session_state.llm_thinking_mode == "auto" and thinking_default_mode != "auto":
    st.session_state.llm_thinking_mode = thinking_default_mode

with st.sidebar:
    if st.session_state.get("draft_session_title_input") != st.session_state.get("draft_session_title"):
        st.session_state.draft_session_title_input = st.session_state.draft_session_title
    if st.session_state.get("session_selector_input") != st.session_state.get("session_selector"):
        st.session_state.session_selector_input = st.session_state.session_selector

    st.header("会话")
    session_map = get_session_map()
    current_session_label = next(
        (label for label, session_id in session_map.items() if session_id == st.session_state.active_session_id),
        "新对话",
    )
    if st.session_state.get("session_selector") not in session_map:
        st.session_state.session_selector = current_session_label
    if st.session_state.get("session_selector_input") not in session_map:
        st.session_state.session_selector_input = current_session_label
    selected_session_label = st.selectbox(
        "切换会话",
        options=list(session_map.keys()),
        key="session_selector_input",
        on_change=sync_session_selector_from_input,
    )
    selected_session_id = session_map[selected_session_label]
    if selected_session_id != st.session_state.active_session_id:
        if st.session_state.active_run:
            cancel_background_run()
        st.session_state.active_session_id = selected_session_id
        st.session_state.active_task_id = None
        st.session_state.latest_result = None
        st.session_state.attached_data_files = []
        st.session_state.attached_template_file = None
        if selected_session_label == "新对话":
            st.session_state.draft_session_title = "新对话"
        else:
            st.session_state.draft_session_title = selected_session_label.split(" | ")[0]
        st.rerun()

    if st.button("新建对话", use_container_width=True):
        if st.session_state.active_run:
            cancel_background_run()
        st.session_state.active_session_id = None
        st.session_state.active_task_id = None
        st.session_state.latest_result = None
        st.session_state.draft_session_title = "新对话"
        st.session_state.attached_data_files = []
        st.session_state.attached_template_file = None
        st.session_state.session_selector = "新对话"
        st.rerun()

    st.text_input(
        "当前会话标题",
        key="draft_session_title_input",
        on_change=sync_draft_session_title_from_input,
    )

    st.header("当前附件")
    data_uploads = st.file_uploader(
        "上传测试数据文件",
        type=["csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="data_files",
    )
    template_upload = st.file_uploader(
        "可选：上传输出模板",
        type=["xlsx", "xlsm"],
        key="template_file",
    )
    if data_uploads:
        st.session_state.attached_data_files = [str(save_uploaded_file(item)) for item in data_uploads]
    if template_upload:
        st.session_state.attached_template_file = str(save_uploaded_file(template_upload))
    if st.button("清空当前附件", use_container_width=True):
        st.session_state.attached_data_files = []
        st.session_state.attached_template_file = None
        st.rerun()

    if st.session_state.attached_data_files:
        st.caption("已附加数据文件：" + ", ".join(Path(path).name for path in st.session_state.attached_data_files))
    if st.session_state.attached_template_file:
        st.caption(f"已附加模板：{Path(st.session_state.attached_template_file).name}")

    st.header("LLM 配置")
    st.text_input("LLM Base URL", key="llm_base_url")
    st.text_input("LLM Model", key="llm_model")
    st.text_input("LLM API Key", type="password", key="llm_api_key")
    st.selectbox(
        "LLM Thinking",
        options=list(thinking_option_labels.keys()),
        index=list(thinking_option_labels.keys()).index(st.session_state.llm_thinking_mode),
        format_func=lambda key: thinking_option_labels[key],
        key="llm_thinking_mode",
        help="对支持的 OpenAI-compatible 提供商传递 enable_thinking。JSON 结构化输出场景建议关闭。",
    )
    llm_available = bool(st.session_state.llm_api_key and st.session_state.llm_model)
    if not llm_available:
        st.error("当前没有可用的 LLM API 配置。该数据分析系统必须依赖 LLM，未配置时无法使用。")

    workbook_context = None
    source_sheet_options = [""]
    active_task_records, latest_session_task, active_session_meta = load_active_task_records()
    inherited_data_files = latest_session_task.input_files if latest_session_task else []
    inherited_template_file = latest_session_task.template_file if latest_session_task else None
    effective_data_files = st.session_state.attached_data_files or inherited_data_files
    effective_template_file = st.session_state.attached_template_file or inherited_template_file
    if effective_data_files:
        workbook_context, _, _ = workflow.inspect_inputs(effective_data_files, template_file=effective_template_file)
        source_sheet_options += workbook_context.sheet_names

    with st.expander("高级补充信息", expanded=False):
        source_sheet_override = st.selectbox("Source sheet", options=source_sheet_options, index=0)
        metric_override_text = st.text_area("手动指定测试项", placeholder="用英文逗号分隔")
        stage_value = st.text_input("阶段值", placeholder="如 EVT / DVT")
        remark_value = st.text_input("remark", placeholder="可选")
        decay_method = st.selectbox(
            "衰减公式",
            options=["", "relative_to_first", "delta_from_first", "delta_from_previous"],
            index=0,
        )
        aggregate_by_fields = st.multiselect(
            "分别统计维度",
            options=["test_item", "sn", "stage", "test_node", "source_file"],
            default=[],
        )
        requested_statistics = st.multiselect(
            "汇总统计项",
            options=["min", "max", "mean", "decay"],
            default=[],
        )
        group_by = st.multiselect("明细分组字段", options=["stage", "test_item", "sn", "test_node", "source_file"], default=[])
        output_sheet_name = st.text_input("输出 sheet 名称", value="KPI Summary")
        template_sheet_name = st.text_input("模板 sheet 名称", value="WG-PRO-KPI Summary")
        threshold_rules_text = st.text_area("阈值规则", placeholder="例如：\nANSIContrast_Res_LvContRatio >= 20")
        revision_note = st.text_input("版本备注", placeholder="本次修订说明，可选")
        need_chart = st.checkbox("生成图表", value=True)
        need_report = st.checkbox("生成 Markdown 报告", value=True)
        need_pdf = st.checkbox("生成 PDF 报告", value=True)
        need_docx = st.checkbox("生成 Docx 报告", value=True)

    if workbook_context:
        with st.expander("输入文件结构摘要", expanded=False):
            for file_context in workbook_context.input_files:
                st.caption(f"{file_context.role} | {file_context.file_path.name}")
                for sheet in file_context.sheet_summaries:
                    st.write(f"sheet={sheet.name} | rows={sheet.rows} | cols={sheet.columns}")
                    st.write("列名：", sheet.column_names[:20])
                    st.dataframe(to_streamlit_safe_dataframe(sheet.preview_records), width="stretch")

task_tab, history_tab, config_tab = st.tabs(["连续对话", "历史版本", "配置管理"])


with task_tab:
    task_records, latest_session_task, active_session_meta = load_active_task_records()
    if active_session_meta and st.session_state.draft_session_title == "新对话":
        st.session_state.draft_session_title = active_session_meta["title"]

    current_title = active_session_meta["title"] if active_session_meta else st.session_state.draft_session_title
    st.subheader(current_title or "新对话")
    if st.session_state.active_session_id:
        st.caption(f"当前活跃会话：{st.session_state.active_session_id[:8]}，直接发送下一条消息即可继续修订结果。")
    else:
        st.caption("当前是新对话。上传文件后直接输入需求，后续消息会默认继续沿用当前线程。")

    effective_data_files = st.session_state.attached_data_files or (latest_session_task.input_files if latest_session_task else [])
    if effective_data_files:
        st.info("当前会话将使用数据文件：" + ", ".join(Path(path).name for path in effective_data_files))
    elif latest_session_task and latest_session_task.input_files:
        st.info("当前会话将沿用历史输入文件：" + ", ".join(Path(path).name for path in latest_session_task.input_files))
    else:
        st.info("请先在左侧上传数据文件，然后直接开始对话。")

    if latest_session_task and latest_session_task.requirement:
        with st.expander("当前会话已确认上下文", expanded=False):
            confirmed_plan = (latest_session_task.artifacts or {}).get("analysis_plan") or {}
            st.json(
                {
                    key: confirmed_plan.get(key)
                    for key in [
                        "source_sheet",
                        "group_dimensions",
                        "metric_targets",
                        "statistics",
                        "decay_method",
                        "output_granularity",
                        "template_sheet_name",
                    ]
                    if confirmed_plan.get(key) not in (None, "", [], {})
                }
            )

    render_conversation_history(task_records)
    latest_result = st.session_state.latest_result
    if latest_result and latest_result.session_id == st.session_state.active_session_id:
        with st.chat_message("assistant"):
            st.caption("当前活跃轮次结果")
            render_result_details(latest_result)

    if st.session_state.run_feedback:
        feedback = st.session_state.run_feedback
        getattr(st, feedback["type"])(feedback["message"])
        st.session_state.run_feedback = None

    active_run = st.session_state.active_run
    if active_run:
        st.warning("当前有任务正在后台处理中。你可以刷新状态，或先终止再发送新的需求。")
        st.caption(f"处理中请求：{active_run['request_text'][:120]}")
        col_refresh, col_cancel = st.columns(2)
        if col_refresh.button("刷新运行状态", use_container_width=True):
            st.rerun()
        if col_cancel.button("停止处理", type="primary", use_container_width=True):
            cancel_background_run()
            st.rerun()

    llm_available = bool(st.session_state.llm_api_key and st.session_state.llm_model)
    run_disabled = not user or not permission_manager.can(user, "run_analysis") or not llm_available
    run_disabled = run_disabled or active_run is not None
    if run_disabled:
        if not llm_available:
            st.info("请先在左侧配置可用的 LLM API Key 和 Model，当前系统不会再使用规则兜底。")
        elif active_run is not None:
            st.info("当前正在处理上一条需求。如需修改，请先点击“停止处理”。")
        else:
            st.info("当前账号没有执行分析权限，请使用 analyst 或 admin 账号登录。")

    prompt = st.chat_input(
        "继续描述、补充或修订你的需求",
        disabled=run_disabled,
    )
    if prompt:
        effective_data_files = st.session_state.attached_data_files or (latest_session_task.input_files if latest_session_task else [])
        if not effective_data_files:
            st.warning("当前还没有可用的数据文件，请先在左侧上传文件。")
        else:
            effective_template_file = st.session_state.attached_template_file or (
                latest_session_task.template_file if latest_session_task else None
            )
            session_title = (st.session_state.draft_session_title or "").strip() or derive_session_title(prompt)
            overrides = build_overrides(
                source_sheet_override,
                metric_override_text,
                stage_value,
                remark_value,
                decay_method,
                aggregate_by_fields,
                requested_statistics,
                group_by,
                output_sheet_name,
                template_sheet_name,
                threshold_rules_text,
                revision_note,
                need_chart,
                need_report,
                need_pdf,
                need_docx,
            )
            start_background_run(
                effective_data_files=effective_data_files,
                effective_template_file=effective_template_file,
                prompt=prompt,
                overrides=overrides,
                latest_session_task=latest_session_task,
                session_title=session_title,
                user=user,
            )
            st.rerun()


with history_tab:
    st.subheader("任务历史与版本")
    sessions = workflow.task_store.list_sessions(limit=20)
    st.dataframe(to_streamlit_safe_dataframe(sessions), width="stretch")

    selected_history_session = None
    if sessions:
        label_map = {f"{item['title']} | {item['session_id'][:8]}": item["session_id"] for item in sessions}
        selected_history_session = label_map[st.selectbox("选择要查看的会话", options=list(label_map.keys()))]

    if selected_history_session:
        task_records = workflow.task_store.list_tasks(session_id=selected_history_session, limit=50)
        st.dataframe(
            to_streamlit_safe_dataframe([task.model_dump(mode="json") for task in task_records]),
            width="stretch",
        )
        if task_records:
            task_label_map = {f"r{task.revision} | {task.status} | {task.task_id[:8]}": task for task in task_records}
            selected_task = task_label_map[st.selectbox("查看版本详情", options=list(task_label_map.keys()))]
            st.json(selected_task.requirement)
            st.subheader("Trace")
            st.dataframe(to_streamlit_safe_dataframe(selected_task.trace_events), width="stretch")
            st.subheader("产物")
            st.json(selected_task.artifacts)


with config_tab:
    st.subheader("配置管理")
    if not user or not permission_manager.can(user, "manage_config"):
        st.info("当前账号没有配置管理权限，请使用 admin 登录。")
    else:
        alias_text = st.text_area(
            "字段与指标别名字典（JSON）",
            value=json.dumps(config_manager.field_aliases(), ensure_ascii=False, indent=2),
            height=280,
        )
        rule_text = st.text_area(
            "业务规则配置（JSON）",
            value=json.dumps(config_manager.business_rules(), ensure_ascii=False, indent=2),
            height=280,
        )
        permission_text = st.text_area(
            "权限配置（JSON）",
            value=json.dumps(config_manager.permissions(), ensure_ascii=False, indent=2),
            height=240,
        )
        if st.button("保存配置"):
            try:
                config_manager.save("field_aliases.json", json.loads(alias_text))
                config_manager.save("business_rules.json", json.loads(rule_text))
                config_manager.save("permissions.json", json.loads(permission_text))
                st.success("配置已保存。重新运行任务即可生效。")
            except Exception as exc:
                st.error(f"保存配置失败：{exc}")
