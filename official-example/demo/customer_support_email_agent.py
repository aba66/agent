"""LangGraph customer support email agent demo."""

from __future__ import annotations

import operator
import os
from dataclasses import asdict, dataclass
from pprint import pprint
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

try:
    from langchain.tools import tool
    from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
    from langchain_deepseek import ChatDeepSeek
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "pip install langgraph langchain langchain-core langchain-deepseek pydantic\n"
        f"原始错误: {exc}"
    ) from exc

from support_knowledge_base import SUPPORT_DOCUMENTS


class EmailClassification(BaseModel):
    """Structured result returned by the LLM for the incoming email."""

    urgency: Literal["low", "medium", "high", "urgent"] = Field(
        description="How quickly support should react."
    )
    topic: Literal[
        "account",
        "bug_report",
        "billing",
        "feature_request",
        "technical_issue",
        "general_question",
    ] = Field(description="Main support topic of the email.")
    sentiment: Literal["calm", "frustrated", "angry", "neutral"] = Field(
        description="Customer emotion inferred from the email."
    )
    summary: str = Field(description="A short one-sentence summary of the issue.")
    needs_document_search: bool = Field(
        description="Whether the agent should search documentation before replying."
    )
    needs_human_escalation: bool = Field(
        description="Whether a human support teammate should take over or help."
    )
    needs_follow_up: bool = Field(
        description="Whether the case should be scheduled for a later follow-up."
    )
    search_query: str = Field(description="Best search query for the internal docs.")


@dataclass
class DocumentMatch:
    """A lightweight result object for retrieved support documents."""

    id: str
    topic: str
    title: str
    score: int
    snippet: str


class AgentState(TypedDict):
    """Shared state flowing through the LangGraph nodes."""

    messages: Annotated[list[AnyMessage], operator.add]
    email_text: str
    customer_email: str
    classification: dict[str, Any]
    doc_results: list[dict[str, Any]]
    escalation_record: dict[str, Any] | None
    follow_up_record: dict[str, Any] | None
    final_response: str
    action_log: Annotated[list[str], operator.add]


def print_banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_kv(key: str, value: Any) -> None:
    print(f"{key}: {value}")


def build_model() -> ChatDeepSeek:
    """Create the chat model used by the agent."""

    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-090d98717bb741ecb7f770242743be47")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise SystemExit(
            "没有检测到 DEEPSEEK_API_KEY。\n"
            "请先执行：export DEEPSEEK_API_KEY='你的_key'\n"
            "可选：export DEEPSEEK_MODEL='deepseek-chat'"
        )

    print_banner("模型初始化")
    print_kv("model", model_name)
    return ChatDeepSeek(model=model_name, api_key=api_key, temperature=0)


def tokenize(text: str) -> set[str]:
    """Turn text into a simple lowercase token set for local search."""

    cleaned = text.lower().replace("-", " ").replace("/", " ").replace("_", " ")
    return {token.strip(".,!?:;()[]'\"") for token in cleaned.split() if token.strip()}


@tool
def search_support_docs(query: str, topic: str = "") -> list[dict[str, Any]]:
    """Search the local support knowledge base with a tiny keyword matcher."""
    """
    optimize: 
        Retry strategy: with exponential backoff for transient failures
        Caching: Could cache common queries to reduce API calls
    """

    print_banner("工具执行: search_support_docs")
    print_kv("query", query)
    print_kv("topic", topic or "<not provided>")

    query_tokens = tokenize(query)
    matches: list[DocumentMatch] = []

    for document in SUPPORT_DOCUMENTS:
        if topic and document["topic"] != topic:
            continue

        doc_tokens = tokenize(
            " ".join(
                [
                    document["title"],
                    document["content"],
                    " ".join(document["tags"]),
                    document["topic"],
                ]
            )
        )
        score = len(query_tokens & doc_tokens)
        if score == 0:
            continue

        matches.append(
            DocumentMatch(
                id=document["id"],
                topic=document["topic"],
                title=document["title"],
                score=score,
                snippet=document["content"][:180] + "...",
            )
        )

    matches.sort(key=lambda item: item.score, reverse=True)
    result = [asdict(match) for match in matches[:3]]

    print("命中文档:")
    pprint(result)
    return result


@tool
def escalate_case(summary: str, reason: str, urgency: str) -> dict[str, Any]:
    """Create a mock escalation ticket for a human support teammate."""

    print_banner("工具执行: escalate_case")
    print_kv("summary", summary)
    print_kv("reason", reason)
    print_kv("urgency", urgency)

    ticket = {
        "team": "human_support_queue",
        "priority": urgency,
        "reason": reason,
        "summary": summary,
        "ticket_id": f"ESC-{abs(hash((summary, reason, urgency))) % 100000:05d}",
    }
    print("升级结果:")
    pprint(ticket)
    return ticket


@tool
def schedule_follow_up(customer_email: str, days: int, reason: str) -> dict[str, Any]:
    """Create a mock follow-up task."""

    print_banner("工具执行: schedule_follow_up")
    print_kv("customer_email", customer_email)
    print_kv("days", days)
    print_kv("reason", reason)

    task = {
        "customer_email": customer_email,
        "days": days,
        "reason": reason,
        "task_id": f"FU-{abs(hash((customer_email, days, reason))) % 100000:05d}",
    }
    print("回访任务:")
    pprint(task)
    return task


class CustomerSupportEmailAgent:
    """Encapsulates the graph, model, and helper methods."""

    def __init__(self) -> None:
        self.model = build_model()
        self.classifier = self.model.with_structured_output(EmailClassification)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("classify_email", self.classify_email)
        builder.add_node("search_docs", self.search_docs)
        builder.add_node("handle_case_actions", self.handle_case_actions)
        builder.add_node("draft_response", self.draft_response)

        builder.add_edge(START, "classify_email")
        builder.add_conditional_edges(
            "classify_email",
            self.route_after_classification,
            ["search_docs", "handle_case_actions"],
        )
        builder.add_edge("search_docs", "handle_case_actions")
        builder.add_edge("handle_case_actions", "draft_response")
        builder.add_edge("draft_response", END)
        return builder.compile()

    def classify_email(self, state: AgentState) -> dict[str, Any]:
        """Ask the LLM to classify the customer email."""

        print_banner("节点 1: classify_email")
        print("收到客户邮件:")
        print(state["email_text"])

        result = self.classifier.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a support triage specialist. "
                        "Classify the incoming email carefully and decide "
                        "whether the agent should search docs, escalate, or "
                        "schedule a follow-up."
                    )
                ),
                HumanMessage(content=state["email_text"]),
            ]
        )

        classification = result.model_dump()
        print("分类结果:")
        pprint(classification)

        return {
            "classification": classification,
            "action_log": [
                f"classified topic={classification['topic']} urgency={classification['urgency']}"
            ],
        }

    def route_after_classification(
        self, state: AgentState
    ) -> Literal["search_docs", "handle_case_actions"]:
        """Decide whether we need to search the knowledge base first."""

        print_banner("路由判断: after classify_email")
        needs_search = state["classification"]["needs_document_search"]
        print_kv("needs_document_search", needs_search)

        if needs_search:
            print("下一步: search_docs")
            return "search_docs"

        print("下一步: handle_case_actions")
        return "handle_case_actions"

    def search_docs(self, state: AgentState) -> dict[str, Any]:
        """Search the local support docs for helpful answer context."""

        print_banner("节点 2: search_docs")
        classification = state["classification"]
        docs = search_support_docs.invoke(
            {
                "query": classification["search_query"],
                "topic": classification["topic"],
            }
        )
        return {
            "doc_results": docs,
            "action_log": [f"searched docs with query='{classification['search_query']}'"],
        }

    def handle_case_actions(self, state: AgentState) -> dict[str, Any]:
        """Trigger operational actions such as escalation and follow-up."""

        print_banner("节点 3: handle_case_actions")
        classification = state["classification"]
        updates: dict[str, Any] = {"action_log": []}

        if classification["needs_human_escalation"]:
            escalation_reason = (
                "Urgent customer impact or complex issue requires a human teammate."
            )
            updates["escalation_record"] = escalate_case.invoke(
                {
                    "summary": classification["summary"],
                    "reason": escalation_reason,
                    "urgency": classification["urgency"],
                }
            )
            updates["action_log"].append("escalation ticket created")
        else:
            print("本案无需升级人工。")
            updates["escalation_record"] = None

        if classification["needs_follow_up"]:
            days = {"urgent": 1, "high": 1, "medium": 2, "low": 3}[classification["urgency"]]
            updates["follow_up_record"] = schedule_follow_up.invoke(
                {
                    "customer_email": state["customer_email"],
                    "days": days,
                    "reason": classification["summary"],
                }
            )
            updates["action_log"].append("follow-up scheduled")
        else:
            print("本案无需安排回访。")
            updates["follow_up_record"] = None

        return updates

    def draft_response(self, state: AgentState) -> dict[str, Any]:
        """Generate the final customer-facing reply."""

        print_banner("节点 4: draft_response")
        prompt = self._build_response_prompt(state)
        ai_message = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a thoughtful customer support agent. "
                        "Write a clear, empathetic email reply. "
                        "If the case was escalated, say that the team is reviewing it. "
                        "If a follow-up was scheduled, mention that the customer will hear back."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )

        print("最终回复草稿:")
        print(ai_message.content)

        return {
            "messages": [AIMessage(content=ai_message.content)],
            "final_response": ai_message.content,
            "action_log": ["drafted final response"],
        }

    def _build_response_prompt(self, state: AgentState) -> str:
        """Collect all current state into one drafting prompt."""

        lines = [
            "Incoming customer email:",
            state["email_text"],
            "",
            "Classification:",
            str(state["classification"]),
            "",
            "Retrieved docs:",
            str(state.get("doc_results", [])),
            "",
            "Escalation record:",
            str(state.get("escalation_record")),
            "",
            "Follow-up record:",
            str(state.get("follow_up_record")),
            "",
            "Write the final reply in a professional support tone.",
        ]
        return "\n".join(lines)

    def run(self, email_text: str, customer_email: str = "customer@example.com") -> dict[str, Any]:
        """Execute the graph from start to finish."""

        print_banner("开始运行 Customer Support Email Agent")
        initial_state: AgentState = {
            "messages": [HumanMessage(content=email_text)],
            "email_text": email_text,
            "customer_email": customer_email,
            "classification": {},
            "doc_results": [],
            "escalation_record": None,
            "follow_up_record": None,
            "final_response": "",
            "action_log": [],
        }

        final_state = self.graph.invoke(initial_state)

        print_banner("运行结束: Final State Summary")
        print_kv("classification", final_state["classification"])
        print_kv("doc_results", final_state["doc_results"])
        print_kv("escalation_record", final_state["escalation_record"])
        print_kv("follow_up_record", final_state["follow_up_record"])
        print_kv("action_log", final_state["action_log"])

        return final_state


EXAMPLE_EMAILS = {
    "password": "How do I reset my password? I cannot log in to my account.",
    "bug": "The export feature crashes when I select PDF format on Windows 11.",
    # "billing": "I was charged twice for my subscription today. Please fix this urgently.",
    "billing": "I was charged thrice for my subscription today.",
    "feature": "Can you add dark mode to the mobile app? It would help our team a lot.",
    "api": "Our API integration fails intermittently with 504 errors in production.",
}


def main() -> None:
    """Simple CLI entry so the demo is easy to run and inspect."""

    import argparse

    parser = argparse.ArgumentParser(description="Run the customer support email agent demo.")
    parser.add_argument(
        "--scenario",
        choices=sorted(EXAMPLE_EMAILS.keys()),
        default="billing",
        help="Which built-in email example to run.",
    )
    parser.add_argument(
        "--email",
        default="learner@example.com",
        help="Customer email address used for the follow-up task demo.",
    )
    args = parser.parse_args()

    email_text = EXAMPLE_EMAILS[args.scenario]
    agent = CustomerSupportEmailAgent()
    final_state = agent.run(email_text=email_text, customer_email=args.email)

    print_banner("最终客户回复")
    print(final_state["final_response"])


if __name__ == "__main__":
    main()

