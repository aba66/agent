# Step 1: Define tools and model
#
# 这个文件演示一个最小 LangGraph agent 是怎么工作的：
# 1. 用户先发来一条消息
# 2. LLM 判断自己是否需要调用工具
# 3. 如果需要，代码执行工具
# 4. 工具结果再回给 LLM
# 5. LLM 基于工具结果给出最终答案
#
# 所以这不是“单次问答”，而是一个小循环：
# HumanMessage -> AIMessage(tool call) -> ToolMessage -> AIMessage(final answer)

import operator
import os
from pprint import pprint

from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from typing import Literal
from typing_extensions import Annotated, TypedDict





# 从环境变量读取模型配置。
# 如果你没有提前 export 这些环境变量，这里会使用默认模型名。
api_key = os.getenv("DEEPSEEK_API_KEY", "sk-090d98717bb741ecb7f770242743be47")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 初始化一个聊天模型。
# 此时 model 只是普通模型，还不知道有哪些工具可用。
model = ChatDeepSeek(model=model_name, api_key=api_key)


# Define tools
# @tool 会把普通 Python 函数包装成“可被模型调用的工具”。
# 对模型来说，它会看到：
# - 工具名
# - 参数名和参数类型
# - docstring 里的说明
# 然后由模型决定要不要调用它。
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# 把工具收集起来。
# tools_by_name 用于后面收到 tool_call 后，按名字找到对应工具函数。
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}

# bind_tools 后，模型就具备“返回 tool_calls”的能力了。
# 也就是说，它可以不直接回答，而是先说：
# “请帮我调用 add(a=3, b=4)”
model_with_tools = model.bind_tools(tools)


# Step 2: Define state
#
# LangGraph 的每个节点都在读写“状态”。
# 这里我们定义整张图流转的状态结构。
class MessagesState(TypedDict):
    # messages 是一个消息列表，里面会逐步累积：
    # HumanMessage / AIMessage / ToolMessage / AIMessage ...
    #
    # Annotated[list[AnyMessage], operator.add] 的重点是 operator.add。
    # 它告诉 LangGraph：
    # 当节点返回 {"messages": [...]} 时，
    # 不要覆盖旧 messages，而要做列表拼接。
    #
    # 例如：
    # 旧状态: {"messages": [HumanMessage("Add 3 and 4.")]}
    # 新返回: {"messages": [AIMessage("我准备调用 add")]}
    # 合并后: {"messages": [HumanMessage(...), AIMessage(...)]}
    messages: Annotated[list[AnyMessage], operator.add]

    # 额外记录：到目前为止模型被调用了多少次。
    # 这个字段没有指定 operator.add，所以后写入的值会覆盖旧值。
    llm_calls: int

def print_state_snapshot(stage: str, state: MessagesState) -> None:
    """Print the current graph state in a beginner-friendly way."""
    print(f"\n===== {stage} =====")
    print(f"llm_calls: {state.get('llm_calls', 0)}")
    print(f"messages count: {len(state['messages'])}")

    for index, message in enumerate(state["messages"], start=1):
        message_type = type(message).__name__
        content = getattr(message, "content", str(message))
        print(f"[{index}] {message_type}: {content}")

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("    tool_calls:")
            pprint(tool_calls)

# Step 3: Define model node
#
# 这个节点负责“让模型思考一次”。
# 它可能：
# - 直接给最终答案
# - 或者返回 tool_calls，请下一步去执行工具
def llm_call(state: MessagesState):
    """Ask the LLM to answer or request a tool call."""

    print_state_snapshot("Entering llm_call", state)

    # 这里在每次模型调用前都加一条系统提示，
    # 明确要求模型扮演“算术助手”。
    #
    # 传给模型的消息顺序是：
    # [SystemMessage, 历史消息1, 历史消息2, ...]
    ai_message = model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + state["messages"]
    )

    print("\nllm_call produced a new AIMessage:")
    print(f"type: {type(ai_message).__name__}")
    print(f"content: {ai_message.content}")
    if ai_message.tool_calls:
        print("tool_calls returned by the model:")
        pprint(ai_message.tool_calls)

    # 节点返回的是“局部状态更新”。
    # 由于 messages 使用 operator.add 合并，
    # 所以这里返回的 [ai_message] 会被追加到现有 messages 后面。
    return {
        "messages": [ai_message],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Step 4: Define tool node
#
# 这个节点负责执行模型刚刚请求的工具。
def tool_node(state: MessagesState):
    """Execute the tool calls requested by the latest AI message."""

    print_state_snapshot("Entering tool_node", state)

    result = []

    # 按约定，最后一条消息应该是刚刚产生的 AIMessage。
    # 如果模型决定调用工具，那么这条消息里会带 tool_calls。
    #
    # 每个 tool_call 大致长这样：
    # {"name": "add", "args": {"a": 3, "b": 4}, "id": "..."}
    for tool_call in state["messages"][-1].tool_calls:
        print("\nExecuting tool_call:")
        pprint(tool_call)
        tool = tools_by_name[tool_call["name"]]

        # 真正调用本地 Python 工具函数。
        observation = tool.invoke(tool_call["args"])
        print(f"tool result: {observation}")

        # 把工具执行结果包装成 ToolMessage。
        # tool_call_id 的作用是告诉模型：
        # “这个工具结果对应你刚才发起的哪一次调用”
        result.append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"])
        )

    # 返回的 ToolMessage 也会被追加到 messages 末尾。
    return {"messages": result}


# Step 5: Define logic to determine whether to end
#
# 这个函数控制图的分支走向：
# - 如果模型还要调用工具，就去 tool_node
# - 如果模型已经给出了最终答案，就结束
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Route to the tool node if the model requested a tool call."""

    print_state_snapshot("Evaluating should_continue", state)

    messages = state["messages"]
    last_message = messages[-1]

    # 如果最后一条 AIMessage 里有 tool_calls，
    # 说明模型还没完成任务，而是在请求外部工具帮助。
    if last_message.tool_calls:
        print("Route decision: tool_node")
        return "tool_node"

    # 没有 tool_calls，说明模型已经完成最终回答。
    print("Route decision: END")
    return END


# Step 6: Build agent
#
# 这里真正把节点和边连成一张图。
agent_builder = StateGraph(MessagesState)

# 图里有两个节点：
# - llm_call: 模型思考
# - tool_node: 执行工具
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 连线逻辑：
# START -> llm_call
# llm_call -> tool_node 或 END
# tool_node -> llm_call
#
# 所以整体会形成一个循环：
# 用户消息 -> 模型 -> 工具 -> 模型 -> 工具 -> ... -> 结束
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")

# compile 后得到一个真正可执行的 agent
agent = agent_builder.compile()


# 显示图结构，帮助理解流程。
# 如果在非 notebook 环境运行，这段图像显示可能不会生效，但不影响 agent 本身。
from IPython.display import Image, display

display(Image(agent.get_graph(xray=True).draw_mermaid_png()))


# Invoke
#
# 初始输入只有一条用户消息。
# 注意：agent.invoke 返回的是“最终状态字典”，不是单独一条回复。
messages = [HumanMessage(content="Add 3 and 4.")]
final_state = agent.invoke({"messages": messages})

print_state_snapshot("Final state after agent.invoke", final_state)

# 打印整个执行过程累计下来的所有消息。
# 常见顺序如下：
# 1. HumanMessage: Add 3 and 4.
# 2. AIMessage: 我准备调用 add
# 3. ToolMessage: 7
# 4. AIMessage: 最终答案是 7
for message in final_state["messages"]:
    message.pretty_print()
