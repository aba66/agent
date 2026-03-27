from langgraph.graph import END, START, MessagesState, StateGraph


def get_message_content(message):
    if hasattr(message, "content"):
        return message.content
    if isinstance(message, dict):
        return message.get("content", "")
    return str(message)


def mock_llm(state: MessagesState):
    user_message = get_message_content(state["messages"][-1])
    print(f"[mock_llm] 收到用户消息: {user_message}")
    return {"messages": [{"role": "ai", "content": "hello world"}]}


def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("mock_llm", mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    return graph.compile()


def main():
    print("开始执行 LangGraph 示例...")
    graph = build_graph()
    input_state = {"messages": [{"role": "user", "content": "hi!"}]}
    print(f"输入状态: {input_state}")

    result = graph.invoke(input_state)

    print("图执行完成。")
    print(f"返回结果: {result}")
    print(f"最终 AI 回复: {get_message_content(result['messages'][-1])}")


if __name__ == "__main__":
    main()
