import os


def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"The current weather in {location} is sunny."


def main():
    try:
        from langchain_deepseek import ChatDeepSeek
    except ModuleNotFoundError:
        print("缺少依赖: langchain_deepseek")
        print("请先安装：pip install langchain-deepseek")
        return

    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-090d98717bb741ecb7f770242743be47")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        print("缺少 DeepSeek API Key。")
        print("请先在终端设置环境变量，再运行脚本：")
        print("export DEEPSEEK_API_KEY='你的_api_key'")
        print("可选：export DEEPSEEK_MODEL='deepseek-chat'")
        return

    model = ChatDeepSeek(model=model_name, api_key=api_key)
    model_with_tools = model.bind_tools([get_weather])

    user_input = "What's the weather in Boston and Tokyo?"
    print(f"用户输入: {user_input}")
    print("开始流式接收 tool call chunks...\n")

    for chunk in model_with_tools.stream(user_input):
        if not hasattr(chunk, "tool_call_chunks") or not chunk.tool_call_chunks:
            continue

        for tool_chunk in chunk.tool_call_chunks:
            if name := tool_chunk.get("name"):
                print(f"Tool: {name}")
            if id_ := tool_chunk.get("id"):
                print(f"ID: {id_}")
            if args := tool_chunk.get("args"):
                print(f"Args: {args}")


if __name__ == "__main__":
    main()
