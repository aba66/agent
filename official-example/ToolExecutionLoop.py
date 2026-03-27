import os
from pprint import pprint


def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"The current weather in {location} is sunny."


def get_response_text(response) -> str:
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "text"):
        return response.text
    return str(response)


def main():
    try:
        from langchain_deepseek import ChatDeepSeek
    except ModuleNotFoundError:
        print("缺少依赖: langchain_deepseek")
        print("请先安装：pip install langchain-deepseek")
        return

    try:
        from langchain_core.messages import ToolMessage
    except ModuleNotFoundError:
        print("缺少依赖: langchain_core")
        print("请先安装：pip install langchain-core")
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

    messages = [{"role": "user", "content": "What's the weather in Boston and BeiJing?"}]
    print("Step 1: 向模型发送用户消息")
    pprint(messages)

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    print("\nStep 2: 模型返回工具调用")
    print(f"AI 原始回复: {get_response_text(ai_msg)}")
    pprint(ai_msg.tool_calls)

    if not ai_msg.tool_calls:
        print("模型没有触发任何工具调用，流程到此结束。")
        return

    print("\nStep 3: 执行工具，并将结果追加回消息列表")
    for index, tool_call in enumerate(ai_msg.tool_calls, start=1):
        print(f"--- tool_call #{index} ---")
        pprint(tool_call)

        if tool_call["name"] != "get_weather":
            print(f"跳过未知工具: {tool_call['name']}")
            continue

        tool_result = get_weather(**tool_call["args"])
        print(f"Tool Result: {tool_result}")

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        messages.append(tool_message)

    # messages.append({"role": "user", "content": "What's the weather in BeiJing?"})
    print("\nStep 4: 把工具结果回传给模型，生成最终答复")
    final_response = model_with_tools.invoke(messages)

    print(f"最终回复: {get_response_text(final_response)}")


if __name__ == "__main__":
    main()
