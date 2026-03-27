import os
from pprint import pprint

def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


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

    user_input = "What's the weather like in Boston?"
    print(f"用户输入: {user_input}")
    print(f"使用模型: {model_name}")

    response = model_with_tools.invoke(user_input)
    print(f"响应对象类型: {type(response).__name__}")
    print(f"模型输出: {response.content}")
    print("模型返回的原始 tool_calls:")
    pprint(response.tool_calls)

    if hasattr(response, "additional_kwargs") and response.additional_kwargs:
        print("additional_kwargs:")
        pprint(response.additional_kwargs)

    if not response.tool_calls:
        print("模型没有触发任何工具调用。")
        return

    for index, tool_call in enumerate(response.tool_calls, start=1):
        print(f"--- tool_call #{index} ---")
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
        if "id" in tool_call:
            print(f"Id: {tool_call['id']}")
        if "type" in tool_call:
            print(f"Type: {tool_call['type']}")

        if tool_call["name"] == "get_weather":
            tool_result = get_weather(**tool_call["args"])  # 模型根据提示词和工具 schema 推断出来的参数
            print(f"Tool Result: {tool_result}")


if __name__ == "__main__":
    main()
