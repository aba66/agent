from dataclasses import dataclass

def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


@dataclass
class FakeResponse:
    content: str
    tool_calls: list[dict]


class FakeModel:
    """A tiny demo model that decides when to call tools."""

    def __init__(self, tools=None):
        self.tools = tools or []

    def bind_tools(self, tools):
        return FakeModel(tools)

    def invoke(self, user_input: str) -> FakeResponse:
        normalized_text = user_input.lower()
        if "weather" in normalized_text and self.tools:
            location = self._extract_location(user_input)
            return FakeResponse(
                content=f"I should call the weather tool for {location}.",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"location": location},
                    }
                ],
            )

        return FakeResponse(content="No tool call is needed.", tool_calls=[])

    @staticmethod
    def _extract_location(user_input: str) -> str:
        marker = " in "
        lowered = user_input.lower()
        if marker in lowered:
            return user_input[lowered.rfind(marker) + len(marker) :].strip(" ?.!") or "unknown"
        return "unknown"


def main():
    model = FakeModel()
    model_with_tools = model.bind_tools([get_weather])

    user_input = "What's the weather like in Boston?"
    print(f"用户输入: {user_input}")

    response = model_with_tools.invoke(user_input)
    print(f"模型输出: {response.content}")

    if not response.tool_calls:
        print("模型没有触发任何工具调用。")
        return

    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")

        if tool_call["name"] == "get_weather":
            tool_result = get_weather(**tool_call["args"])
            print(f"Tool Result: {tool_result}")


if __name__ == "__main__":
    main()
