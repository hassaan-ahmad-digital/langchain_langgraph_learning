from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")

@tool
def divide(numerator: float, denominator: float) -> str:
    """Divide one number by another and return the exact result."""
    if denominator == 0:
        return "Error: cannot divide by zero."
    return str(numerator / denominator)

@tool
def multiply(val1: float, val2: float) -> float:
    """Multiply one number by another and return the exact result."""
    return val1 * val2

tools = [divide, multiply]

tools_by_name = { t.name: t for t in tools }

model_with_tools = model.bind_tools(tools)

prompt = "What is (((6000 * 5) / 2) / 2) / 2? Answer using tools"
# prompt = "First multiply 5 by 0 using the tools. Then divide 6000 by that result using the tools. Do not reason about it yourself, just call the tools."

messages = [HumanMessage(prompt)]

tool_call_response = model_with_tools.invoke(messages)
messages.append(tool_call_response)

steps = 0
while tool_call_response.tool_calls and steps < 20:
    steps += 1
    for tool_call in tool_call_response.tool_calls:
        print('---------------------------------')
        print(f"{tool_call['name']}: {tool_call['args']}")
        print('---------------------------------')
        selected = tools_by_name.get(tool_call["name"])
        try:
            if selected is None:
                output = f"Unknown Tool: {tool_call['name']}"
            else:
                output = selected.invoke(tool_call['args'])
        except Exception as error:
            output = f"Tool failed: {error}"

        messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    tool_call_response = model_with_tools.invoke(messages)
    messages.append(tool_call_response)

print("************")
print(tool_call_response.content)
print("************")
print(len(messages))