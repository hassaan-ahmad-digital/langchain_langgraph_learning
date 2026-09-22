from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

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

agent = create_agent(model, tools)

# prompt = "What is (((6000 * 5) / 2) / 2) / 2? Answer using tools"
prompt = "First multiply 5 by 0 using the tools. Then divide 6000 by that result using the tools. Do not reason about it yourself, just call the tools."

try:
    result = agent.invoke({"messages": [HumanMessage(prompt)]})
    for message in result["messages"]:
        message.pretty_print()
except Exception as error:
    print("There was an error in agent call: ", error)
