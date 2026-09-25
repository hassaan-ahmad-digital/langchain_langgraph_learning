from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, SummarizationMiddleware
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.runtime import Runtime
from langgraph.graph.message import REMOVE_ALL_MESSAGES

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")

chat_name = input("Write chat name to enter it: ")

config = {"configurable": {"thread_id": chat_name}}

summarizer = SummarizationMiddleware(
    model = model,
    trigger = ("tokens", 600),
    keep = ("messages", 4)
)

@before_model
def log_messages(state: AgentState, runtime: Runtime) -> None:
    print(f"[middleware] model is about to see {len(state['messages'])} messages")
    return None

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

with SqliteSaver.from_conn_string("memory.db") as checkpointer:
    agent = create_agent(
        model,
        tools,
        system_prompt="You are a helpful assistant. Reply in plain text, no Markdown.",
        checkpointer=checkpointer,
        middleware=[summarizer, log_messages]
    )

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        try:
            result = agent.invoke({"messages": [HumanMessage(user_input)]}, config)

            print(result["messages"][-1].content)
            print(len(result["messages"]))
        except Exception as error:
            print("Error while invoking the agent: ", error)