from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")

config = {"configurable": {"thread_id": "chat-1"}}

with SqliteSaver.from_conn_string("memory.db") as checkpointer:
    agent = create_agent(
        model,
        tools=[],
        system_prompt="You are a helpful assistant. Reply in plain text, no Markdown.",
        checkpointer=checkpointer
    )

    # result = agent.invoke({"messages": [HumanMessage("Hi, my name is Hassaan.")]}, config)
    result = agent.invoke({"messages": [HumanMessage("What is my name?")]}, config)

    print(result["messages"][-1].content)
    print(len(result["messages"]))