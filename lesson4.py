from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")

agent = create_agent(
    model,
    tools=[],
    system_prompt="You are a helpful assistant. Reply in plain text, no Markdown.",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "chat-1"}}

# result1 = agent.invoke({"messages": [HumanMessage("Hi, my name is Hassaan.")]}, config)
# print(result1["messages"][-1].content)

result2 = agent.invoke({"messages": [HumanMessage("What is my name?")]}, config)
print(result2["messages"][-1].content)

print(len(result2["messages"])) # predict 4 when only 1 config is used, 2 when seperate configs