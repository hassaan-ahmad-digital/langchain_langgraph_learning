from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")

messages = [
    SystemMessage("You are a helpful assistant. Reply in plain text only, no Markdown."),
]

print("Chatbot ready. Type 'quit' to exit.")

while True:
    user_input = input("You: ")

    if user_input == "quit":
        break

    messages.append(HumanMessage(user_input))

    response = model.invoke(messages)

    messages.append(response)

    print(response.content)