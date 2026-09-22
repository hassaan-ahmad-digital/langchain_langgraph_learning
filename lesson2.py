from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")

class Person(BaseModel):
    name: str | None = Field(default=None, description="The person's full name, if mentioned")
    age: int | None = Field(default=None, description="The person's age in years, if mentioned")
    city: str | None = Field(default=None, description="The city where the person lives")
    hobbies: list[str] = Field(default_factory=list, description="The person's hobbies, each as a short phrase starting with a verb ending in -ing, for example: singing")

structured_model = model.with_structured_output(Person)

text = "Hey! Petricia here. I just turned 31. I moved to Sydney last year, I like to sing, paint and make food tutorials on youtube."
# text = "Hello"

messages = [
    SystemMessage("Extract the person's details from the user's text. Do not reply conversationally."),
    HumanMessage(text),
]

try:
    result = structured_model.invoke(messages)
    print(result)

    if result.name is None:
        raise ValueError("No name found in the prompt.")
    
    print(result.name)

    if result.age is not None:
        print(result.age + 1)
    else:
        print("Age not mentioned")

    for hobby in result.hobbies:
        print("Hobby: ", hobby)

    print(type(result))
except Exception as error:
    print("Could not extract details:", error)

