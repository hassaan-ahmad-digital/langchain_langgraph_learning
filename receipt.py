from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")

class LineItem(BaseModel):
    """A single product line on the receipt."""
    name: str = Field(description="Product name as printed")
    quantity: int = Field(default=1, description="How many units, 1 if not stated")
    unit_price: float = Field(description="Price per single unit.")

class Receipt(BaseModel):
    """A shop receipt."""
    store_name: str | None = Field(default=None, description="Name of the shop")
    date: str | None = Field(default=None, description="Date on the receipt in YYYY-MM-DD format")
    items: list[LineItem] = Field(default_factory=list, description="All product lines")
    total: float | None = Field(default=None, description="Total amount paid")

structured_model = model.with_structured_output(Receipt)

receipt_text = """
GREEN MART
12 March 2026

Milk 2L        x2     450.00
Bread                 180.00
Eggs (dozen)   x1     520.00
Tea bags       x3     300.00

TOTAL               2150.00
"""

messages = [
    SystemMessage("Extract the receipt data. Do not reply conversationally."),
    HumanMessage(receipt_text),
]

try:
    result = structured_model.invoke(messages)
    print(result)

    if result.total is None:
        raise ValueError("Total is either not provided.")

    total = 0

    for item in result.items:
        print(f"{item.name:<20} x{item.quantity:<3} @{item.unit_price:>8.2f}")
        # name_gap = " " * (20 - len(item.name))
        # print(item.name, name_gap, "  x",item.quantity,"  @",item.unit_price)

        total += item.quantity * item.unit_price

    total_difference = abs(total - result.total)

    if total_difference < 0.01:
        print("Total is accurate")
    else:
        print("Total is inacurate. Accurate total is: ",total, " while, receipt shows: ", result.total, ". Difference is of ", total_difference)

except Exception as error:
    print("Error: ", error)
