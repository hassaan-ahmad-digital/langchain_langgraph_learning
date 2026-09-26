from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

load_dotenv()

model = init_chat_model("groq:openai/gpt-oss-120b", reasoning_effort="low")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def is_useful(chunk) -> bool:
    text = chunk.page_content
    if text.count("Retrieved") >= 2:
        return False
    if text.lower().count("http") >= 3:
        return False
    if len(re.findall(r"^\s*\d{1,3}\.\s", text, flags=re.MULTILINE)) >= 3:
        return False
    return True

def clean(text: str) -> str:
    text = re.sub(r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2} Vehicle insurance - Wikipedia", "", text)
    text = re.sub(r"https://en\.wikipedia\.org/wiki/Vehicle_insurance \d+/\d+", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

reader = PdfReader("vehicle_insurance.pdf")
print(f"Pages: {len(reader.pages)}")

pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if not text.strip():
        continue
    pages.append(
        Document(
            page_content=clean(text),
            metadata={"source": "vehicle_insurance.pdf", "page": i + 1},
        )
    )

total_chars = sum(len(p.page_content) for p in pages)
print(f"Pages with text: {len(pages)}, total characters: {total_chars}")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"Chunks: {len(chunks)}")

chunks = [c for c in chunks if is_useful(c)]
print(f"Chunks after filtering: {len(chunks)}")

store = InMemoryVectorStore(embeddings)
store.add_documents(chunks)
print(f"Indexed {len(chunks)} chunks")

@tool
def search_policy_docs(query: str) -> str:
    """Search the vehicle insurance reference document.
    Use this for any question about vehicle insurance: coverage types,
    legal requirements, history, or how premiums are calculated."""
    results = store.similarity_search_with_score(query, k=4)

    if not results:
        return "No relevant passages found."

    parts = []
    for doc, score in results:
        page = doc.metadata.get("page", "?")
        parts.append(f"[page {page} | score {score:.3f}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)

agent = create_agent(
    model,
    tools=[search_policy_docs],
    system_prompt=(
        "You answer questions about vehicle insurance using the search tool. "
        "Always search before answering. "
        "Answer only from the search results. "
        "If the results do not contain the answer, say you do not know. "
        "Cite the page number for each fact, like (page 4). "
        "Reply in plain text, no Markdown."
    ),
)

# questions = [
#     "When did the UK make car insurance compulsory?",
#     "What is an excess or deductible?",
#     "Who is the current president of Pakistan?",
# ]

# for question in questions:
#     print(f"\nQ: {question}")
#     result = agent.invoke({"messages": [{"role": "user", "content": question}]})
#     print(f"A: {result['messages'][-1].content}")

result = agent.invoke({"messages": [{"role": "user", "content": "What is an excess or deductible?"}]})
for m in result["messages"]:
    m.pretty_print()