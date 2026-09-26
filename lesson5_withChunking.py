from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

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


for chunk in chunks[:3]:
    print(f"--- page {chunk.metadata['page']} ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content)

lastChunk = chunks[-1]

print(f"--- page {lastChunk.metadata['page']} ({len(lastChunk.page_content)} chars) ---")
print(lastChunk.page_content)