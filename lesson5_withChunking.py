from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("vehicle_insurance.pdf")
print(f"Pages: {len(reader.pages)}")

pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if not text.strip():
        continue
    pages.append(
        Document(
            page_content=text,
            metadata={"source": "vehicle_insurance.pdf", "page": i + 1},
        )
    )

total_chars = sum(len(p.page_content) for p in pages)
print(f"Pages with text: {len(pages)}, total characters: {total_chars}")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)

print(f"Chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print(f"--- page {chunk.metadata['page']} ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content)