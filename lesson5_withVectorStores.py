from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

docs = [
    Document(page_content="How do I cancel my insurance policy?", metadata={"source": "faq", "id": 1}),
    Document(page_content="Can I cancel my insurance policy?", metadata={"source": "faq", "id": 2}),
    Document(page_content="What is the process for terminating my coverage?", metadata={"source": "policy", "id": 3}),
    Document(page_content="My car broke down on the motorway.", metadata={"source": "claims", "id": 4}),
    Document(page_content="The capital of France is Paris.", metadata={"source": "random", "id": 5}),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

store = InMemoryVectorStore(embeddings)
store.add_documents(docs)

results = store.similarity_search_with_score("I want to end my policy", k=3)

for doc, score in results:
    print(f"{score:.4f}  [{doc.metadata['source']}]  {doc.page_content}")