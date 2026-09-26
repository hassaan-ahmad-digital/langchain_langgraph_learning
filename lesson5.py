from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np  

def cosine_similarity(a,b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How do I cancel my insurance policy?",
    "Can I cancel my insurance policy?",
    "What is the process for terminating my coverage?",
    "My car broke down on the motorway.",
    "The capital of France is Paris.",
]

query = "I want to end my policy"
query_vector = embeddings.embed_query(query)

document_vectors = embeddings.embed_documents(sentences)

scored_sentence_tuples = []

for sentence, vector in zip(sentences, document_vectors):
    score = cosine_similarity(query_vector, vector)

    scored_sentence_tuples.append((score, sentence))

results = sorted(scored_sentence_tuples, key= lambda pair: pair[0], reverse=True)

for score, sentence in results[:3]:
    print(f"{score:.4f}  {sentence}")