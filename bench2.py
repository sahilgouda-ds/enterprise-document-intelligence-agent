import time
import statistics
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ef = DefaultEmbeddingFunction()
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("handbook")

questions = [
    "How many casual leaves do employees get?",
    "What is the hotel limit for a trip to Delhi?",
    "Who is the CEO of TechNova?",
    "What is the notice period?",
    "How many work from home days are allowed?",
    "What is the password policy?",
    "How much is the health insurance cover?",
    "When is salary credited?",
    "Who heads the Internal Complaints Committee?",
    "What is the daily food allowance during travel?",
]

# Warm-up (loads the model, not counted)
ef([questions[0]])

embed_times = []
search_times = []

for _ in range(5):
    for q in questions:
        t0 = time.perf_counter()
        emb = ef([q])
        t1 = time.perf_counter()
        collection.query(query_embeddings=emb, n_results=3)
        t2 = time.perf_counter()
        embed_times.append((t1 - t0) * 1000)
        search_times.append((t2 - t1) * 1000)

print(f"Embedding the question: avg {statistics.mean(embed_times):.1f} ms, median {statistics.median(embed_times):.1f} ms")
print(f"Searching ChromaDB:     avg {statistics.mean(search_times):.1f} ms, median {statistics.median(search_times):.1f} ms")