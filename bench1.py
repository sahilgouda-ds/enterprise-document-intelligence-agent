import time
import statistics
import chromadb

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("handbook")
print("Chunks in DB:", collection.count())

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

# Warm-up run (the first query loads the embedding model, so we don't count it)
collection.query(query_texts=[questions[0]], n_results=3)

times = []
for _ in range(5):
    for q in questions:
        start = time.perf_counter()
        collection.query(query_texts=[q], n_results=3)
        times.append((time.perf_counter() - start) * 1000)

print("Queries run:", len(times))
print(f"Average: {statistics.mean(times):.1f} ms")
print(f"Median:  {statistics.median(times):.1f} ms")
print(f"Slowest: {max(times):.1f} ms")