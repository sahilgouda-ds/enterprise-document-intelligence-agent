import time
import statistics
from functools import lru_cache
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

# 50 queries: the same 10 questions asked 5 times (users repeat questions)
workload = questions * 5

# Warm-up (loads the model, not counted)
ef([questions[0]])


def search_plain(q):
    emb = ef([q])
    return collection.query(query_embeddings=emb, n_results=3)


@lru_cache(maxsize=256)
def cached_embed(q):
    return ef([q])


def search_cached(q):
    return collection.query(query_embeddings=cached_embed(q), n_results=3)


def run(fn):
    times = []
    for q in workload:
        t = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - t) * 1000)
    return times


before = run(search_plain)
cached_embed.cache_clear()
after = run(search_cached)

b = statistics.mean(before)
a = statistics.mean(after)
info = cached_embed.cache_info()

print(f"Without cache: avg {b:.1f} ms")
print(f"With cache:    avg {a:.1f} ms  (hits={info.hits}, misses={info.misses})")
print(f"Improvement:   {(b - a) / b * 100:.1f}%")