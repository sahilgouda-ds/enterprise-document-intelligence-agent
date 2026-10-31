import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai

load_dotenv()
client = genai.Client()
app = FastAPI(title="Enterprise Document Intelligence Agent")

# Load the PDF, split it, and store the chunks in ChromaDB (runs once at startup)
reader = PdfReader("sample.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(text)

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("handbook")
collection.upsert(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection.count()}


@app.post("/ask")
async def ask(body: Question):
    # Run the (blocking) Chroma search in a thread so the server stays responsive
    results = await asyncio.to_thread(
        collection.query, query_texts=[body.question], n_results=3
    )
    docs = results["documents"][0]
    context = "\n\n".join(docs)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {body.question}"""

    response = await client.aio.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    return {
        "answer": response.text,
        "sources": docs,
    }