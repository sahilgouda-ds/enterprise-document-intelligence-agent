from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai

load_dotenv()
client = genai.Client()

# 1. Read the PDF and split into chunks
reader = PdfReader("sample.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(text)

# 2. Store the chunks in ChromaDB
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("handbook")
collection.upsert(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

# 3. The RAG function: retrieve, then generate
def ask(question):
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    return response.text

# 4. Try it
questions = [
    "How many casual leaves do employees get?",
    "What is the hotel limit for a trip to Delhi?",
    "Who is the CEO of TechNova?",
    "What is the company's refund policy?",  # not in the document
]

for q in questions:
    print("Q:", q)
    print("A:", ask(q))
    print()