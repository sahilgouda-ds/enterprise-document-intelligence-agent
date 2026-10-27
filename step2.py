from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# 1. Read the PDF and split into chunks (same as step 1)
reader = PdfReader("sample.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(text)

# 2. Create a local ChromaDB database (saved in the chroma_db folder)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("handbook")

# 3. Store the chunks
collection.upsert(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)
print("Chunks stored:", collection.count())

# 4. Ask a question and find the most relevant chunks
question = "How many casual leaves do employees get?"
results = collection.query(query_texts=[question], n_results=2)

print("\nQuestion:", question)
for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- Match {i + 1} ---")
    print(doc)