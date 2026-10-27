from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Read the PDF
reader = PdfReader("sample.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print("Total characters:", len(text))

# 2. Split the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))
print("\n--- First chunk ---")
print(chunks[0])