from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pdf_reader import extract_text
import os, time
from dotenv import load_dotenv

load_dotenv()

text = extract_text("paper.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)
print(f"Total chunks: {len(chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Build FAISS index in batches
batch_size = 10
db = None

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    print(f"Embedding batch {i//batch_size + 1}... ({i+len(batch)}/{len(chunks)})")
    
    if db is None:
        db = FAISS.from_texts(batch, embeddings)
    else:
        db.add_texts(batch)
    
    time.sleep(10)  # wait 10 seconds between batches

db.save_local("research_db")
print("✅ research_db created successfully!")