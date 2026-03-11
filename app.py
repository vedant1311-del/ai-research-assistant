import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("AI Research Paper Chatbot")
st.write("Upload any research paper and ask questions about it.")

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    batch_size = 10
    db = None

    progress = st.progress(0, text="Building knowledge base...")
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if db is None:
            db = FAISS.from_texts(batch, embeddings)
        else:
            db.add_texts(batch)

        batch_num = i // batch_size + 1
        progress.progress(batch_num / total_batches, text=f"Processing chunks... ({min(i + batch_size, len(chunks))}/{len(chunks)})")

        if i + batch_size < len(chunks):
            time.sleep(10)

    progress.empty()
    return db

# Session state to persist the vector store across interactions
if "db" not in st.session_state:
    st.session_state.db = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file is not None:
    # Only rebuild if a new file is uploaded
    if uploaded_file.name != st.session_state.uploaded_filename:
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.db = None

        with st.spinner("Reading PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        if not text.strip():
            st.error("Could not extract text from this PDF. It may be scanned or image-based.")
        else:
            st.info(f"Building knowledge base from **{uploaded_file.name}**... This may take a minute.")
            st.session_state.db = build_vector_store(text)
            st.success("Ready! Ask your questions below.")

if st.session_state.db is not None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    retriever = st.session_state.db.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=300
    )

    prompt = ChatPromptTemplate.from_template("""
Answer briefly based only on the context below.
Context: {context}
Question: {question}
Answer in 2-3 sentences only.
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = st.text_input("Ask about the research paper")

    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.write(answer)

elif uploaded_file is None:
    st.info("👆 Upload a PDF to get started.")