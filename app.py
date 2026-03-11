import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.title("AI Research Paper Chatbot")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

db = FAISS.load_local("research_db", embeddings, allow_dangerous_deserialization=True)

# Only fetch 2 chunks instead of 4 (saves tokens)
retriever = db.as_retriever(search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_output_tokens=300  # limit response size
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