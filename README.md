# AI Research Paper Chatbot 🤖

A RAG-based chatbot that answers questions about research papers using Google Gemini AI.

## Tech Stack
- Streamlit
- LangChain
- FAISS
- Google Gemini AI

## Setup
1. Clone the repo
2. Install dependencies:
```
   pip install -r requirements.txt
```
3. Create `.env` file:
```
   GOOGLE_API_KEY=your_key_here
```
4. Run:
```
   python vector_store.py
   streamlit run app.py
```