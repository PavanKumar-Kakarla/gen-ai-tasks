import streamlit as st
import faiss
import os
from PyPDF2 import PdfReader
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Chat with your PDF", layout="wide")
st.title("Chat with your PDF")

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "index" not in st.session_state:
    st.session_state["index"] = None
if "sources" not in st.session_state:
    st.session_state["sources"] = {}

# File Uploader
uploaded_files = st.file_uploader("Upload one or more PDF's", type="pdf", accept_multiple_files=True)

chunk_size = st.number_input("Chunk size (tokens approx)", 200, 1000, 500, step=50)
chunk_overlap = st.number_input("Chunk overlap", 0, 200, 50, step=10)
top_k = st.slider("Top-k retrieved chunks", 1, 10, 3)

if uploaded_files and st.button("Ingest PDFs"):
    st.session_state["chunks"].clear()
    st.session_state["sources"].clear()

    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            for start in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[start:start+chunk_size]
                if chunk.strip():
                    st.session_state["chunks"].append(chunk)
                    st.session_state["sources"][len(st.session_state["chunks"]) - 1] = f"{file.name} (page {i+1})"
    
    st.success(f"Extracted {len(st.session_state['chunks'])} chunks from {len(uploaded_files)} PDFs")

    # Create Embeddings
    st.write("Creating Embeddings")
    embeddings = embedder.encode(st.session_state["chunks"], convert_to_numpy=True)

    #Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    st.session_state["embeddings"] = embeddings
    st.session_state["index"] = index
    st.success("Embedding stored in FAISS index")

# Question input
question = st.text_input("Ask a question about your PDF(s)")

if st.button("Ask"):
    if st.session_state["index"] is None:
        st.error("Please upload and ingest PDFs first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        # Adjust top_k to avoid asking more than available chunks
        actual_k = min(top_k, len(st.session_state["chunks"]))

        # Embed query
        query_emb = embedder.encode([question], convert_to_numpy=True)
        D, I = st.session_state["index"].search(query_emb, actual_k)

        # Filter valid indices (avoid -1 errors)
        valid_indices = [i for i in I[0] if i != -1]
        retrieved_chunks = [st.session_state["chunks"][i] for i in valid_indices]
        sources = [st.session_state["sources"][i] for i in valid_indices]

        # Compose context
        context = "\n\n".join([f"Source {j+1} ({sources[j]}): {retrieved_chunks[j]}" 
                               for j in range(len(retrieved_chunks))])

        # Call LLM
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions grounded in the provided documents. Always cite sources."},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content
            st.success(answer)

            # Toggle sources
            if st.checkbox("Show sources"):
                for j, src in enumerate(sources, 1):
                    st.write(f"Source {j}: {src}")

        except Exception as e:
            st.error(f"Error: {str(e)}")


    