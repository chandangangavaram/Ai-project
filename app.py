import os
import time
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Directory where Chroma DB is stored
PERSIST_DIR = "chroma_db"

# Streamlit UI Setup
st.set_page_config(page_title="Local RAG Bot", page_icon="📚")
st.title("📚 Local AI Q&A Bot")
st.write("Powered by Ollama + LangChain (Runs locally)")

# Sidebar: Model and Retrieval Settings
model_choice = st.sidebar.selectbox("Select Model", ["mistral", "llama3", "gemma"], index=0)
top_k = st.sidebar.slider("Retriever Top-K", 1, 8, 3)
st.sidebar.caption("Tip: lower K for faster but riskier answers, higher K for more context.")

# load vectorstore
@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="mistral")  # Keep embeddings fixed for retrieval
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

# Initialize LLM and QA Chain
llm = Ollama(model=model_choice)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # Enables citations
)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input and Response Handling
query = st.text_input("Ask a question about your documents:")

if st.button("Submit") and query.strip():
    with st.spinner("Thinking..."):
        start = time.time()
        result = qa_chain({"query": query})
        elapsed = round(time.time() - start, 2)

        answer = result["result"]
        sources = result.get("source_documents", [])

        st.session_state.chat_history.append(
            {"q": query, "a": answer, "t": elapsed, "sources": sources}
        )

# Display Chat History
for idx, item in enumerate(st.session_state.chat_history, start=1):
    st.markdown(f"**Q{idx}:** {item['q']}")
    st.markdown(f"**A{idx}:** {item['a']}")
    st.caption(f"⏱️ Response time: {item['t']} sec | Top-K: {top_k}")
    
    if item["sources"]:
        with st.expander("Show retrieved chunks"):
            for i, doc in enumerate(item["sources"], 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))

