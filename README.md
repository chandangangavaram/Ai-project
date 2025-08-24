# Local AI Q&A Bot (Ollama + LangChain)

This project builds a local Retrieval-Augmented Generation (RAG) chatbot that answers questions from uploaded documents. It uses **Ollama** for local LLM inference and **Chroma** for vector storage.

---

## ✅ Features
- Runs entirely offline
- Uses Ollama models (Mistral, LLaMA3, Gemma)
- Document ingestion (PDF, TXT, MD)
- Streamlit UI with chat history & multi-model selection
- Vector store with ChromaDB
- Cited context from retrieved chunks

---

## ✅ Installation

### 1. Install Ollama
Download and install from: [Ollama Website](https://ollama.com/download)

Verify:
```bash
ollama --version
```

## Recommended VS Code Extensions

Install these extensions for the best development experience:

- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Jupyter (ms-toolsai.jupyter)
- Streamlit Snippets (ms-vscode.vscode-streamlit)
- HTML CSS Support (ms-vscode.vscode-html, ms-vscode.vscode-css) *(optional)*
- Black Formatter (ms-python.black-formatter)
- GitLens (eamodio.gitlens)
- Python Environment Manager (donjayamanne.python-environment-manager)

You can search for these in the VS Code Extensions marketplace.
