# 🚀 AURA-X Chatbot: Indian Space Policy 2023

An Explainable AI framework for Indian Space Policy compliance, built with Streamlit.

## Features

- **RAG-based Q&A** — Ask questions about the Indian Space Policy 2023
- **Vector Search** — Uses ChromaDB + sentence-transformers for semantic retrieval
- **Source Attribution** — View the exact source passages used to generate answers
- **Lightweight** — Runs entirely on CPU with GPT-2 text generation

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Vector Store | ChromaDB (in-memory) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Generation | GPT-2 |
| PDF Parsing | pypdf |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run aura_x_chatbot.py
```

## Live Demo

Deployed on [Streamlit Community Cloud](https://share.streamlit.io/).

## License

MIT
