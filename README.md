# 📄 PDF RAG Assistant

<div align="center">

### Ask questions. Retrieve relevant context. Get grounded answers.

A lightweight **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions about their contents using natural language.

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)
![Llama](https://img.shields.io/badge/Llama_3.2-3B-7F52FF?style=for-the-badge)

</div>

---

## ✨ Overview

**PDF RAG Assistant** is a question-answering application built using a Retrieval-Augmented Generation pipeline.

The application allows a user to:

- 📄 Upload a PDF document
- ✂️ Split the document into smaller chunks
- 🧠 Generate embeddings for the chunks
- 🔎 Retrieve the most relevant sections using FAISS
- 🤖 Generate an answer using Llama 3.2
- 📚 Display the source pages used for the answer

The goal is to provide answers that are **grounded in the uploaded document** rather than relying only on the language model's general knowledge.

---

## 🔄 How It Works

```text
                📄 PDF
                  │
                  ▼
        ┌──────────────────┐
        │   PDF Extraction  │
        │     PyMuPDF       │
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │      Chunking     │
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │    Embeddings     │
        │ Sentence          │
        │ Transformers      │
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │       FAISS       │
        │   Vector Index    │
        └─────────┬────────┘
                  │
                  │
          💬 User Question
                  │
                  ▼
        ┌──────────────────┐
        │    Retrieval      │
        │ Relevant Chunks   │
        └─────────┬────────┘
                  │
                  ▼
        ┌──────────────────┐
        │    Llama 3.2      │
        │     Ollama        │
        └─────────┬────────┘
                  │
                  ▼
             🤖 Answer
                  │
                  ▼
             📚 Sources
