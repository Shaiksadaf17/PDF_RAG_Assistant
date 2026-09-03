import fitz
import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"


# Load embedding model once
_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return _embedding_model


def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    doc.close()

    return pages


def create_chunks(pages, chunk_size=500, overlap=80):
    chunks = []

    for page in pages:
        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            text = " ".join(words[start:end])

            if text.strip():
                chunks.append({
                    "text": text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


def create_faiss_index(chunks):

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def retrieve(question, chunks, index, top_k=5):

    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        results.append({
            "text": chunks[index_number]["text"],
            "page": chunks[index_number]["page"],
            "score": float(score)
        })

    return results


def generate_answer(question, results):

    context = "\n\n".join(
        f"[Page {result['page']}]\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are a PDF question-answering assistant.

Answer the question using ONLY the information
contained in the provided PDF context.

Do not invent information.

If the answer cannot be found in the context,
say:

"I could not find this information in the uploaded PDF."

Give a clear and concise answer.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def answer_question(pdf_path, question):

    pages = extract_pdf(pdf_path)

    chunks = create_chunks(pages)

    index = create_faiss_index(chunks)

    results = retrieve(
        question,
        chunks,
        index,
        top_k=5
    )

    answer = generate_answer(
        question,
        results
    )

    sources = [
        {
            "page": result["page"],
            "score": result["score"]
        }
        for result in results
    ]

    return answer, sources