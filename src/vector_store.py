import faiss
import numpy as np

from embedder import create_embeddings
from chunker import create_chunks
from pdf_reader import extract_text_from_pdf


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    return index


if __name__ == "__main__":
    pdf_path = "documents/Hybrid_CNN-Transformer_ECG_Arrhythmia_Research_Paper_Revised.pdf"

    pages = extract_text_from_pdf(pdf_path)
    chunks = create_chunks(pages)

    embeddings = create_embeddings(chunks)

    index = create_faiss_index(embeddings)

    print(f"Number of vectors in index: {index.ntotal}")
    print(f"Vector dimension: {index.d}")