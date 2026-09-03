from sentence_transformers import SentenceTransformer
from chunker import create_chunks
from pdf_reader import extract_text_from_pdf


def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


if __name__ == "__main__":
    pdf_path = "documents/Hybrid_CNN-Transformer_ECG_Arrhythmia_Research_Paper_Revised.pdf"

    pages = extract_text_from_pdf(pdf_path)

    chunks = create_chunks(pages)

    embeddings = create_embeddings(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")