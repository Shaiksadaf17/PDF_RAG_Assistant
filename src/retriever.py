import faiss
from sentence_transformers import SentenceTransformer

from pdf_reader import extract_text_from_pdf
from chunker import create_chunks


PDF_PATH = "documents/Hybrid_CNN-Transformer_ECG_Arrhythmia_Research_Paper_Revised.pdf"


def create_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    return index


def search(query, index, chunks, model, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for distance, position in zip(distances[0], indices[0]):
        results.append({
            "page": chunks[position]["page"],
            "text": chunks[position]["text"],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":

    print("Loading PDF...")

    pages = extract_text_from_pdf(PDF_PATH)
    chunks = create_chunks(pages)

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nCreating FAISS index...")

    index = create_index(chunks, model)

    print(f"Index contains {index.ntotal} vectors.")

    question = input("\nAsk a question about the PDF: ")

    print("\nSearching...")

    results = search(
        question,
        index,
        chunks,
        model,
        top_k=5
    )

    print("\n===== RETRIEVED RESULTS =====")

    for result in results:
        print(f"\n--- Page {result['page']} ---")
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"])