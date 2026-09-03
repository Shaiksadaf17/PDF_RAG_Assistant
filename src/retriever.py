import faiss

from sentence_transformers import SentenceTransformer

from pdf_reader import extract_text_from_pdf
from chunker import create_chunks
from embedder import create_embeddings


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

    for distance, index_position in zip(distances[0], indices[0]):
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":
    pdf_path = "documents/Hybrid_CNN-Transformer_ECG_Arrhythmia_Research_Paper_Revised.pdf"

    pages = extract_text_from_pdf(pdf_path)
    chunks = create_chunks(pages)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = create_embeddings(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    question = input("\nWhat is the beat segmentation procedure? ")

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