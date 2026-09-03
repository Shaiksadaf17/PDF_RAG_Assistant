from pdf_reader import extract_text_from_pdf


def create_chunks(pages, chunk_size=1000, overlap=200):
    chunks = []

    chunk_id = 0

    for page in pages:
        text = page["text"].strip()
        page_number = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    pdf_path = "documents/Hybrid_CNN-Transformer_ECG_Arrhythmia_Research_Paper_Revised.pdf"

    pages = extract_text_from_pdf(pdf_path)

    chunks = create_chunks(pages)

    print(f"Number of pages: {len(pages)}")
    print(f"Number of chunks: {len(chunks)}")

    for chunk in chunks[:5]:
        print("\n--- Chunk ---")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Page: {chunk['page']}")
        print(chunk["text"])