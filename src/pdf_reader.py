import fitz


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = "documents/Hybrid_CNN-Transformer_ECG_Arrhythmia_Research_Paper_Revised.pdf"
    pages = extract_text_from_pdf(pdf_path)

    for page in pages:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:500])