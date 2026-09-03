import os
import tempfile

import streamlit as st

from rag import answer_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered",
)


# ============================================================
# SIMPLE UI
# ============================================================

st.title("📄 PDF RAG Assistant")
st.write("Upload a PDF and ask questions about its contents.")


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)


if uploaded_file:

    # --------------------------------------------------------
    # Save uploaded PDF temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as tmp_file:

        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    st.success(f"✓ {uploaded_file.name} uploaded")


    # ========================================================
    # QUESTION
    # ========================================================

    question = st.text_input(
        "Ask a question about your PDF",
        placeholder="e.g. What is the beat segmentation procedure?",
    )


    if question.strip():

        with st.spinner("Searching the PDF and generating answer..."):

            try:

                answer, sources = answer_question(
                    pdf_path,
                    question,
                )

            except Exception as e:

                st.error(f"Error while processing the question: {e}")
                answer = None
                sources = []


        # ====================================================
        # ANSWER
        # ====================================================

        if answer:

            st.subheader("🤖 AI Assistant")

            st.markdown(answer)


            # =================================================
            # SOURCES
            # =================================================

            if sources:

                st.subheader("📚 Sources")

                for source in sources:

                    if isinstance(source, dict):

                        page = source.get("page", "Unknown")

                        st.markdown(
                            f"📄 **Page {page}**"
                        )

                    else:

                        st.markdown(
                            f"📄 **{source}**"
                        )


    # ========================================================
    # CLEAN UP TEMPORARY PDF
    # ========================================================

    if os.path.exists(pdf_path):
        try:
            os.unlink(pdf_path)
        except PermissionError:
            pass