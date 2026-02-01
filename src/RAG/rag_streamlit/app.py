import os.path

import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question

# set working dir
working_dir = os.path.dirname(os.path.abspath(__file__))

st.title(" Document RAG ")

uploader_file = st.file_uploader("Upload a pdf file", type=["pdf"])

if uploader_file is not None:
    # define save path
    save_path = os.path.join(working_dir, uploader_file.name)
    with open(save_path, "wb") as f:
        f.write(uploader_file.getbuffer())

    process_document = process_document_to_chroma_db(uploader_file.name)
    st.info("Document processed succcessfully")

# text widget to user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = answer_question(user_question)

    st.markdown("### Response")
    st.markdown(answer)