import streamlit as st
from Bot_QA import QA_Bot
from PDF_Process import PDF_4_QA
from PIL import Image

# Streamlit app
def main():
    

    # Page config
    st.set_page_config(page_title="Q&A ChatBot",
                       layout="wide"
                       )

    st.sidebar.title("Upload PDF")


    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully.")
        vector_store = PDF_4_QA(uploaded_file)
        QA_Bot(vector_store)

if __name__ == '__main__':
    main()