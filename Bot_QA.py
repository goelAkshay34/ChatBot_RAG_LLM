import streamlit as st
import re,time

from langchain.chains import RetrievalQA
#from Api_Key import google_palm
from langchain.llms import GooglePalm
import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("GOOGLE_API_KEY")
google_palm = os.getenv("GOOGLE_API_KEY")


def QA_Bot(vectorstore):
    st.title("Q&A Bot")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        ai_response = Q_A(vectorstore,prompt)
        response = f"Echo: {ai_response}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in re.split(r'(\s+)', response):
                full_response += chunk + " "
                time.sleep(0.01)

                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def Q_A(vectorstore,question):
    google_llm = GooglePalm(google_api_key=google_palm, temperature=0.5)
    qa = RetrievalQA.from_chain_type(llm=google_llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    answer = qa.run(question)

    return answer