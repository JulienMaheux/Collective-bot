import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="Collective Agreement Chatbot", page_icon="🤖")

st.title("📘 Collective Agreement Chatbot")
st.write("Upload a collective agreement PDF and ask questions about it.")

# API Key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.text_input("Enter your OpenAI API Key:", type="password")
if not openai_api_key:
    st.stop()

# Upload
uploaded_pdf = st.file_uploader("Upload a collective agreement (PDF):", type=["pdf"])

if uploaded_pdf:
    with st.spinner("Processing the PDF..."):
        loader = PyPDFLoader(uploaded_pdf)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(chunks, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    st.success("✅ Chatbot ready! Ask a question below.")

    query = st.text_input("Ask a question about the collective agreement:")
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})
            st.write("### 🤖 Answer:")
            st.write(result["result"])
else:
    st.info("👆 Upload a PDF to begin.")
