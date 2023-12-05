from langchain.document_loaders import DirectoryLoader  # load file from dataloader
from langchain.text_splitter import CharacterTextSplitter  # split data to get chunk
import os
import pinecone
from langchain.vectorstores import Pinecone  # use for vector store
from langchain.embeddings.openai import OpenAIEmbeddings  # open ai embedding
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # chatgpt openAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',  # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    # split the text with chunk
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    # split the docs
    docs_split = text_splitter.split_documents(docs)
    return docs_split


# create embedding for me and store in the first time
@st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    # initiating pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docs_split = doc_preprocessing()
    docs_db = Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name='langchain-demo-indexes'
    )
    return docs_db


llm = ChatOpenAI()
doc_db = embedding_db()


def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # too many checkout
        retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result


def main():
    st.title("Question and Answering App powered by LLM and Pinecone")

    text_input = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)


if __name__ == "__main__":
    main()
