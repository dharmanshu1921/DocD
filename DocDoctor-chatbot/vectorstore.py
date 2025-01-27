from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY


def create_vectorstore(documents, index_name="docdoctor-faiss-index"):
    """
    Create a FAISS vectorstore index.

    Args:
        documents (list): List of LangChain Document objects.
        index_name (str): Name of the FAISS index file to save locally.

    Returns:
        FAISS: LangChain-compatible FAISS vectorstore.
    """
    # Initialize the OpenAI embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Chunk the documents if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create the FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the FAISS index locally
    vectorstore.save_local(index_name)
    return vectorstore


def load_vectorstore(index_name="docdoctor-faiss-index"):
    """
    Load an existing FAISS vectorstore index.

    Args:
        index_name (str): Name of the FAISS index file to load.

    Returns:
        FAISS: LangChain-compatible FAISS vectorstore.
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return FAISS.load_local(index_name, embeddings)


def get_retriever(vectorstore, k=10):
    """
    Create a retriever from the vectorstore.

    Args:
        vectorstore (FAISS): LangChain-compatible FAISS vectorstore.
        k (int): Number of documents to retrieve.

    Returns:
        Retriever: LangChain retriever instance.
    """
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
