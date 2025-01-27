from langchain.schema import Document


def process_patient_data(patient_data):
    """
    Convert patient data into LangChain-compatible documents for vectorstore.

    Args:
        patient_data (dict): Patient data dictionary.

    Returns:
        list: List of LangChain Document objects.
    """
    documents = []
    for key, value in patient_data.items():
        # Flatten nested dictionaries and create documents
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                content = f"{sub_key}: {sub_value}"
                documents.append(Document(page_content=content))
        else:
            content = f"{key}: {value}"
            documents.append(Document(page_content=content))
    return documents


def process_uploaded_documents(uploaded_files):
    """
    Process uploaded files into LangChain-compatible documents.

    Args:
        uploaded_files (list): List of uploaded file objects.

    Returns:
        list: List of LangChain Document objects.
    """
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader

    documents = []

    for uploaded_file in uploaded_files:
        # Save each file temporarily and process
        file_path = f"./data/tmp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Determine the file type and use the appropriate loader
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file format.")

        documents.extend(loader.load())

    return documents
