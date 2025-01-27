import re


def clean_text(text):
    """
    Clean and preprocess text for uniform formatting.

    Args:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\s+", " ", text)  # Remove excessive whitespace
    text = text.strip()  # Remove leading and trailing spaces
    return text


def format_documents(documents, separator="\n\n"):
    """
    Format a list of documents for display.

    Args:
        documents (list): List of LangChain Document objects.
        separator (str): Separator between documents.

    Returns:
        str: Formatted document content.
    """
    formatted_docs = [f"{doc.page_content}" for doc in documents]
    return separator.join(formatted_docs)


def validate_uploaded_files(files, allowed_extensions=None):
    """
    Validate uploaded files against allowed extensions.

    Args:
        files (list): List of uploaded file objects.
        allowed_extensions (list): Allowed file extensions.

    Returns:
        bool: True if all files are valid, False otherwise.
    """
    if not allowed_extensions:
        allowed_extensions = [".pdf", ".txt", ".csv", ".docx"]

    for file in files:
        if not any(file.name.endswith(ext) for ext in allowed_extensions):
            return False
    return True
