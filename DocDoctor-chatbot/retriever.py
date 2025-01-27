def ask_question(retriever, query):
    documents = retriever.get_relevant_documents(query)
    if documents:
        return documents[0].page_content
    return "No relevant data found."
