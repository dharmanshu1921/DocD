from langchain.chains import ConversationalRetrievalChain
from templates import get_standalone_question_prompt, get_answer_prompt
from memory import create_memory


def create_conversational_chain(llm, retriever, language="english"):
    """
    Create a conversational retrieval chain with memory.

    Args:
        llm: Language model instance.
        retriever: Retriever instance for retrieving relevant documents.
        language (str): Language for the prompts (default: English).

    Returns:
        ConversationalRetrievalChain: LangChain retrieval chain.
    """
    # Get prompts for standalone question generation and answer formatting
    standalone_question_prompt = get_standalone_question_prompt()
    answer_prompt = get_answer_prompt(language=language)

    # Create memory
    memory = create_memory()

    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=standalone_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        return_source_documents=True,
    )

    return chain
