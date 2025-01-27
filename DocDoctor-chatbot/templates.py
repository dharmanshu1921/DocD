from langchain.prompts import PromptTemplate


def get_standalone_question_prompt():
    """
    Get the prompt template for generating standalone questions.

    Returns:
        PromptTemplate: Template for standalone question generation.
    """
    template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question in its original language.

    Chat History:
    {chat_history}

    Follow-Up Input: {question}
    
    Standalone Question:"""
    return PromptTemplate(input_variables=["chat_history", "question"], template=template)


def get_answer_prompt(language="english"):
    """
    Get the prompt template for generating answers.

    Args:
        language (str): Language for the output answer.

    Returns:
        PromptTemplate: Template for generating answers.
    """
    template = f"""Answer the question at the end using the provided context (delimited by <context></context>).
    Your answer must be in {language}.

    <context>
    {{chat_history}}

    {{context}}
    </context>

    Question: {{question}}
    """
    return PromptTemplate.from_template(template)
