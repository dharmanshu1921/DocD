from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY


def create_memory(model_name="gpt-3.5-turbo", max_token_limit=1024):
    """
    Create a memory object for conversation tracking.

    Args:
        model_name (str): Model name for the LLM.
        max_token_limit (int): Maximum token limit for the memory.

    Returns:
        ConversationSummaryBufferMemory: Memory object.
    """
    llm = ChatOpenAI(model_name=model_name, api_key=OPENAI_API_KEY, temperature=0.1)

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=max_token_limit,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )
    return memory
