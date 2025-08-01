from typing import Optional
from langchain.prompts import PromptTemplate

def get_standalone_question_prompt(language: str = "english") -> PromptTemplate:
    """
    Generate a prompt template to rephrase a follow-up question as a standalone question.
    
    Uses advanced prompt engineering techniques including role assignment, clear instructions,
    structured thinking, and output formatting to ensure high-quality question rephrasing.
    
    Args:
        language (str): The language in which the question should be rephrased (default: "english").
        
    Returns:
        PromptTemplate: A LangChain PromptTemplate for standalone question generation.
        
    Raises:
        ValueError: If language is not a string or is empty.
    """
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Language must be a non-empty string")
    
    template = """You are an expert communication specialist skilled at transforming conversational questions into clear, standalone queries.

Your task is to analyze the chat history and follow-up question, then rephrase the follow-up question into a completely self-contained question that captures all necessary context and intent.

<requirements>
- The output must be in {language}
- Preserve all essential information from the chat history
- Make the question understandable without any prior context
- Maintain the original question's intent and scope
- Use clear, precise language
</requirements>

<thinking_process>
1. Identify key entities, concepts, and context from the chat history
2. Determine what information the follow-up question assumes from previous conversation
3. Incorporate necessary context into the standalone question
4. Ensure the question is complete and unambiguous
</thinking_process>

<chat_history>
{chat_history}
</chat_history>

<follow_up_question>
{question}
</follow_up_question>

Now, transform the follow-up question into a standalone question:

<standalone_question>
"""
    
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template,
        partial_variables={"language": language}
    )

def get_answer_prompt(language: str = "english") -> PromptTemplate:
    """
    Generate a prompt template for generating answers using provided context.
    
    Employs prompt engineering techniques including role definition, structured reasoning,
    context delimitation, and quality guidelines to produce accurate, well-supported answers.
    
    Args:
        language (str): The language in which the answer should be provided (default: "english").
        
    Returns:
        PromptTemplate: A LangChain PromptTemplate for answer generation.
        
    Raises:
        ValueError: If language is not a string or is empty.
    """
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Language must be a non-empty string")
    
    template = """You are a knowledgeable research assistant with expertise in analyzing information and providing accurate, well-reasoned answers.

Your task is to answer the user's question using only the information provided in the context below. Think step-by-step and provide a comprehensive yet concise response.

<instructions>
- Answer in {language}
- Base your response solely on the provided context
- If the context lacks sufficient information, explicitly state this limitation
- Structure your answer logically and coherently
- Support claims with evidence from the context
- Be precise and avoid speculation
</instructions>

<reasoning_approach>
1. Carefully read and understand the question
2. Identify relevant information in the context
3. Analyze how the context relates to the question
4. Formulate a clear, evidence-based response
5. Verify your answer aligns with the available information
</reasoning_approach>

<context>
Chat History:
{chat_history}

Relevant Information:
{context}
</context>

<question>
{question}
</question>

Provide your well-reasoned answer below:

<answer>
"""
    
    return PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template,
        partial_variables={"language": language}
    )