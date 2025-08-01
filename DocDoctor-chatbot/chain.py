import logging
from typing import Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from templates import get_standalone_question_prompt, get_answer_prompt
from memory import create_memory, create_secure_memory, MemoryManager, MemoryConfig, GeminiModel
from config import GOOGLE_API_KEY

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('conversational_chain.log')
    ]
)
logger = logging.getLogger(__name__)

class ChainMode(Enum):
    """Enumeration of chain operation modes."""
    STANDARD = "standard"
    SECURE = "secure"
    PERFORMANCE = "performance"
    CREATIVE = "creative"

@dataclass
class ChainConfig:
    """Configuration class for conversational chain settings."""
    language: str = "english"
    return_source_documents: bool = True
    return_generated_question: bool = False
    max_tokens_limit: int = 2048
    search_kwargs: Dict[str, Any] = None
    chain_type: str = "stuff"
    verbose: bool = False
    mode: ChainMode = ChainMode.STANDARD
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.search_kwargs is None:
            self.search_kwargs = {"k": 4}

class ConversationalChainManager:
    """Enhanced conversational chain manager with Gemini integration."""
    
    def __init__(self, config: Optional[ChainConfig] = None):
        """
        Initialize the ConversationalChainManager.
        
        Args:
            config: ChainConfig object with settings, or None for defaults.
        """
        self.config = config or ChainConfig()
        self._validate_environment()
        self.memory_manager = None
        self.chain = None
        
    def _validate_environment(self) -> None:
        """Validate that required environment variables are set."""
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set in config. "
                "Please obtain an API key from Google AI Studio."
            )
    
    def _get_optimized_llm(self, mode: ChainMode) -> ChatGoogleGenerativeAI:
        """
        Get an optimized LLM instance based on the specified mode.
        
        Args:
            mode: Chain operation mode.
            
        Returns:
            ChatGoogleGenerativeAI: Configured LLM instance.
        """
        mode_configs = {
            ChainMode.STANDARD: {
                "model": GeminiModel.GEMINI_1_5_FLASH.value,
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024
            },
            ChainMode.SECURE: {
                "model": GeminiModel.GEMINI_1_5_PRO.value,
                "temperature": 0.05,
                "top_p": 0.7,
                "top_k": 20,
                "max_output_tokens": 512
            },
            ChainMode.PERFORMANCE: {
                "model": GeminiModel.GEMINI_1_5_FLASH.value,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 50,
                "max_output_tokens": 2048
            },
            ChainMode.CREATIVE: {
                "model": GeminiModel.GEMINI_1_5_PRO.value,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 60,
                "max_output_tokens": 1536
            }
        }
        
        config = mode_configs.get(mode, mode_configs[ChainMode.STANDARD])
        
        logger.info(f"Creating LLM for mode: {mode.value} with model: {config['model']}")
        
        try:
            return ChatGoogleGenerativeAI(
                google_api_key=GOOGLE_API_KEY,
                **config,
                verbose=self.config.verbose
            )
        except Exception as e:
            logger.error(f"Failed to create LLM: {str(e)}")
            raise RuntimeError(f"LLM creation failed: {str(e)}") from e
    
    def _create_memory_for_mode(self, mode: ChainMode) -> Any:
        """
        Create memory instance optimized for the specified mode.
        
        Args:
            mode: Chain operation mode.
            
        Returns:
            Memory instance configured for the mode.
        """
        try:
            if mode == ChainMode.SECURE:
                logger.info("Creating secure memory with enhanced safety settings")
                return create_secure_memory(
                    model_name=GeminiModel.GEMINI_1_5_PRO,
                    safety_level="strict",
                    # Ensure chain compatibility
                    return_messages=False
                )
            else:
                # Use memory manager for other modes
                memory_configs = {
                    ChainMode.STANDARD: {
                        "model_name": GeminiModel.GEMINI_1_5_FLASH,
                        "max_token_limit": self.config.max_tokens_limit,
                        "temperature": 0.1,
                        "return_messages": False  # Critical for chain compatibility
                    },
                    ChainMode.PERFORMANCE: {
                        "model_name": GeminiModel.GEMINI_1_5_FLASH,
                        "max_token_limit": self.config.max_tokens_limit * 2,
                        "temperature": 0.2,
                        "return_messages": False
                    },
                    ChainMode.CREATIVE: {
                        "model_name": GeminiModel.GEMINI_1_5_PRO,
                        "max_token_limit": self.config.max_tokens_limit,
                        "temperature": 0.7,
                        "return_messages": False
                    }
                }
                
                config = memory_configs.get(mode, memory_configs[ChainMode.STANDARD])
                logger.info(f"Creating memory for mode: {mode.value}")
                
                memory_config = MemoryConfig(**config)
                self.memory_manager = MemoryManager(memory_config)
                return self.memory_manager.create_memory()
                
        except Exception as e:
            logger.error(f"Failed to create memory for mode {mode.value}: {str(e)}")
            raise RuntimeError(f"Memory creation failed: {str(e)}") from e
    
    def create_chain(
        self, 
        retriever: BaseRetriever,
        llm: Optional[BaseLanguageModel] = None,
        custom_memory: Optional[Any] = None
    ) -> ConversationalRetrievalChain:
        """
        Create a conversational retrieval chain with Gemini integration.
        
        Args:
            retriever: Retriever instance for retrieving relevant documents.
            llm: Optional custom LLM instance. If None, creates optimized LLM.
            custom_memory: Optional custom memory instance. If None, creates optimized memory.
            
        Returns:
            ConversationalRetrievalChain: The configured conversational chain.
            
        Raises:
            RuntimeError: If chain creation fails.
        """
        try:
            logger.info(f"Creating conversational chain in {self.config.mode.value} mode")
            
            # Get or create LLM
            if llm is None:
                llm = self._get_optimized_llm(self.config.mode)
            
            # Get or create memory
            if custom_memory is None:
                memory = self._create_memory_for_mode(self.config.mode)
            else:
                memory = custom_memory
                # Ensure memory has required configuration
                if hasattr(memory, 'memory_key') and memory.memory_key != "chat_history":
                    logger.warning(f"Custom memory uses key '{memory.memory_key}' instead of 'chat_history'")
            
            # Get prompts with language support
            logger.info(f"Fetching prompts for language: {self.config.language}")
            standalone_question_prompt = get_standalone_question_prompt(self.config.language)
            answer_prompt = get_answer_prompt(self.config.language)
            
            # Configure retriever
            if hasattr(retriever, 'search_kwargs'):
                retriever.search_kwargs.update(self.config.search_kwargs)
            
            logger.info("Initializing ConversationalRetrievalChain")
            
            # Create chain with enhanced configuration
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                condense_question_prompt=standalone_question_prompt,
                combine_docs_chain_kwargs={
                    "prompt": answer_prompt,
                    "document_variable_name": "context"
                },
                return_source_documents=self.config.return_source_documents,
                return_generated_question=self.config.return_generated_question,
                chain_type=self.config.chain_type,
                verbose=self.config.verbose,
                max_tokens_limit=self.config.max_tokens_limit
            )
            
            logger.info("Conversational chain created successfully")
            return self.chain
            
        except Exception as e:
            logger.error(f"Error creating conversational chain: {str(e)}")
            raise RuntimeError(f"Chain creation failed: {str(e)}") from e
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current chain and memory state.
        
        Returns:
            Dict containing chain and memory statistics.
        """
        stats = {
            "chain_initialized": self.chain is not None,
            "config": {
                "language": self.config.language,
                "mode": self.config.mode.value,
                "max_tokens_limit": self.config.max_tokens_limit,
                "return_source_documents": self.config.return_source_documents
            }
        }
        
        if self.memory_manager:
            stats["memory"] = self.memory_manager.get_memory_stats()
        
        return stats
    
    def reset_conversation(self) -> None:
        """Reset the conversation by clearing memory."""
        try:
            if self.memory_manager:
                self.memory_manager.clear_memory()
                logger.info("Conversation memory cleared")
            elif self.chain and hasattr(self.chain, 'memory'):
                if hasattr(self.chain.memory, 'clear'):
                    self.chain.memory.clear()
                    logger.info("Cleared chain memory directly")
                else:
                    logger.warning("Chain memory doesn't have clear method")
            else:
                logger.warning("No memory manager available to clear")
        except Exception as e:
            logger.error(f"Failed to reset conversation: {str(e)}")

# Enhanced factory functions for backward compatibility and ease of use

def create_conversational_chain(
    llm: Optional[BaseLanguageModel],
    retriever: BaseRetriever, 
    language: str = "english", 
    custom_memory: Optional[Any] = None,
    mode: Union[str, ChainMode] = ChainMode.STANDARD,
    **kwargs
) -> ConversationalRetrievalChain:
    """
    Create a conversational retrieval chain with memory (backward compatible).
    
    This function sets up a LangChain ConversationalRetrievalChain with Gemini integration by:
      - Using optimized Gemini models based on the specified mode
      - Retrieving and formatting prompts with language support
      - Creating or using a provided memory instance with Gemini LLM
      - Initializing the chain with enhanced configuration
    
    Args:
        llm: Language model instance (can be None for auto-creation).
        retriever: Retriever instance for retrieving relevant documents.
        language (str): Language for the prompts (default is "english").
        custom_memory: (Optional) A custom memory instance. If None, creates optimized memory.
        mode: Chain operation mode (standard, secure, performance, creative).
        **kwargs: Additional configuration parameters.
    
    Returns:
        ConversationalRetrievalChain: The configured LangChain retrieval chain.
    
    Raises:
        RuntimeError: If an error occurs during the creation of the chain.
    """
    # Handle string mode for backward compatibility
    if isinstance(mode, str):
        try:
            mode = ChainMode(mode)
        except ValueError:
            logger.warning(f"Unknown mode '{mode}', using standard mode")
            mode = ChainMode.STANDARD
    
    # Create configuration
    config = ChainConfig(
        language=language,
        mode=mode,
        **kwargs
    )
    
    # Create and use manager
    manager = ConversationalChainManager(config)
    return manager.create_chain(retriever, llm, custom_memory)

def create_gemini_chain(
    retriever: BaseRetriever,
    model_name: Union[str, GeminiModel] = GeminiModel.GEMINI_1_5_FLASH,
    language: str = "english",
    temperature: float = 0.1,
    mode: ChainMode = ChainMode.STANDARD,
    **kwargs
) -> ConversationalRetrievalChain:
    """
    Create a conversational chain with specific Gemini model configuration.
    
    Args:
        retriever: Retriever instance for retrieving relevant documents.
        model_name: Specific Gemini model to use.
        language: Language for prompts and responses.
        temperature: Temperature setting for the model.
        mode: Chain operation mode.
        **kwargs: Additional configuration parameters.
        
    Returns:
        ConversationalRetrievalChain: Configured chain with specific Gemini model.
    """
    # Handle string model names
    if isinstance(model_name, str):
        try:
            model_name = GeminiModel(model_name)
        except ValueError:
            logger.warning(f"Unknown model '{model_name}', using default")
            model_name = GeminiModel.GEMINI_1_5_FLASH
    
    # Create custom LLM with specific configuration
    llm = ChatGoogleGenerativeAI(
        model=model_name.value,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        verbose=kwargs.get('verbose', False)
    )
    
    config = ChainConfig(
        language=language,
        mode=mode,
        **kwargs
    )
    
    manager = ConversationalChainManager(config)
    return manager.create_chain(retriever, llm)

def create_secure_conversational_chain(
    retriever: BaseRetriever,
    language: str = "english",
    safety_level: str = "strict"
) -> ConversationalRetrievalChain:
    """
    Create a conversational chain with enhanced security settings.
    
    Args:
        retriever: Retriever instance for retrieving relevant documents.
        language: Language for prompts and responses.
        safety_level: Safety level for content filtering.
        
    Returns:
        ConversationalRetrievalChain: Secure conversational chain.
    """
    config = ChainConfig(
        language=language,
        mode=ChainMode.SECURE,
        max_tokens_limit=1024,  # Lower limit for security
        return_source_documents=True,
        verbose=False
    )
    
    manager = ConversationalChainManager(config)
    
    # Create secure memory
    secure_memory = create_secure_memory(
        model_name=GeminiModel.GEMINI_1_5_PRO,
        safety_level=safety_level,
        return_messages=False  # Ensure chain compatibility
    )
    
    return manager.create_chain(retriever, custom_memory=secure_memory)

# Utility function to get chain performance metrics
def analyze_chain_performance(chain: ConversationalRetrievalChain) -> Dict[str, Any]:
    """
    Analyze the performance characteristics of a conversational chain.
    
    Args:
        chain: The conversational chain to analyze.
        
    Returns:
        Dict containing performance metrics and configuration details.
    """
    try:
        analysis = {
            "chain_type": type(chain).__name__,
            "llm_model": getattr(chain.combine_docs_chain.llm_chain.llm, 'model_name', 'unknown'),
            "has_memory": hasattr(chain, 'memory') and chain.memory is not None,
            "returns_sources": getattr(chain, 'return_source_documents', False),
            "retriever_type": type(chain.retriever).__name__
        }
        
        if hasattr(chain, 'memory') and chain.memory:
            if hasattr(chain.memory, 'buffer'):
                analysis["memory_buffer_size"] = len(getattr(chain.memory, 'buffer', []))
            if hasattr(chain.memory, 'max_token_limit'):
                analysis["memory_max_tokens"] = chain.memory.max_token_limit
        
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze chain performance: {str(e)}")
        return {"error": str(e)}