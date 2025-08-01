import logging
from typing import Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from config import GOOGLE_API_KEY

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('memory.log')
    ]
)
logger = logging.getLogger(__name__)

class GeminiModel(Enum):
    """Enumeration of available Gemini models."""
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"

@dataclass
class MemoryConfig:
    """Configuration class for memory settings."""
    model_name: GeminiModel = GeminiModel.GEMINI_1_5_FLASH
    max_token_limit: int = 2048
    memory_key: str = "chat_history"
    input_key: str = "question"
    output_key: str = "answer"
    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 1024
    return_messages: bool = True
    verbose: bool = False
    safety_settings: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.model_name, GeminiModel):
            raise ValueError(f"model_name must be a GeminiModel enum, got {type(self.model_name)}")
        
        if self.max_token_limit <= 0:
            raise ValueError("max_token_limit must be positive")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

class MemoryManager:
    """Enhanced memory manager with Gemini integration and advanced features."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the MemoryManager.
        
        Args:
            config: MemoryConfig object with settings, or None for defaults.
        """
        self.config = config or MemoryConfig()
        self._validate_api_key()
        self._llm = None
        self._memory = None
        
    def _validate_api_key(self) -> None:
        """Validate that the Google API key is available."""
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set in config. "
                "Please obtain an API key from Google AI Studio and set it in your config."
            )
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create and configure the Gemini LLM instance."""
        logger.info(
            "Initializing ChatGoogleGenerativeAI with model: %s, temperature: %s",
            self.config.model_name.value,
            self.config.temperature
        )
        
        llm_params = {
            "model": self.config.model_name.value,
            "google_api_key": GOOGLE_API_KEY,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_output_tokens": self.config.max_output_tokens,
            "verbose": self.config.verbose
        }
        
        # Add safety settings if provided
        if self.config.safety_settings:
            llm_params["safety_settings"] = self.config.safety_settings
        
        try:
            return ChatGoogleGenerativeAI(**llm_params)
        except Exception as e:
            logger.error("Failed to initialize Gemini LLM: %s", str(e))
            raise RuntimeError(f"LLM initialization failed: {str(e)}") from e
    
    def create_memory(self) -> ConversationSummaryBufferMemory:
        """
        Create a memory object for conversation tracking using ConversationSummaryBufferMemory.
        
        Returns:
            ConversationSummaryBufferMemory: Configured memory object with Gemini LLM.
            
        Raises:
            RuntimeError: If memory creation fails.
        """
        try:
            # Create LLM if not already created
            if self._llm is None:
                self._llm = self._create_llm()
            
            logger.info(
                "Creating ConversationSummaryBufferMemory with max_token_limit: %s",
                self.config.max_token_limit
            )
            
            self._memory = ConversationSummaryBufferMemory(
                llm=self._llm,
                max_token_limit=self.config.max_token_limit,
                memory_key=self.config.memory_key,
                input_key=self.config.input_key,
                output_key=self.config.output_key,
                return_messages=self.config.return_messages,
                verbose=self.config.verbose
            )
            
            logger.info("Memory object created successfully")
            return self._memory
            
        except Exception as e:
            logger.error("Failed to create memory object: %s", str(e))
            raise RuntimeError(f"Memory creation failed: {str(e)}") from e
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.
        
        Returns:
            Dict containing memory statistics.
        """
        if self._memory is None:
            return {"status": "not_initialized"}
        
        try:
            buffer = getattr(self._memory, 'buffer', [])
            summary = getattr(self._memory, 'moving_summary_buffer', "")
            
            return {
                "status": "initialized",
                "buffer_length": len(buffer),
                "summary_length": len(summary) if summary else 0,
                "max_token_limit": self.config.max_token_limit,
                "model_name": self.config.model_name.value,
                "memory_key": self.config.memory_key
            }
        except Exception as e:
            logger.warning("Failed to get memory stats: %s", str(e))
            return {"status": "error", "error": str(e)}
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        if self._memory:
            self._memory.clear()
            logger.info("Memory cleared successfully")
        else:
            logger.warning("Attempted to clear uninitialized memory")
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save conversation context to memory.
        
        Args:
            inputs: Dictionary containing input variables.
            outputs: Dictionary containing output variables.
        """
        if self._memory is None:
            raise RuntimeError("Memory not initialized. Call create_memory() first.")
        
        try:
            self._memory.save_context(inputs, outputs)
            logger.debug("Context saved to memory")
        except Exception as e:
            logger.error("Failed to save context: %s", str(e))
            raise

# Factory function for backward compatibility and ease of use
def create_memory(
    model_name: Union[str, GeminiModel] = GeminiModel.GEMINI_1_5_FLASH,
    max_token_limit: int = 2048,
    memory_key: str = "chat_history",
    input_key: str = "question",
    output_key: str = "answer",
    temperature: float = 0.1,
    **kwargs
) -> ConversationSummaryBufferMemory:
    """
    Factory function to create a memory object with Gemini LLM.
    
    Args:
        model_name: Gemini model to use (default: GEMINI_1_5_FLASH).
        max_token_limit: Maximum token limit for memory summary (default: 2048).
        memory_key: Key for conversation history storage (default: "chat_history").
        input_key: Key for incoming questions (default: "question").
        output_key: Key for LLM answers (default: "answer").
        temperature: Temperature setting for LLM (default: 0.1).
        **kwargs: Additional configuration parameters.
        
    Returns:
        ConversationSummaryBufferMemory: Configured memory object.
        
    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If memory creation fails.
    """
    # Handle string model names for backward compatibility
    if isinstance(model_name, str):
        try:
            model_name = GeminiModel(model_name)
        except ValueError:
            logger.warning("Unknown model name '%s', using default", model_name)
            model_name = GeminiModel.GEMINI_1_5_FLASH
    
    config = MemoryConfig(
        model_name=model_name,
        max_token_limit=max_token_limit,
        memory_key=memory_key,
        input_key=input_key,
        output_key=output_key,
        temperature=temperature,
        **kwargs
    )
    
    manager = MemoryManager(config)
    return manager.create_memory()

# Advanced memory creation with custom safety settings
def create_secure_memory(
    model_name: GeminiModel = GeminiModel.GEMINI_1_5_PRO,
    safety_level: str = "strict"
) -> ConversationSummaryBufferMemory:
    """
    Create a memory object with enhanced safety settings.
    
    Args:
        model_name: Gemini model to use.
        safety_level: Safety level ("strict", "moderate", "permissive").
        
    Returns:
        ConversationSummaryBufferMemory: Memory object with safety settings.
    """
    safety_configs = {
        "strict": {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
        },
        "moderate": {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH"
        },
        "permissive": {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
    }
    
    config = MemoryConfig(
        model_name=model_name,
        safety_settings=safety_configs.get(safety_level, safety_configs["moderate"])
    )
    
    manager = MemoryManager(config)
    return manager.create_memory()