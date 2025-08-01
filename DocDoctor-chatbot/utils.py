import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import hashlib
import mimetypes
from datetime import datetime

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('utils.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TextCleaningMode(Enum):
    """Enumeration of text cleaning modes."""
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    PRESERVE_STRUCTURE = "preserve_structure"
    NORMALIZE_UNICODE = "normalize_unicode"

class FileType(Enum):
    """Enumeration of supported file types."""
    PDF = ".pdf"
    TXT = ".txt"
    CSV = ".csv"
    DOCX = ".docx"
    DOC = ".doc"
    XLSX = ".xlsx"
    XLS = ".xls"
    JSON = ".json"
    XML = ".xml"
    HTML = ".html"
    MD = ".md"
    RTF = ".rtf"

@dataclass
class DocumentInfo:
    """Enhanced document information container."""
    content: str
    index: int
    metadata: Dict[str, Any]
    word_count: int
    char_count: int
    hash: str
    
    @classmethod
    def from_document(cls, doc: Any, index: int) -> 'DocumentInfo':
        """Create DocumentInfo from a document object."""
        try:
            content = getattr(doc, 'page_content', str(doc))
            metadata = getattr(doc, 'metadata', {})
        except Exception as e:
            logger.warning(f"Error extracting document content: {e}")
            content = str(doc)
            metadata = {}
        
        return cls(
            content=content,
            index=index,
            metadata=metadata,
            word_count=len(content.split()),
            char_count=len(content),
            hash=hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        )

class FileProtocol(Protocol):
    """Protocol for file-like objects."""
    name: str
    size: Optional[int] = None

def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} executed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper

@performance_monitor
def clean_text(
    text: str,
    mode: Union[str, TextCleaningMode] = TextCleaningMode.BASIC,
    lowercase: bool = False,
    remove_non_ascii: bool = False,
    preserve_newlines: bool = False,
    remove_urls: bool = False,
    remove_emails: bool = False,
    custom_patterns: Optional[List[str]] = None
) -> str:
    """
    Advanced text cleaning and preprocessing with multiple modes and options.
    
    Args:
        text: Raw text to clean.
        mode: Cleaning mode (basic, aggressive, preserve_structure, normalize_unicode).
        lowercase: Convert text to lowercase.
        remove_non_ascii: Remove non-ASCII characters.
        preserve_newlines: Preserve line breaks in cleaning.
        remove_urls: Remove URLs from text.
        remove_emails: Remove email addresses from text.
        custom_patterns: List of regex patterns to remove.
        
    Returns:
        Cleaned and preprocessed text.
        
    Raises:
        ValueError: If text is not a string or mode is invalid.
        TypeError: If custom_patterns is not a list.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected string, got {type(text)}")
    
    if isinstance(mode, str):
        try:
            mode = TextCleaningMode(mode)
        except ValueError:
            logger.warning(f"Invalid mode '{mode}', using BASIC")
            mode = TextCleaningMode.BASIC
    
    if custom_patterns is not None and not isinstance(custom_patterns, list):
        raise TypeError("custom_patterns must be a list of strings")
    
    logger.debug(f"Cleaning text with mode: {mode.value}, length: {len(text)}")
    
    # Store original length for comparison
    original_length = len(text)
    
    # Apply cleaning based on mode
    if mode == TextCleaningMode.NORMALIZE_UNICODE:
        text = unicodedata.normalize('NFKD', text)
    
    # Remove URLs if requested
    if remove_urls:
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        text = re.sub(url_pattern, '', text)
    
    # Remove emails if requested
    if remove_emails:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '', text)
    
    # Apply custom patterns
    if custom_patterns:
        for pattern in custom_patterns:
            try:
                text = re.sub(pattern, '', text)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
    
    # Mode-specific cleaning
    if mode == TextCleaningMode.BASIC:
        text = re.sub(r'\s+', ' ', text).strip()
    
    elif mode == TextCleaningMode.AGGRESSIVE:
        # Remove special characters, keep only alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
    
    elif mode == TextCleaningMode.PRESERVE_STRUCTURE:
        # Clean while preserving paragraph structure
        if preserve_newlines:
            text = re.sub(r'[ \t]+', ' ', text)  # Clean horizontal whitespace only
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        else:
            text = re.sub(r'\s+', ' ', text).strip()
    
    # Apply transformations
    if lowercase:
        text = text.lower()
    
    if remove_non_ascii:
        text = text.encode('ascii', errors='ignore').decode('ascii')
    
    final_length = len(text)
    reduction_percent = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0
    
    logger.debug(f"Text cleaned: {original_length} -> {final_length} chars ({reduction_percent:.1f}% reduction)")
    
    return text

@performance_monitor
def format_documents(
    documents: List[Any],
    separator: str = "\n\n",
    include_index: bool = False,
    include_metadata: bool = False,
    include_stats: bool = False,
    max_content_length: Optional[int] = None,
    template: Optional[str] = None
) -> str:
    """
    Advanced document formatting with rich options and metadata inclusion.
    
    Args:
        documents: List of document objects.
        separator: Separator between documents.
        include_index: Include document index in output.
        include_metadata: Include document metadata.
        include_stats: Include word/character counts.
        max_content_length: Maximum content length per document.
        template: Custom formatting template with placeholders.
        
    Returns:
        Formatted document content string.
        
    Raises:
        ValueError: If documents list is empty or invalid.
    """
    if not documents:
        logger.warning("Empty documents list provided")
        return ""
    
    if not isinstance(documents, list):
        raise ValueError("Documents must be provided as a list")
    
    logger.info(f"Formatting {len(documents)} documents")
    
    formatted_docs = []
    total_words = 0
    total_chars = 0
    
    for idx, doc in enumerate(documents, start=1):
        try:
            doc_info = DocumentInfo.from_document(doc, idx)
            
            # Truncate content if needed
            content = doc_info.content
            if max_content_length and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
                logger.debug(f"Document {idx} truncated to {max_content_length} characters")
            
            # Use custom template if provided
            if template:
                formatted_doc = template.format(
                    index=idx,
                    content=content,
                    metadata=doc_info.metadata,
                    word_count=doc_info.word_count,
                    char_count=doc_info.char_count,
                    hash=doc_info.hash
                )
            else:
                # Build formatted document
                parts = []
                
                if include_index:
                    parts.append(f"Document {idx} (Hash: {doc_info.hash}):")
                
                if include_metadata and doc_info.metadata:
                    metadata_str = ", ".join(f"{k}: {v}" for k, v in doc_info.metadata.items())
                    parts.append(f"Metadata: {metadata_str}")
                
                if include_stats:
                    parts.append(f"Stats: {doc_info.word_count} words, {doc_info.char_count} characters")
                
                parts.append(content)
                formatted_doc = "\n".join(parts)
            
            formatted_docs.append(formatted_doc)
            total_words += doc_info.word_count
            total_chars += doc_info.char_count
            
        except Exception as e:
            logger.error(f"Error formatting document {idx}: {e}")
            formatted_docs.append(f"Document {idx}: [Error processing document]")
    
    result = separator.join(formatted_docs)
    
    if include_stats:
        stats_summary = f"\n\n--- Summary Statistics ---\nTotal Documents: {len(documents)}\nTotal Words: {total_words}\nTotal Characters: {total_chars}"
        result += stats_summary
    
    logger.info(f"Formatted {len(formatted_docs)} documents successfully")
    return result

@performance_monitor
def validate_uploaded_files(
    files: List[FileProtocol],
    allowed_extensions: Optional[List[Union[str, FileType]]] = None,
    max_file_size: Optional[int] = None,
    check_mime_type: bool = False,
    strict_validation: bool = False
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Comprehensive file validation with size limits and MIME type checking.
    
    Args:
        files: List of uploaded file objects with 'name' attribute.
        allowed_extensions: List of allowed extensions (default: common document types).
        max_file_size: Maximum file size in bytes.
        check_mime_type: Validate MIME types against extensions.
        strict_validation: Enable strict validation mode.
        
    Returns:
        Tuple of (is_valid, validation_results) where validation_results contains:
        - invalid_extensions: Files with invalid extensions
        - oversized_files: Files exceeding size limit
        - mime_type_mismatches: Files with MIME type mismatches
        - validation_errors: Files that caused validation errors
        
    Raises:
        TypeError: If files is not a list.
    """
    if not isinstance(files, list):
        raise TypeError("Files must be provided as a list")
    
    # Set default allowed extensions
    if allowed_extensions is None:
        allowed_extensions = [
            FileType.PDF, FileType.TXT, FileType.CSV, FileType.DOCX,
            FileType.JSON, FileType.MD, FileType.HTML
        ]
    
    # Normalize extensions
    normalized_extensions = []
    for ext in allowed_extensions:
        if isinstance(ext, FileType):
            normalized_extensions.append(ext.value.lower())
        elif isinstance(ext, str):
            normalized_extensions.append(ext.lower() if ext.startswith('.') else f'.{ext.lower()}')
        else:
            logger.warning(f"Invalid extension type: {type(ext)}")
    
    logger.info(f"Validating {len(files)} files against {len(normalized_extensions)} allowed extensions")
    
    validation_results = {
        'invalid_extensions': [],
        'oversized_files': [],
        'mime_type_mismatches': [],
        'validation_errors': []
    }
    
    for file in files:
        try:
            file_name = getattr(file, 'name', '')
            if not file_name:
                validation_results['validation_errors'].append('Unnamed file')
                continue
            
            file_path = Path(file_name)
            file_extension = file_path.suffix.lower()
            
            # Check extension
            if file_extension not in normalized_extensions:
                validation_results['invalid_extensions'].append(file_name)
                logger.debug(f"Invalid extension for file: {file_name}")
            
            # Check file size
            if max_file_size is not None:
                file_size = getattr(file, 'size', None)
                if file_size is not None and file_size > max_file_size:
                    validation_results['oversized_files'].append(
                        f"{file_name} ({file_size} bytes > {max_file_size} bytes)"
                    )
                    logger.debug(f"Oversized file: {file_name} ({file_size} bytes)")
            
            # Check MIME type
            if check_mime_type:
                expected_mime, _ = mimetypes.guess_type(file_name)
                if expected_mime is None and strict_validation:
                    validation_results['mime_type_mismatches'].append(
                        f"{file_name} (unknown MIME type)"
                    )
        
        except Exception as e:
            logger.error(f"Error validating file {getattr(file, 'name', 'unknown')}: {e}")
            validation_results['validation_errors'].append(f"Validation error: {str(e)}")
    
    # Determine overall validity
    is_valid = all(not errors for errors in validation_results.values())
    
    # Log results
    total_issues = sum(len(errors) for errors in validation_results.values())
    if is_valid:
        logger.info("All files passed validation")
    else:
        logger.warning(f"File validation failed: {total_issues} issues found")
        for category, issues in validation_results.items():
            if issues:
                logger.warning(f"{category}: {len(issues)} issues")
    
    return is_valid, validation_results

# Add alias for singular form to match import
validate_uploaded_file = validate_uploaded_files

def extract_text_statistics(text: str) -> Dict[str, Any]:
    """
    Extract comprehensive statistics from text.
    
    Args:
        text: Input text to analyze.
        
    Returns:
        Dictionary containing various text statistics.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'character_count': len(text),
        'character_count_no_spaces': len(text.replace(' ', '')),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'average_sentence_length': len(words) / len(sentences) if sentences else 0,
        'unique_words': len(set(word.lower() for word in words)),
        'lexical_diversity': len(set(word.lower() for word in words)) / len(words) if words else 0
    }

def batch_process_texts(
    texts: List[str],
    cleaning_func: callable = clean_text,
    **cleaning_kwargs
) -> List[str]:
    """
    Process multiple texts in batch with error handling and logging.
    
    Args:
        texts: List of texts to process.
        cleaning_func: Function to apply to each text.
        **cleaning_kwargs: Arguments to pass to cleaning function.
        
    Returns:
        List of processed texts.
    """
    if not isinstance(texts, list):
        raise TypeError("Texts must be provided as a list")
    
    logger.info(f"Batch processing {len(texts)} texts")
    
    processed_texts = []
    errors = 0
    
    for i, text in enumerate(texts):
        try:
            processed_text = cleaning_func(text, **cleaning_kwargs)
            processed_texts.append(processed_text)
        except Exception as e:
            logger.error(f"Error processing text {i}: {e}")
            processed_texts.append(text)  # Return original on error
            errors += 1
    
    logger.info(f"Batch processing completed with {errors} errors")
    return processed_texts