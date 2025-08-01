import os
import logging
import tempfile
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from langchain.schema import Document
from langchain_core.document_loaders import BaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY

# Import utils
from utils import (
    clean_text,
    validate_uploaded_file,
    TextCleaningMode,
    FileProtocol,
    performance_monitor
)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('document_loader.log')
    ]
)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Enumeration of supported document types."""
    PDF = "pdf"
    TEXT = "txt"
    DOCX = "docx"
    DOC = "doc"
    CSV = "csv"
    EXCEL = "xlsx"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "md"
    RTF = "rtf"
    UNKNOWN = "unknown"

class ProcessingMode(Enum):
    """Document processing modes."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    AI_ASSISTED = "ai_assisted"
    STRUCTURED = "structured"

@dataclass
class DocumentMetadata:
    """Enhanced metadata for processed documents."""
    file_name: str
    file_size: int
    file_type: DocumentType
    processing_mode: ProcessingMode
    created_at: datetime = field(default_factory=datetime.now)
    file_hash: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    source_path: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoaderConfig:
    """Configuration for document loading."""
    temp_dir: str = "./data/tmp"
    processing_mode: ProcessingMode = ProcessingMode.ENHANCED
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enable_ai_analysis: bool = True
    preserve_formatting: bool = True
    extract_images: bool = False
    language_detection: bool = True
    max_file_size_mb: int = 100
    allowed_extensions: Set[str] = field(default_factory=lambda: {
        'pdf', 'txt', 'docx', 'doc', 'csv', 'xlsx', 'json', 'xml', 'html', 'md', 'rtf'
    })
    text_cleaning_mode: TextCleaningMode = TextCleaningMode.PRESERVE_STRUCTURE
    clean_text: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")

class EnhancedDocumentLoader:
    """Enhanced document loader with AI analysis and robust processing."""
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the enhanced document loader.
        
        Args:
            config: LoaderConfig object with settings, or None for defaults.
        """
        self.config = config or LoaderConfig()
        self._ensure_temp_dir()
        self._gemini_llm = None
        self._processed_files_cache: Dict[str, List[Document]] = {}
        
    def _ensure_temp_dir(self) -> None:
        """Ensure temporary directory exists."""
        os.makedirs(self.config.temp_dir, exist_ok=True)
        logger.info(f"Temporary directory ensured: {self.config.temp_dir}")
    
    def _get_gemini_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """Get or create Gemini LLM instance for AI-assisted processing."""
        if not self.config.enable_ai_analysis:
            return None
            
        if self._gemini_llm is None and GOOGLE_API_KEY:
            try:
                self._gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.1,
                    max_output_tokens=1024
                )
                logger.info("Gemini LLM initialized for document analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini LLM: {str(e)}")
                
        return self._gemini_llm
    
    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def _detect_document_type(self, filename: str, content: bytes) -> DocumentType:
        """
        Detect document type from filename and content.
        
        Args:
            filename: Name of the file.
            content: File content as bytes.
            
        Returns:
            DocumentType: Detected document type.
        """
        # First try by extension
        ext = Path(filename).suffix.lower().lstrip('.')
        if ext in ['pdf']:
            return DocumentType.PDF
        elif ext in ['txt', 'text']:
            return DocumentType.TEXT
        elif ext in ['docx']:
            return DocumentType.DOCX
        elif ext in ['doc']:
            return DocumentType.DOC
        elif ext in ['csv']:
            return DocumentType.CSV
        elif ext in ['xlsx', 'xls']:
            return DocumentType.EXCEL
        elif ext in ['json']:
            return DocumentType.JSON
        elif ext in ['xml']:
            return DocumentType.XML
        elif ext in ['html', 'htm']:
            return DocumentType.HTML
        elif ext in ['md', 'markdown']:
            return DocumentType.MARKDOWN
        elif ext in ['rtf']:
            return DocumentType.RTF
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            if 'pdf' in mime_type:
                return DocumentType.PDF
            elif 'text' in mime_type:
                return DocumentType.TEXT
            elif 'word' in mime_type or 'officedocument' in mime_type:
                return DocumentType.DOCX
        
        # Content-based detection for common formats
        if content.startswith(b'%PDF'):
            return DocumentType.PDF
        elif content.startswith(b'PK') and b'word/' in content[:1024]:
            return DocumentType.DOCX
        
        return DocumentType.UNKNOWN
    
    @performance_monitor
    def _validate_file(self, filename: str, content: bytes) -> Tuple[bool, str]:
        """
        Validate uploaded file using utils function.
        
        Args:
            filename: Name of the file.
            content: File content as bytes.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        # Use utils validation function
        return validate_uploaded_file(
            file=FileProtocol(name=filename, size=len(content)),
            allowed_extensions=self.config.allowed_extensions,
            max_file_size=self.config.max_file_size_mb * 1024 * 1024,
            check_mime_type=True,
            strict_validation=True
        )

    def _create_document_metadata(
        self, 
        filename: str, 
        content: bytes, 
        doc_type: DocumentType,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentMetadata:
        """Create comprehensive metadata for a document."""
        metadata = DocumentMetadata(
            file_name=filename,
            file_size=len(content),
            file_type=doc_type,
            processing_mode=self.config.processing_mode,
            file_hash=self._calculate_file_hash(content)
        )
        
        if extra_metadata:
            metadata.custom_metadata.update(extra_metadata)
            
        return metadata
    
    def _load_with_appropriate_loader(self, file_path: str, doc_type: DocumentType) -> List[Document]:
        """
        Load document using the appropriate loader based on document type.
        
        Args:
            file_path: Path to the document file.
            doc_type: Detected document type.
            
        Returns:
            List of loaded Document objects.
        """
        try:
            if doc_type == DocumentType.PDF:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                
            elif doc_type == DocumentType.TEXT:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding='utf-8')
                
            elif doc_type == DocumentType.DOCX:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                
            elif doc_type == DocumentType.DOC:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(file_path)
                
            elif doc_type == DocumentType.CSV:
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(file_path)
                
            elif doc_type == DocumentType.EXCEL:
                from langchain_community.document_loaders import UnstructuredExcelLoader
                loader = UnstructuredExcelLoader(file_path)
                
            elif doc_type == DocumentType.JSON:
                from langchain_community.document_loaders import JSONLoader
                loader = JSONLoader(file_path, jq_schema='.', text_content=False)
                
            elif doc_type == DocumentType.HTML:
                from langchain_community.document_loaders import UnstructuredHTMLLoader
                loader = UnstructuredHTMLLoader(file_path)
                
            elif doc_type == DocumentType.MARKDOWN:
                from langchain_community.document_loaders import UnstructuredMarkdownLoader
                loader = UnstructuredMarkdownLoader(file_path)
                
            elif doc_type == DocumentType.RTF:
                from langchain_community.document_loaders import UnstructuredRTFLoader
                loader = UnstructuredRTFLoader(file_path)
                
            else:
                # Fallback to text loader
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding='utf-8')
            
            return loader.load()
            
        except ImportError as e:
            logger.error(f"Required loader not available for {doc_type.value}: {str(e)}")
            # Fallback to basic text loading
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return [Document(page_content=content)]
            
        except Exception as e:
            logger.error(f"Error loading {doc_type.value} document: {str(e)}")
            raise
    
    def _enhance_document_with_ai(self, document: Document, metadata: DocumentMetadata) -> Document:
        """
        Enhance document with AI-powered analysis using Gemini.
        
        Args:
            document: Original document.
            metadata: Document metadata.
            
        Returns:
            Enhanced document with AI-generated insights.
        """
        llm = self._get_gemini_llm()
        if not llm or self.config.processing_mode != ProcessingMode.AI_ASSISTED:
            return document
        
        try:
            analysis_prompt = f"""
            Analyze the following document and provide:
            1. A concise summary (2-3 sentences)
            2. Key topics or themes (3-5 items)
            3. Document language detection
            4. Content type classification
            
            Document content (first 1000 characters):
            {document.page_content[:1000]}
            
            Provide response in JSON format:
            {{
                "summary": "...",
                "key_topics": ["topic1", "topic2", ...],
                "language": "language_code",
                "content_type": "type"
            }}
            """
            
            response = llm.invoke(analysis_prompt)
            
            try:
                analysis = json.loads(response.content)
                
                # Add AI analysis to metadata
                enhanced_metadata = document.metadata.copy()
                enhanced_metadata.update({
                    'ai_summary': analysis.get('summary', ''),
                    'ai_topics': analysis.get('key_topics', []),
                    'detected_language': analysis.get('language', ''),
                    'content_classification': analysis.get('content_type', ''),
                    'ai_enhanced': True
                })
                
                return Document(
                    page_content=document.page_content,
                    metadata=enhanced_metadata
                )
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI analysis JSON, using original document")
                
        except Exception as e:
            logger.warning(f"AI enhancement failed: {str(e)}")
        
        return document
    
    def _chunk_document(self, document: Document) -> List[Document]:
        """
        Split document into chunks if needed.
        
        Args:
            document: Original document.
            
        Returns:
            List of chunked documents.
        """
        if len(document.page_content) <= self.config.chunk_size:
            return [document]
        
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_text(document.page_content)
            
            chunked_docs = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'is_chunked': True
                })
                
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
            
            return chunked_docs
            
        except ImportError:
            logger.warning("RecursiveCharacterTextSplitter not available, returning original document")
            return [document]
    
    @performance_monitor
    def process_uploaded_documents(self, uploaded_files: List[Any]) -> List[Document]:
        """
        Process uploaded files into LangChain-compatible Document objects with enhanced features.
        
        Args:
            uploaded_files: List of uploaded file objects with .name and .getvalue() methods.
        
        Returns:
            List of processed LangChain Document objects.
        """
        documents: List[Document] = []
        processed_count = 0
        failed_count = 0
        
        logger.info(f"Starting to process {len(uploaded_files)} uploaded files")
        
        for uploaded_file in uploaded_files:
            try:
                # Get file content
                file_content = uploaded_file.getvalue()
                file_hash = self._calculate_file_hash(file_content)
                
                # Check cache
                if file_hash in self._processed_files_cache:
                    logger.info(f"Using cached version of {uploaded_file.name}")
                    documents.extend(self._processed_files_cache[file_hash])
                    processed_count += 1
                    continue
                
                # Validate file using utils function
                is_valid, error_msg = self._validate_file(uploaded_file.name, file_content)
                if not is_valid:
                    logger.error(f"File validation failed for {uploaded_file.name}: {error_msg}")
                    failed_count += 1
                    continue
                
                # Detect document type
                doc_type = self._detect_document_type(uploaded_file.name, file_content)
                logger.info(f"Detected document type for {uploaded_file.name}: {doc_type.value}")
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    mode='wb', 
                    suffix=f".{doc_type.value}", 
                    dir=self.config.temp_dir, 
                    delete=False
                ) as tmp_file:
                    tmp_file.write(file_content)
                    temp_file_path = tmp_file.name
                
                try:
                    # Load document
                    loaded_docs = self._load_with_appropriate_loader(temp_file_path, doc_type)
                    
                    # Create metadata
                    metadata = self._create_document_metadata(
                        uploaded_file.name, 
                        file_content, 
                        doc_type,
                        {'temp_file_path': temp_file_path}
                    )
                    
                    # Process each loaded document
                    processed_docs = []
                    for doc in loaded_docs:
                        # Clean text if enabled
                        if self.config.clean_text:
                            doc.page_content = clean_text(
                                doc.page_content,
                                mode=self.config.text_cleaning_mode,
                                lowercase=False,
                                remove_non_ascii=True,
                                preserve_newlines=True,
                                remove_urls=True,
                                remove_emails=True
                            )
                        
                        # Add base metadata to document
                        doc.metadata.update({
                            'source_file': uploaded_file.name,
                            'file_type': doc_type.value,
                            'file_size': metadata.file_size,
                            'file_hash': metadata.file_hash,
                            'processing_timestamp': metadata.created_at.isoformat(),
                            'processing_mode': metadata.processing_mode.value
                        })
                        
                        # AI enhancement if enabled
                        if self.config.processing_mode == ProcessingMode.AI_ASSISTED:
                            doc = self._enhance_document_with_ai(doc, metadata)
                        
                        # Chunk if necessary
                        if self.config.processing_mode in [ProcessingMode.ENHANCED, ProcessingMode.AI_ASSISTED]:
                            chunked_docs = self._chunk_document(doc)
                            processed_docs.extend(chunked_docs)
                        else:
                            processed_docs.append(doc)
                    
                    # Cache processed documents
                    self._processed_files_cache[file_hash] = processed_docs
                    documents.extend(processed_docs)
                    processed_count += 1
                    
                    logger.info(
                        f"Successfully processed {uploaded_file.name}: "
                        f"{len(processed_docs)} document(s) created"
                    )
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except OSError:
                        pass
                        
            except Exception as e:
                logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                failed_count += 1
                continue
        
        logger.info(
            f"Document processing completed: {processed_count} successful, "
            f"{failed_count} failed, {len(documents)} total documents created"
        )
        
        return documents

def flatten_dict(d: Dict[Any, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary with enhanced handling.
    
    Args:
        d: The dictionary to flatten.
        parent_key: Base key for recursive calls.
        sep: Separator to use between keys.
    
    Returns:
        A flattened dictionary with keys joined by the separator.
    """
    items: Dict[str, Any] = {}
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            if not v:  # Empty list
                items[new_key] = ""
            elif all(isinstance(item, dict) for item in v):
                # List of dictionaries - flatten each and join
                flattened_items = []
                for i, item in enumerate(v):
                    flattened = flatten_dict(item, f"{new_key}[{i}]", sep=sep)
                    flattened_items.append(flattened)
                items.update({k: v for d in flattened_items for k, v in d.items()})
            else:
                # List of primitives
                items[new_key] = ", ".join(str(item) for item in v)
        elif v is None:
            items[new_key] = ""
        else:
            items[new_key] = str(v)
    
    return items

@performance_monitor
def process_patient_data(
    patient_data: Dict[str, Any], 
    processing_mode: ProcessingMode = ProcessingMode.ENHANCED
) -> List[Document]:
    """
    Convert patient data into LangChain-compatible Document objects with enhanced processing.
    
    Args:
        patient_data: Patient data dictionary.
        processing_mode: Processing mode to use.
    
    Returns:
        List of LangChain Document objects with enhanced metadata.
    """
    documents: List[Document] = []
    
    logger.info(f"Processing patient data with {len(patient_data)} sections in {processing_mode.value} mode")
    
    for section, content in patient_data.items():
        try:
            if isinstance(content, dict):
                # Flatten nested dictionary
                flat_content = flatten_dict(content)
                
                # Create structured content
                if processing_mode == ProcessingMode.STRUCTURED:
                    # JSON format for structured processing
                    content_str = json.dumps(flat_content, indent=2)
                    content_format = "json"
                else:
                    # Human-readable format
                    content_str = "\n".join(f"{key}: {value}" for key, value in flat_content.items())
                    content_format = "text"
                
                # Create comprehensive content with header
                full_content = f"=== {section.upper()} ===\n{content_str}"
                
                # Enhanced metadata
                metadata = {
                    'section': section,
                    'content_type': 'patient_data',
                    'format': content_format,
                    'field_count': len(flat_content),
                    'processing_mode': processing_mode.value,
                    'created_at': datetime.now().isoformat()
                }
                
                # Add section-specific metadata
                if 'demographics' in section.lower():
                    metadata.update({'category': 'demographics', 'priority': 'high'})
                elif 'clinical' in section.lower() or 'notes' in section.lower():
                    metadata.update({'category': 'clinical', 'priority': 'high'})
                elif 'medication' in section.lower():
                    metadata.update({'category': 'medication', 'priority': 'medium'})
                else:
                    metadata.update({'category': 'general', 'priority': 'low'})
                
            else:
                # Simple content
                full_content = f"=== {section.upper()} ===\n{section}: {content}"
                metadata = {
                    'section': section,
                    'content_type': 'patient_data',
                    'format': 'simple',
                    'processing_mode': processing_mode.value,
                    'created_at': datetime.now().isoformat(),
                    'category': 'general',
                    'priority': 'low'
                }
            
            documents.append(Document(
                page_content=full_content,
                metadata=metadata
            ))
            
        except Exception as e:
            logger.error(f"Error processing section {section}: {str(e)}")
            # Create fallback document
            fallback_content = f"=== {section.upper()} ===\nError processing section: {str(e)}"
            documents.append(Document(
                page_content=fallback_content,
                metadata={
                    'section': section,
                    'content_type': 'patient_data',
                    'format': 'error',
                    'error': str(e),
                    'processing_mode': processing_mode.value,
                    'created_at': datetime.now().isoformat()
                }
            ))
    
    logger.info(f"Created {len(documents)} patient data documents")
    return documents

# Add top-level function for uploaded documents processing
def process_uploaded_documents(uploaded_files: List[Any]) -> List[Document]:
    """
    Top-level function to process uploaded documents using default configuration.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        List of processed Document objects
    """
    loader = EnhancedDocumentLoader()
    return loader.process_uploaded_documents(uploaded_files)

# Factory functions for ease of use
def create_document_loader(
    processing_mode: ProcessingMode = ProcessingMode.ENHANCED,
    enable_ai_analysis: bool = True,
    clean_text: bool = True,
    **kwargs
) -> EnhancedDocumentLoader:
    """
    Factory function to create an enhanced document loader.
    
    Args:
        processing_mode: Processing mode to use.
        enable_ai_analysis: Whether to enable AI-powered analysis.
        clean_text: Whether to clean text content.
        **kwargs: Additional configuration parameters.
        
    Returns:
        Configured EnhancedDocumentLoader instance.
    """
    config = LoaderConfig(
        processing_mode=processing_mode,
        enable_ai_analysis=enable_ai_analysis,
        clean_text=clean_text,
        **kwargs
    )
    
    return EnhancedDocumentLoader(config)

def load_documents_simple(uploaded_files: List[Any]) -> List[Document]:
    """
    Simple document loading function for backward compatibility.
    
    Args:
        uploaded_files: List of uploaded file objects.
        
    Returns:
        List of processed documents.
    """
    loader = create_document_loader(processing_mode=ProcessingMode.BASIC)
    return loader.process_uploaded_documents(uploaded_files)