import logging
import os
import hashlib
import json
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain.retrievers import (
    BM25Retriever, 
    EnsembleRetriever,
    MultiQueryRetriever,
    ContextualCompressionRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
import google.generativeai as genai
from sentence_transformers import CrossEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from config import GEMINI_API_KEY

# Set up advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class RetrievalConfig:
    """Configuration for advanced retrieval strategies."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_top_k: int = 10
    bm25_top_k: int = 10
    ensemble_weights: List[float] = None
    rerank_top_k: int = 5
    use_mmr: bool = True
    mmr_diversity_threshold: float = 0.7
    use_contextual_compression: bool = True
    use_query_expansion: bool = True
    use_hypothetical_questions: bool = True
    use_parent_document_retrieval: bool = True
    cache_embeddings: bool = True
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = [0.6, 0.4]  # Semantic, BM25

class AdvancedRAGManager:
    """Advanced RAG system with multiple retrieval strategies and optimization."""
    
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: Optional[str] = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 config: Optional[RetrievalConfig] = None):
        """
        Initialize advanced RAG manager with multiple retrieval strategies.
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key for cloud instance
            embedding_model: HuggingFace embedding model name
            rerank_model: Cross-encoder model for reranking
            config: Retrieval configuration
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.config = config or RetrievalConfig()
        
        self._client = None
        self._embeddings = None
        self._reranker = None
        self._llm = None
        self._chat_llm = None
        self._embedding_cache = {}
        self._document_store = {}  # For parent document retrieval
        
        # Initialize Gemini
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in config.")
        genai.configure(api_key=GEMINI_API_KEY)
        
        logging.info("Advanced RAG Manager initialized with config: %s", self.config)

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key
            )
        return self._client
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create embedding model."""
        if self._embeddings is None:
            logging.info("Initializing HuggingFace embeddings: %s", self.embedding_model)
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    @property
    def reranker(self) -> CrossEncoder:
        """Get or create reranking model."""
        if self._reranker is None:
            logging.info("Loading reranking model: %s", self.rerank_model)
            self._reranker = CrossEncoder(self.rerank_model)
        return self._reranker
    
    @property
    def llm(self) -> GoogleGenerativeAI:
        """Get or create Gemini LLM."""
        if self._llm is None:
            self._llm = GoogleGenerativeAI(
                google_api_key=GEMINI_API_KEY,
                model="gemini-pro",
                temperature=0.1
            )
        return self._llm
    
    @property
    def chat_llm(self) -> ChatGoogleGenerativeAI:
        """Get or create Gemini Chat LLM."""
        if self._chat_llm is None:
            self._chat_llm = ChatGoogleGenerativeAI(
                google_api_key=GEMINI_API_KEY,
                model="gemini-pro",
                temperature=0.1
            )
        return self._chat_llm

    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID."""
        return hashlib.md5(content.encode()).hexdigest()

    def _create_hierarchical_chunks(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """Create hierarchical chunks (parent and child documents)."""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size * 4,  # Larger parent chunks
            chunk_overlap=self.config.chunk_overlap * 2
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        parent_docs = parent_splitter.split_documents(documents)
        child_docs = []
        
        for i, parent_doc in enumerate(parent_docs):
            parent_id = f"parent_{i}"
            parent_doc.metadata['parent_id'] = parent_id
            self._document_store[parent_id] = parent_doc
            
            # Create child chunks from parent
            child_chunks = child_splitter.split_documents([parent_doc])
            for j, child_doc in enumerate(child_chunks):
                child_doc.metadata['parent_id'] = parent_id
                child_doc.metadata['child_id'] = f"{parent_id}_child_{j}"
                child_docs.append(child_doc)
        
        logging.info("Created %d parent docs and %d child docs", len(parent_docs), len(child_docs))
        return parent_docs, child_docs

    def _generate_hypothetical_questions(self, document: Document) -> List[str]:
        """Generate hypothetical questions for each document chunk."""
        prompt = """
        Based on the following document content, generate 3 diverse questions that this content could answer.
        Make the questions specific and varied in style (factual, analytical, practical).
        
        Document content:
        {content}
        
        Questions (one per line):
        """
        
        try:
            response = self.llm.invoke(prompt.format(content=document.page_content[:1000]))
            questions = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('#')]
            return questions[:3]  # Limit to 3 questions
        except Exception as e:
            logging.warning("Failed to generate hypothetical questions: %s", str(e))
            return []

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms."""
        prompt = """
        Expand the following query by generating 2 alternative phrasings that preserve the original intent.
        Focus on synonyms, different question styles, and related concepts.
        
        Original query: {query}
        
        Alternative queries (one per line):
        """
        
        try:
            response = self.llm.invoke(prompt.format(query=query))
            expanded_queries = [query]  # Include original
            alt_queries = [q.strip() for q in response.split('\n') if q.strip()]
            expanded_queries.extend(alt_queries[:2])  # Limit to 2 alternatives
            return expanded_queries
        except Exception as e:
            logging.warning("Failed to expand query: %s", str(e))
            return [query]

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return documents
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Sort documents by score
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in doc_scores[:self.config.rerank_top_k]]
        logging.info("Reranked %d documents, kept top %d", len(documents), len(reranked_docs))
        
        return reranked_docs

    def _apply_mmr(self, query: str, documents: List[Document]) -> List[Document]:
        """Apply Maximal Marginal Relevance for diversity."""
        if len(documents) <= 1:
            return documents
        
        # Get embeddings for query and documents
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        
        # Calculate similarity matrices
        query_doc_sim = cosine_similarity([query_embedding], doc_embeddings)[0]
        doc_doc_sim = cosine_similarity(doc_embeddings)
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(documents)))
        
        # Select first document (highest query similarity)
        first_idx = np.argmax(query_doc_sim)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining documents using MMR
        while remaining_indices and len(selected_indices) < self.config.rerank_top_k:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score
                relevance = query_doc_sim[idx]
                
                # Diversity score (max similarity to already selected)
                if selected_indices:
                    diversity = max(doc_doc_sim[idx][sel_idx] for sel_idx in selected_indices)
                else:
                    diversity = 0
                
                # MMR score
                mmr_score = (self.config.mmr_diversity_threshold * relevance - 
                           (1 - self.config.mmr_diversity_threshold) * diversity)
                mmr_scores.append((idx, mmr_score))
            
            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        mmr_docs = [documents[idx] for idx in selected_indices]
        logging.info("Applied MMR, selected %d diverse documents", len(mmr_docs))
        
        return mmr_docs

def create_advanced_vectorstore(documents: List[Document],
                              collection_name: str = "advanced-rag-collection",
                              rag_manager: Optional[AdvancedRAGManager] = None,
                              **kwargs) -> Tuple[Qdrant, AdvancedRAGManager]:
    """
    Create advanced vectorstore with hierarchical chunking and metadata enrichment.
    
    Args:
        documents: List of LangChain Document objects
        collection_name: Name of the Qdrant collection
        rag_manager: Optional existing RAG manager
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (Qdrant vectorstore, AdvancedRAGManager)
    """
    if not documents:
        raise ValueError("No documents provided to create_advanced_vectorstore.")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in config.")
    
    # Initialize RAG manager
    if rag_manager is None:
        config = RetrievalConfig(**kwargs)
        rag_manager = AdvancedRAGManager(config=config)
    
    # Create hierarchical chunks
    parent_docs, child_docs = rag_manager._create_hierarchical_chunks(documents)
    
    # Enrich documents with hypothetical questions if enabled
    if rag_manager.config.use_hypothetical_questions:
        logging.info("Generating hypothetical questions for documents...")
        for doc in child_docs:
            questions = rag_manager._generate_hypothetical_questions(doc)
            if questions:
                doc.metadata['hypothetical_questions'] = questions
                # Add questions to searchable content
                doc.page_content += "\n\nRelated questions: " + " ".join(questions)
    
    # Create vectorstore from child documents (for granular search)
    logging.info("Creating advanced Qdrant vectorstore with %d child documents", len(child_docs))
    vectorstore = Qdrant.from_documents(
        child_docs,
        rag_manager.embeddings,
        url=rag_manager.qdrant_url,
        api_key=rag_manager.qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True
    )
    
    logging.info("Advanced vectorstore created successfully with collection '%s'", collection_name)
    return vectorstore, rag_manager

def create_advanced_retriever(vectorstore: Qdrant,
                            rag_manager: AdvancedRAGManager,
                            retrieval_strategy: str = "ensemble") -> Any:
    """
    Create advanced retriever with multiple strategies.
    
    Args:
        vectorstore: Qdrant vectorstore
        rag_manager: Advanced RAG manager
        retrieval_strategy: Strategy type ("ensemble", "multi_query", "contextual_compression", "all")
        
    Returns:
        Advanced retriever instance
    """
    # Base semantic retriever
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": rag_manager.config.similarity_top_k}
    )
    
    # BM25 retriever (keyword-based)
    all_docs = []
    try:
        # Get all documents from vectorstore for BM25
        all_docs = vectorstore.similarity_search("", k=10000)  # Get all docs
    except:
        logging.warning("Could not retrieve all documents for BM25, using empty list")
    
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = rag_manager.config.bm25_top_k
    
    if retrieval_strategy == "ensemble" or retrieval_strategy == "all":
        # Ensemble retriever combining semantic and BM25
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=rag_manager.config.ensemble_weights
        )
        base_retriever = ensemble_retriever
    else:
        base_retriever = semantic_retriever
    
    # Multi-query retriever
    if retrieval_strategy == "multi_query" or retrieval_strategy == "all":
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=rag_manager.llm
        )
        base_retriever = multi_query_retriever
    
    # Contextual compression retriever
    if (retrieval_strategy == "contextual_compression" or retrieval_strategy == "all") and rag_manager.config.use_contextual_compression:
        compressor = LLMChainExtractor.from_llm(rag_manager.llm)
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        base_retriever = contextual_retriever
    
    logging.info("Created advanced retriever with strategy: %s", retrieval_strategy)
    return base_retriever

class AdvancedRAGChain:
    """Advanced RAG chain with conversation memory and adaptive retrieval."""
    
    def __init__(self, 
                 vectorstore: Qdrant,
                 rag_manager: AdvancedRAGManager,
                 retrieval_strategy: str = "all"):
        """
        Initialize advanced RAG chain.
        
        Args:
            vectorstore: Qdrant vectorstore
            rag_manager: Advanced RAG manager
            retrieval_strategy: Retrieval strategy to use
        """
        self.vectorstore = vectorstore
        self.rag_manager = rag_manager
        self.retrieval_strategy = retrieval_strategy
        self.retriever = create_advanced_retriever(vectorstore, rag_manager, retrieval_strategy)
        
        # Conversation memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Remember last 5 exchanges
        )
        
        # Enhanced prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""
            You are an advanced AI assistant with access to relevant documents. Use the provided context to answer questions accurately and comprehensively.
            
            Previous conversation:
            {chat_history}
            
            Relevant context from documents:
            {context}
            
            Current question: {question}
            
            Instructions:
            1. Answer based on the provided context when possible
            2. If the context doesn't contain enough information, clearly state what's missing
            3. Provide comprehensive answers with specific details from the context
            4. Consider the conversation history for context
            5. If asked about sources, reference the document metadata when available
            
            Answer:
            """
        )
    
    def query(self, question: str, use_advanced_retrieval: bool = True) -> Dict[str, Any]:
        """
        Query the advanced RAG system.
        
        Args:
            question: User question
            use_advanced_retrieval: Whether to use advanced retrieval techniques
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = datetime.now()
        
        # Query expansion if enabled
        queries = [question]
        if self.rag_manager.config.use_query_expansion:
            queries = self.rag_manager._expand_query(question)
        
        # Retrieve documents for all expanded queries
        all_docs = []
        for query in queries:
            docs = self.retriever.get_relevant_documents(query)
            all_docs.extend(docs)
        
        # Remove duplicates based on content
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
        
        # Apply advanced retrieval techniques
        if use_advanced_retrieval:
            # Reranking
            if len(unique_docs) > self.rag_manager.config.rerank_top_k:
                unique_docs = self.rag_manager._rerank_documents(question, unique_docs)
            
            # MMR for diversity
            if self.rag_manager.config.use_mmr:
                unique_docs = self.rag_manager._apply_mmr(question, unique_docs)
        
        # Parent document retrieval if enabled
        if self.rag_manager.config.use_parent_document_retrieval:
            parent_docs = []
            for doc in unique_docs:
                parent_id = doc.metadata.get('parent_id')
                if parent_id and parent_id in self.rag_manager._document_store:
                    parent_doc = self.rag_manager._document_store[parent_id]
                    if parent_doc not in parent_docs:
                        parent_docs.append(parent_doc)
            
            # Use parent documents for context
            if parent_docs:
                unique_docs = parent_docs[:self.rag_manager.config.rerank_top_k]
        
        # Prepare context
        context = "\n\n".join([
            f"Document {i+1}: {doc.page_content}" 
            for i, doc in enumerate(unique_docs)
        ])
        
        # Get chat history
        chat_history = self.memory.chat_memory.messages
        history_text = "\n".join([
            f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg.content}"
            for i, msg in enumerate(chat_history[-6:])  # Last 3 exchanges
        ])
        
        # Generate answer
        prompt = self.prompt_template.format(
            context=context,
            question=question,
            chat_history=history_text
        )
        
        answer = self.rag_manager.llm.invoke(prompt)
        
        # Update memory
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)
        
        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "parent_id": doc.metadata.get('parent_id'),
                    "hypothetical_questions": doc.metadata.get('hypothetical_questions', [])
                }
                for doc in unique_docs
            ],
            "expanded_queries": queries if len(queries) > 1 else None,
            "num_documents_retrieved": len(all_docs),
            "num_documents_used": len(unique_docs),
            "processing_time_seconds": processing_time,
            "retrieval_strategy": self.retrieval_strategy
        }

# Convenience functions maintaining backward compatibility
def create_vectorstore(documents: List[Document], 
                      collection_name: str = "advanced-rag-collection",
                      **kwargs) -> Qdrant:
    """Create vectorstore with advanced RAG capabilities."""
    vectorstore, _ = create_advanced_vectorstore(documents, collection_name, **kwargs)
    return vectorstore

def load_vectorstore(collection_name: str = "advanced-rag-collection",
                    qdrant_url: str = "http://localhost:6333",
                    qdrant_api_key: Optional[str] = None,
                    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Qdrant:
    """Load existing vectorstore."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in config.")
    
    rag_manager = AdvancedRAGManager(qdrant_url, qdrant_api_key, embedding_model)
    
    logging.info("Loading Qdrant collection '%s'", collection_name)
    
    vectorstore = Qdrant(
        client=rag_manager.client,
        collection_name=collection_name,
        embeddings=rag_manager.embeddings
    )
    
    logging.info("Advanced vectorstore loaded successfully.")
    return vectorstore

def get_retriever(vectorstore: Qdrant, 
                 k: int = 10,
                 retrieval_strategy: str = "ensemble") -> Any:
    """Get advanced retriever."""
    rag_manager = AdvancedRAGManager()
    rag_manager.config.similarity_top_k = k
    rag_manager.config.bm25_top_k = k
    
    return create_advanced_retriever(vectorstore, rag_manager, retrieval_strategy)