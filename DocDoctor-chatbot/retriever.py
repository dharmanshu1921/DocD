import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Structure for retrieval results with metadata"""
    content: str
    score: Optional[float] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedRetriever:
    """Enhanced retriever with better error handling and configuration"""
    
    def __init__(self, base_retriever, max_docs: int = 5, min_score: float = 0.0):
        self.base_retriever = base_retriever
        self.max_docs = max_docs
        self.min_score = min_score
    
    def ask_question(self, query: str, **kwargs) -> RetrievalResult:
        """
        Enhanced question answering with multiple retrieval strategies
        
        Args:
            query: The question to search for
            **kwargs: Additional parameters for retrieval
            
        Returns:
            RetrievalResult with content and metadata
        """
        if not query or not query.strip():
            return RetrievalResult(
                content="Please provide a valid question.",
                metadata={"error": "empty_query"}
            )
        
        try:
            # Get relevant documents
            documents = self._get_documents(query, **kwargs)
            
            if not documents:
                return RetrievalResult(
                    content="No relevant information found for your query.",
                    metadata={"query": query, "doc_count": 0}
                )
            
            # Process and combine results
            result = self._process_documents(documents, query)
            return result
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return RetrievalResult(
                content="Sorry, I encountered an error while searching for information.",
                metadata={"error": str(e), "query": query}
            )
    
    def _get_documents(self, query: str, **kwargs):
        """Get documents with error handling"""
        try:
            # Try similarity search with scores if available
            if hasattr(self.base_retriever, 'similarity_search_with_score'):
                docs_with_scores = self.base_retriever.similarity_search_with_score(
                    query, k=self.max_docs, **kwargs
                )
                # Filter by minimum score if specified
                if self.min_score > 0:
                    docs_with_scores = [(doc, score) for doc, score in docs_with_scores 
                                      if score >= self.min_score]
                return docs_with_scores
            else:
                # Fallback to regular retrieval
                return self.base_retriever.get_relevant_documents(query)
                
        except AttributeError:
            # Handle different retriever interfaces
            if hasattr(self.base_retriever, 'retrieve'):
                return self.base_retriever.retrieve(query)
            else:
                raise ValueError("Unsupported retriever interface")
    
    def _process_documents(self, documents, query: str) -> RetrievalResult:
        """Process retrieved documents into a coherent response"""
        if not documents:
            return RetrievalResult(content="No relevant documents found.")
        
        # Handle documents with scores vs without
        if isinstance(documents[0], tuple):
            # Documents with scores
            doc_contents = []
            scores = []
            sources = []
            
            for doc, score in documents:
                content = getattr(doc, 'page_content', str(doc))
                doc_contents.append(content)
                scores.append(score)
                
                # Extract source if available
                source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                sources.append(source)
            
            # Combine top results intelligently
            combined_content = self._combine_contents(doc_contents[:3], query)
            
            return RetrievalResult(
                content=combined_content,
                score=max(scores) if scores else None,
                source=sources[0] if sources else None,
                metadata={
                    "doc_count": len(documents),
                    "all_scores": scores,
                    "all_sources": list(set(sources)),
                    "query": query
                }
            )
        else:
            # Documents without scores
            doc_contents = [getattr(doc, 'page_content', str(doc)) for doc in documents]
            sources = [getattr(doc, 'metadata', {}).get('source', 'Unknown') for doc in documents]
            
            combined_content = self._combine_contents(doc_contents[:3], query)
            
            return RetrievalResult(
                content=combined_content,
                source=sources[0] if sources else None,
                metadata={
                    "doc_count": len(documents),
                    "all_sources": list(set(sources)),
                    "query": query
                }
            )
    
    def _combine_contents(self, contents: List[str], query: str) -> str:
        """Intelligently combine multiple document contents"""
        if not contents:
            return "No content available."
        
        if len(contents) == 1:
            return contents[0].strip()
        
        # For multiple documents, create a more comprehensive response
        combined = []
        seen_content = set()
        
        for i, content in enumerate(contents):
            content = content.strip()
            if content and content not in seen_content:
                if i == 0:
                    combined.append(content)
                else:
                    # Add additional context with separator
                    combined.append(f"\n\nAdditional context:\n{content}")
                seen_content.add(content)
        
        return "".join(combined)
    
    def get_multiple_answers(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Get multiple potential answers for comparison"""
        try:
            documents = self._get_documents(query, k=max(top_k, self.max_docs))
            results = []
            
            if isinstance(documents[0], tuple):
                for doc, score in documents[:top_k]:
                    content = getattr(doc, 'page_content', str(doc))
                    source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                    
                    results.append(RetrievalResult(
                        content=content.strip(),
                        score=score,
                        source=source,
                        metadata={"query": query}
                    ))
            else:
                for doc in documents[:top_k]:
                    content = getattr(doc, 'page_content', str(doc))
                    source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
                    
                    results.append(RetrievalResult(
                        content=content.strip(),
                        source=source,
                        metadata={"query": query}
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting multiple answers: {e}")
            return [RetrievalResult(
                content="Error retrieving multiple answers.",
                metadata={"error": str(e)}
            )]

# Convenience functions for backward compatibility
def ask_question(retriever, query: str) -> str:
    """Simple wrapper for backward compatibility"""
    enhanced = EnhancedRetriever(retriever)
    result = enhanced.ask_question(query)
    return result.content

def ask_question_with_metadata(retriever, query: str, **kwargs) -> RetrievalResult:
    """Enhanced version returning full result object"""
    enhanced = EnhancedRetriever(retriever, **kwargs)
    return enhanced.ask_question(query)

# Example usage
if __name__ == "__main__":
    # Example with mock retriever
    class MockRetriever:
        def get_relevant_documents(self, query):
            # Mock implementation
            class MockDoc:
                def __init__(self, content, metadata=None):
                    self.page_content = content
                    self.metadata = metadata or {}
            
            return [
                MockDoc("This is relevant information about your query.", 
                       {"source": "document1.pdf"}),
                MockDoc("Additional context that might be helpful.",
                       {"source": "document2.pdf"})
            ]
    
    # Test the enhanced retriever
    mock_retriever = MockRetriever()
    enhanced = EnhancedRetriever(mock_retriever)
    
    result = enhanced.ask_question("What is the answer?")
    print(f"Answer: {result.content}")
    print(f"Metadata: {result.metadata}")