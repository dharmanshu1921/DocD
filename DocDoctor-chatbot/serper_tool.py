import asyncio
import aiohttp
import requests
import time
import logging
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote_plus
from functools import wraps
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import SERPER_API_KEY

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_search.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Enumeration of search types supported by Serper API."""
    SEARCH = "search"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"
    PLACES = "places"
    SCHOLAR = "scholar"

class SearchRegion(Enum):
    """Common search regions for localized results."""
    US = "us"
    UK = "uk"
    CA = "ca"
    AU = "au"
    IN = "in"
    DE = "de"
    FR = "fr"
    JP = "jp"
    CN = "cn"

@dataclass
class SearchResult:
    """Enhanced search result data structure."""
    title: str
    url: str
    snippet: str
    full_snippet: str
    domain: str
    search_rank: int
    result_type: str = "organic"
    thumbnail: Optional[str] = None
    date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    word_count: int = 0
    
    def __post_init__(self):
        """Calculate additional metrics after initialization."""
        self.domain = urlparse(self.url).netloc.replace('www.', '')
        self.word_count = len(self.snippet.split())
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'full_snippet': self.full_snippet,
            'domain': self.domain,
            'search_rank': self.search_rank,
            'result_type': self.result_type,
            'thumbnail': self.thumbnail,
            'date': self.date,
            'metadata': self.metadata,
            'relevance_score': self.relevance_score,
            'word_count': self.word_count
        }

@dataclass
class SearchConfig:
    """Configuration for web searches."""
    search_type: SearchType = SearchType.SEARCH
    num_results: int = 10
    max_snippet_words: int = 100
    region: Optional[SearchRegion] = None
    language: Optional[str] = None
    time_range: Optional[str] = None  # 'day', 'week', 'month', 'year'
    safe_search: bool = True
    include_images: bool = False
    include_videos: bool = False
    timeout: int = 30
    retries: int = 3
    cache_ttl: int = 3600  # Cache time-to-live in seconds

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.calls.append(now)

class SearchCache:
    """Simple in-memory cache for search results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Tuple[datetime, List[SearchResult]]] = {}
        self.max_size = max_size
    
    def _generate_key(self, query: str, config: SearchConfig) -> str:
        """Generate cache key from query and config."""
        config_str = f"{config.search_type.value}_{config.num_results}_{config.region}_{config.language}"
        return hashlib.md5(f"{query}_{config_str}".encode()).hexdigest()
    
    def get(self, query: str, config: SearchConfig) -> Optional[List[SearchResult]]:
        """Get cached results if valid."""
        key = self._generate_key(query, config)
        if key in self.cache:
            timestamp, results = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=config.cache_ttl):
                logger.debug(f"Cache hit for query: {query}")
                return results
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, config: SearchConfig, results: List[SearchResult]):
        """Cache search results."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
        
        key = self._generate_key(query, config)
        self.cache[key] = (datetime.now(), results)
        logger.debug(f"Cached results for query: {query}")

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed requests with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator

class EnhancedWebSearcher:
    """Enhanced web searcher with advanced features."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or SERPER_API_KEY
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is not set. Please provide your API key.")
        
        self.rate_limiter = RateLimiter(calls_per_minute=100)
        self.cache = SearchCache()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EnhancedWebSearcher/2.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        query_words = set(query.lower().split())
        
        # Title relevance (40% weight)
        title_words = set(result.get('title', '').lower().split())
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        score += title_overlap * 0.4
        
        # Snippet relevance (30% weight)
        snippet_words = set(result.get('snippet', '').lower().split())
        snippet_overlap = len(query_words.intersection(snippet_words)) / len(query_words) if query_words else 0
        score += snippet_overlap * 0.3
        
        # Position bias (30% weight) - higher positions get higher scores
        position = result.get('position', 10)
        position_score = max(0, (10 - position) / 10)
        score += position_score * 0.3
        
        return min(score, 1.0)
    
    def _process_snippet(self, snippet: str, max_words: int) -> Tuple[str, str]:
        """Process and truncate snippet."""
        if not snippet:
            return "", ""
        
        # Clean the snippet
        snippet = re.sub(r'\s+', ' ', snippet).strip()
        words = snippet.split()
        
        if len(words) <= max_words:
            return snippet, snippet
        
        truncated = ' '.join(words[:max_words]) + "... [Read more]"
        return truncated, snippet
    
    def _extract_search_results(self, data: Dict[str, Any], query: str, config: SearchConfig) -> List[SearchResult]:
        """Extract and process search results from API response."""
        results = []
        
        # Process answer box first (highest priority)
        if "answerBox" in data and data["answerBox"]:
            answer_box = data["answerBox"]
            url = answer_box.get("url", "")
            snippet = (answer_box.get("answer") or 
                      answer_box.get("snippet") or 
                      answer_box.get("title") or "")
            
            if url and snippet:
                truncated_snippet, full_snippet = self._process_snippet(snippet, config.max_snippet_words)
                result = SearchResult(
                    title=answer_box.get("title", "Answer Box"),
                    url=url,
                    snippet=truncated_snippet,
                    full_snippet=full_snippet,
                    domain="",
                    search_rank=0,
                    result_type="answer_box",
                    metadata={"source": "answer_box"}
                )
                result.relevance_score = self._calculate_relevance_score(answer_box, query)
                results.append(result)
        
        # Process organic results
        for idx, result_data in enumerate(data.get("organic", []), start=1):
            if len(results) >= config.num_results:
                break
            
            url = result_data.get("link", "")
            title = result_data.get("title", "")
            snippet = result_data.get("snippet", "")
            
            if not url or not snippet:
                continue
            
            truncated_snippet, full_snippet = self._process_snippet(snippet, config.max_snippet_words)
            
            result = SearchResult(
                title=title,
                url=url,
                snippet=truncated_snippet,
                full_snippet=full_snippet,
                domain="",
                search_rank=idx,
                result_type="organic",
                date=result_data.get("date"),
                metadata={
                    "position": result_data.get("position", idx),
                    "source": "organic"
                }
            )
            result.relevance_score = self._calculate_relevance_score(result_data, query)
            results.append(result)
        
        # Process knowledge graph if available
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            if kg.get("description"):
                truncated_snippet, full_snippet = self._process_snippet(
                    kg["description"], config.max_snippet_words
                )
                result = SearchResult(
                    title=kg.get("title", "Knowledge Graph"),
                    url=kg.get("descriptionLink", ""),
                    snippet=truncated_snippet,
                    full_snippet=full_snippet,
                    domain="",
                    search_rank=-1,  # Highest priority
                    result_type="knowledge_graph",
                    thumbnail=kg.get("imageUrl"),
                    metadata={"source": "knowledge_graph", "type": kg.get("type")}
                )
                results.insert(0, result)  # Insert at beginning
        
        return results[:config.num_results]
    
    @retry_on_failure(max_retries=3)
    def search(self, query: str, config: Optional[SearchConfig] = None) -> List[SearchResult]:
        """
        Perform enhanced web search with comprehensive features.
        
        Args:
            query: Search query string.
            config: Search configuration options.
            
        Returns:
            List of SearchResult objects with enhanced metadata.
            
        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If API request fails.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        config = config or SearchConfig()
        
        # Check cache first
        cached_results = self.cache.get(query, config)
        if cached_results:
            return cached_results
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Build API endpoint and payload
        endpoint_map = {
            SearchType.SEARCH: "https://google.serper.dev/search",
            SearchType.NEWS: "https://google.serper.dev/news",
            SearchType.IMAGES: "https://google.serper.dev/images",
            SearchType.VIDEOS: "https://google.serper.dev/videos",
            SearchType.PLACES: "https://google.serper.dev/places",
            SearchType.SCHOLAR: "https://google.serper.dev/scholar"
        }
        
        url = endpoint_map[config.search_type]
        headers = {"X-API-KEY": self.api_key}
        
        payload = {
            "q": query,
            "num": min(config.num_results, 100)  # Serper API limit
        }
        
        # Add optional parameters
        if config.region:
            payload["gl"] = config.region.value
        if config.language:
            payload["hl"] = config.language
        if config.time_range:
            payload["tbs"] = f"qdr:{config.time_range[0]}"  # 'd', 'w', 'm', 'y'
        if not config.safe_search:
            payload["safe"] = "off"
        
        logger.info(f"Searching for: '{query}' with {config.search_type.value} type")
        
        try:
            start_time = time.time()
            response = self.session.post(
                url, 
                json=payload, 
                headers=headers, 
                timeout=config.timeout
            )
            response.raise_for_status()
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f} seconds")
            
            data = response.json()
            results = self._extract_search_results(data, query, config)
            
            # Sort by relevance score
            results.sort(key=lambda r: r.relevance_score, reverse=True)
            
            # Cache results
            self.cache.set(query, config, results)
            
            logger.info(f"Retrieved {len(results)} results for query: '{query}'")
            return results
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Search request timed out after {config.timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse API response: {e}")
    
    async def async_search(self, query: str, config: Optional[SearchConfig] = None) -> List[SearchResult]:
        """Asynchronous version of search method."""
        config = config or SearchConfig()
        
        # Check cache first
        cached_results = self.cache.get(query, config)
        if cached_results:
            return cached_results
        
        endpoint_map = {
            SearchType.SEARCH: "https://google.serper.dev/search",
            SearchType.NEWS: "https://google.serper.dev/news",
            SearchType.IMAGES: "https://google.serper.dev/images",
            SearchType.VIDEOS: "https://google.serper.dev/videos"
        }
        
        url = endpoint_map[config.search_type]
        headers = {"X-API-KEY": self.api_key}
        payload = {"q": query, "num": config.num_results}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url, 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    results = self._extract_search_results(data, query, config)
                    results.sort(key=lambda r: r.relevance_score, reverse=True)
                    
                    self.cache.set(query, config, results)
                    return results
                    
            except asyncio.TimeoutError:
                raise RuntimeError(f"Async search request timed out after {config.timeout} seconds")
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Async API request failed: {e}")
    
    def multi_search(self, queries: List[str], config: Optional[SearchConfig] = None) -> Dict[str, List[SearchResult]]:
        """Perform multiple searches concurrently."""
        config = config or SearchConfig()
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {
                executor.submit(self.search, query, config): query 
                for query in queries
            }
            
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results[query] = future.result()
                except Exception as e:
                    logger.error(f"Search failed for query '{query}': {e}")
                    results[query] = []
        
        return results
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for a query."""
        # This would require a different API endpoint or service
        # For now, return basic query variations
        suggestions = [
            f"{query} tutorial",
            f"{query} guide",
            f"{query} examples",
            f"how to {query}",
            f"{query} best practices"
        ]
        return suggestions

# Global searcher instance
_searcher = None

def get_searcher() -> EnhancedWebSearcher:
    """Get global searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = EnhancedWebSearcher()
    return _searcher

# Backward compatibility function
def web_search(
    query: str, 
    num_results: int = 5,
    search_type: str = "search",
    max_snippet_words: int = 100
) -> List[Dict[str, str]]:
    """
    Backward compatible web search function.
    
    Args:
        query: Search query string.
        num_results: Number of results to return (default: 5).
        search_type: Type of search (default: "search").
        max_snippet_words: Maximum words in snippet (default: 100).
        
    Returns:
        List of dictionaries with 'Source URL', 'Answer', and 'FullAnswer' keys.
    """
    try:
        search_type_enum = SearchType(search_type)
    except ValueError:
        search_type_enum = SearchType.SEARCH
    
    config = SearchConfig(
        search_type=search_type_enum,
        num_results=num_results,
        max_snippet_words=max_snippet_words
    )
    
    searcher = get_searcher()
    results = searcher.search(query, config)
    
    # Convert to backward compatible format
    return [
        {
            "Source URL": result.url,
            "Answer": result.snippet,
            "FullAnswer": result.full_snippet
        }
        for result in results
    ]