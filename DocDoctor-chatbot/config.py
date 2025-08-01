import os
from dotenv import load_dotenv

load_dotenv()

# Search API configuration
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# OpenAI configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google AI configurations
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# MySQL configurations
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Qdrant Vector Database configurations
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Default to local
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # For cloud instances

# Embedding Model configuration (default to 'sentence-transformers/all-MiniLM-L6-v2')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")