import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the FastAPI backend"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # File upload settings
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
    
    # RAG settings
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
    DEFAULT_MAX_CONTEXT_LENGTH = int(os.getenv("DEFAULT_MAX_CONTEXT_LENGTH", 4000))
    
    @classmethod
    def validate_config(cls):
        """Validate that required environment variables are set"""
        missing_vars = []
        
        if not cls.GROQ_API_KEY:
            missing_vars.append("GROQ_API_KEY")
        if not cls.GOOGLE_API_KEY:
            missing_vars.append("GOOGLE_API_KEY")
        if not cls.OPENAI_API_KEY:
            missing_vars.append("OPENAI_API_KEY")
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
