import os
import sys
import asyncio
import aiohttp
import tempfile
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add the parent directory to the path to import fusion_rag modules
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
import uvicorn

# Import FusionRAG components
from fusion_rag.pipelines.rag_pipeline import RAGPipeline
from fusion_rag.loaders.document_loader_factory import DocumentLoaderFactory
from fusion_rag.core_utils.llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FusionRAG Webhook API",
    description="Webhook API for FusionRAG with Groq LLM integration",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Pydantic models
class WebhookRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class WebhookResponse(BaseModel):
    answers: List[str]

# Configuration for FusionRAG
RAG_CONFIG = {
    'embedding_engine': 'huggingface',
    'embedding_model': 'all-MiniLM-L6-v2',
    'retrieval_engines': ['bm25', 'vector_db'],
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'max_context_length': 4000,
    'max_adaptive_iterations': 3,
    'use_hyde': False
}

# Global variables
rag_pipeline = None
groq_client = None

def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate API key"""
    api_key = credentials.credentials
    expected_key = os.getenv("WEBHOOK_API_KEY", "your-secret-api-key")
    
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

class GroqLLMIntegration:
    """Groq API integration for LLM responses"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama3-8b-8192"  # Using Llama3 model via Groq
    
    async def generate_response(self, query: str, context: str, max_tokens: int = 1000) -> str:
        """Generate response using Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = self._build_prompt(query, context)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. Provide accurate, concise answers based only on the given context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for more consistent answers
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"Groq API error: {response.status} - {error_text}")
                        raise Exception(f"Groq API error: {response.status}")
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM"""
        return f"""Context:
{context}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so. Provide a clear, concise answer."""

async def download_document(url: str) -> str:
    """Download document from URL and save to temporary file"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        content = await response.read()
                        temp_file.write(content)
                        temp_file.flush()
                        return temp_file.name
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to download document: {response.status}"
                    )
    except Exception as e:
        logger.error(f"Error downloading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading document: {str(e)}"
        )

async def initialize_rag_pipeline():
    """Initialize the RAG pipeline and Groq client"""
    global rag_pipeline, groq_client
    
    try:
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        groq_client = GroqLLMIntegration(groq_api_key)
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(RAG_CONFIG)
        
        logger.info("RAG pipeline and Groq client initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    await initialize_rag_pipeline()

@app.post("/hackrx/run", response_model=WebhookResponse)
async def run_webhook(
    request: WebhookRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Main webhook endpoint for processing documents and answering questions
    """
    try:
        logger.info(f"Processing webhook request with {len(request.questions)} questions")
        
        # Download the document
        logger.info(f"Downloading document from: {request.documents}")
        temp_file_path = await download_document(str(request.documents))
        
        try:
            # Load document into RAG pipeline
            rag_pipeline.load_documents([temp_file_path])
            rag_pipeline.index_documents()
            
            logger.info("Document loaded and indexed successfully")
            
            # Process each question
            answers = []
            for i, question in enumerate(request.questions):
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
                
                try:
                    # Get relevant context using RAG pipeline
                    query_result = rag_pipeline.query(question)
                    context = query_result['context']
                    
                    # Generate answer using Groq
                    answer = await groq_client.generate_response(question, context)
                    answers.append(answer)
                    
                    logger.info(f"Generated answer for question {i+1}")
                    
                except Exception as e:
                    logger.error(f"Error processing question {i+1}: {str(e)}")
                    answers.append(f"Error processing question: {str(e)}")
            
            return WebhookResponse(answers=answers)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in webhook processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "FusionRAG Webhook API"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FusionRAG Webhook API",
        "version": "1.0.0",
        "endpoints": {
            "webhook": "/hackrx/run",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
