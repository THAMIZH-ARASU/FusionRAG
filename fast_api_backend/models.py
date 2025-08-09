from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class RetrievalMethod(str, Enum):
    BM25 = "bm25"
    VECTOR_DB = "vector_db"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    HYBRID = "hybrid"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    GOOGLE = "google"

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    retrieval_methods: List[RetrievalMethod] = Field(default=[RetrievalMethod.HYBRID], description="Retrieval methods to use")
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens for response")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for response generation")
    use_adaptive_retrieval: bool = Field(default=True, description="Whether to use adaptive retrieval")

class QueryResponse(BaseModel):
    query: str
    answer: str
    context: str
    retrieval_methods_used: List[str]
    metrics: Dict[str, Any]
    retrieval_logs: Dict[str, Any]
    processing_time: float

class DocumentUploadResponse(BaseModel):
    filename: str
    status: str
    message: str
    document_id: Optional[str] = None

class RAGComparisonRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    retrieval_methods: List[RetrievalMethod] = Field(..., description="Retrieval methods to compare")
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider to use")

class RAGComparisonResponse(BaseModel):
    query: str
    results: Dict[str, QueryResponse]
    comparison_metrics: Dict[str, Any]
    processing_time: float

class SystemStatus(BaseModel):
    status: str
    available_retrieval_methods: List[str]
    available_llm_providers: List[str]
    loaded_documents: int
    indexed_documents: int
