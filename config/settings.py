import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LLMConfig:
    provider: str = "groq"
    model: str = "deepseek-r1-distill-llama-70b"
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60

@dataclass_json
@dataclass
class VectorDBConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "rag_documents"
    embedding_dim: int = 384
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    nlist: int = 1024

@dataclass_json
@dataclass
class RetrievalConfig:
    top_k_bm25: int = 20
    top_k_vector: int = 20
    top_k_kg: int = 15
    final_top_k: int = 10
    rerank_threshold: float = 0.5
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    chunk_overlap: int = 50

@dataclass_json
@dataclass
class RAGConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    max_context_length: int = 16000
    adaptive_retrieval_max_iterations: int = 3
    log_level: str = "INFO"
    enable_caching: bool = True
    cache_ttl: int = 3600