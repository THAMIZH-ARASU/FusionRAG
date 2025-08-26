import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logging import RAGLogger
from core.exceptions import LLMException
from core.interfaces import Document, QueryContext
from config.settings import LLMConfig

@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str

class GroqLLMClient:
    """GROQ API client for LLM generation."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = RAGLogger("groq_llm_client")
        self.base_url = "https://api.groq.com/openai/v1"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using GROQ API."""
        with self.logger.log_operation("llm_generation", model=self.config.model):
            if not self.session:
                raise LLMException("LLM client not initialized. Use async context manager.")
            
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "stream": False
            }
            
            try:
                async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMException(f"GROQ API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    return LLMResponse(
                        content=result["choices"][0]["message"]["content"],
                        usage=result.get("usage", {}),
                        model=result["model"],
                        finish_reason=result["choices"][0]["finish_reason"]
                    )
            
            except aiohttp.ClientError as e:
                raise LLMException(f"HTTP client error: {e}")
            except json.JSONDecodeError as e:
                raise LLMException(f"JSON decode error: {e}")
    
    async def generate_with_context(
        self,
        query_context: QueryContext,
        context_documents: List[Document],
        system_message: Optional[str] = None
    ) -> LLMResponse:
        """Generate response with RAG context."""
        # Build context from documents
        context_parts = []
        for doc in context_documents:
            context_parts.append(doc.content)
        
        context_text = "\n\n".join(context_parts)
        
        # Create RAG prompt
        rag_prompt = f"""Based on the following context information, please provide a comprehensive and accurate answer to the user's question.

Context Information:
{context_text}

User Question: {query_context.original_query}

Instructions:
1. Answer based ONLY on the provided context information
2. If the context doesn't contain enough information to answer the question, clearly state this
3. Include inline citations using [Document X] format where X is the document number
4. Be concise but thorough
5. If there are conflicting pieces of information, acknowledge this and explain the different perspectives

Answer:"""
        
        default_system = """You are a helpful AI assistant that provides accurate, well-cited answers based on provided context. Always ground your responses in the given information and cite your sources appropriately."""
        
        return await self.generate(
            prompt=rag_prompt,
            system_message=system_message or default_system
        )