import os
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import Config
from models import (
    QueryRequest, QueryResponse, DocumentUploadResponse, 
    RAGComparisonRequest, RAGComparisonResponse, SystemStatus,
    RetrievalMethod, LLMProvider
)
from llm_integration import LLMIntegration

# Import FusionRAG components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fusion_rag'))

try:
    from fusion_rag.pipelines.rag_factory import RAGPipelineFactory
    from fusion_rag.core_utils.llm_integration import LLMIntegration as FusionLLMIntegration
except ImportError as e:
    print(f"Warning: Could not import FusionRAG components: {e}")
    RAGPipelineFactory = None
    FusionLLMIntegration = None

app = FastAPI(
    title="FusionRAG API",
    description="A sophisticated Retrieval-Augmented Generation system with multiple retrieval strategies",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store pipeline and documents
rag_pipeline = None
loaded_documents = []
indexed_documents = []

def get_rag_pipeline():
    """Get or create RAG pipeline"""
    global rag_pipeline
    if rag_pipeline is None:
        if RAGPipelineFactory is None:
            # Create a mock pipeline for testing
            rag_pipeline = MockRAGPipeline()
        else:
            config = {
                'embedding_engine': 'huggingface',
                'embedding_model': 'all-MiniLM-L6-v2',
                'retrieval_engines': ['bm25', 'vector_db'],
                'chunk_size': Config.DEFAULT_CHUNK_SIZE,
                'chunk_overlap': Config.DEFAULT_CHUNK_OVERLAP,
                'max_context_length': Config.DEFAULT_MAX_CONTEXT_LENGTH,
                'max_adaptive_iterations': 3,
                'use_hyde': False
            }
            rag_pipeline = RAGPipelineFactory.create_basic_pipeline()
    return rag_pipeline

class MockRAGPipeline:
    """Mock RAG pipeline for testing when FusionRAG is not available"""
    
    def __init__(self):
        self.documents = []
        self.indexed = False
        self.document_contents = {}  # Store actual document contents
    
    def load_documents(self, file_paths):
        """Mock document loading - actually read the files"""
        print(f"Mock: Loading documents: {file_paths}")
        for file_path in file_paths:
            try:
                content = self._read_file_content(file_path)
                if content and not content.startswith("Error reading"):
                    self.document_contents[file_path] = content
                    print(f"Mock: Loaded content from {file_path} ({len(content)} characters)")
                    print(f"Mock: Content preview: {content[:200]}...")
                else:
                    print(f"Mock: No content extracted from {file_path}")
            except Exception as e:
                print(f"Mock: Error reading {file_path}: {e}")
                self.document_contents[file_path] = f"Error reading file: {str(e)}"
        
        self.documents.extend(file_paths)
        print(f"Mock: Loaded {len(file_paths)} documents")
        print(f"Mock: Total document contents: {list(self.document_contents.keys())}")
        
        # Auto-index documents when they are loaded
        if self.document_contents:
            self.indexed = True
            print(f"Mock: Auto-indexed {len(self.document_contents)} documents")
    
    def _read_file_content(self, file_path):
        """Read content from different file types"""
        file_extension = file_path.lower().split('.')[-1]
        
        try:
            if file_extension == 'txt' or file_extension == 'md':
                # Read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif file_extension == 'pdf':
                # Read as PDF
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        content = ""
                        for page in pdf_reader.pages:
                            content += page.extract_text() + "\n"
                        return content
                except ImportError:
                    return f"PyPDF2 not available for reading PDF: {file_path}"
                except Exception as e:
                    return f"Error reading PDF {file_path}: {str(e)}"
            
            elif file_extension == 'docx':
                # Read as DOCX
                try:
                    from docx import Document
                    doc = Document(file_path)
                    content = ""
                    for paragraph in doc.paragraphs:
                        content += paragraph.text + "\n"
                    return content
                except ImportError:
                    return f"python-docx not available for reading DOCX: {file_path}"
                except Exception as e:
                    return f"Error reading DOCX {file_path}: {str(e)}"
            
            else:
                # Try to read as text
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    return f"Could not decode file as text: {file_path}"
        
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"
    
    def index_documents(self):
        """Mock document indexing"""
        if not self.document_contents:
            print("Mock: No documents to index")
            return
        
        self.indexed = True
        print(f"Mock: Indexed {len(self.document_contents)} documents")
    
    def query(self, query, use_adaptive_retrieval=True):
        """Mock query processing - actually search through document contents"""
        print(f"Mock: Query received: '{query}'")
        print(f"Mock: Indexed status: {self.indexed}")
        print(f"Mock: Document contents keys: {list(self.document_contents.keys())}")
        
        # Auto-index if not indexed but documents are loaded
        if not self.indexed and self.document_contents:
            print("Mock: Auto-indexing documents")
            self.indexed = True
        
        if not self.document_contents:
            print("Mock: No document contents")
            return {
                'context': "No documents loaded. Please upload documents first.",
                'documents': self.documents,
                'scores': []
            }
        
        # Simple keyword-based search through document contents
        relevant_content = []
        query_terms = query.lower().split()
        print(f"Mock: Query terms: {query_terms}")
        
        for file_path, content in self.document_contents.items():
            print(f"Mock: Processing file: {file_path}")
            if not content or content.startswith("Error reading"):
                print(f"Mock: Skipping file {file_path} - content issue")
                continue
                
            content_lower = content.lower()
            relevance_score = 0
            
            # Calculate relevance based on term frequency and partial matches
            for term in query_terms:
                # Exact match
                if term in content_lower:
                    relevance_score += content_lower.count(term) * 2
                
                # Partial match (for misspellings and variations)
                for word in content_lower.split():
                    if term in word or word in term:
                        relevance_score += 1
            
            print(f"Mock: Relevance score for {file_path}: {relevance_score}")
            
            if relevance_score > 0:
                # Extract a snippet around the most relevant part
                snippet = self._extract_relevant_snippet(content, query_terms, 500)
                filename = file_path.split('/')[-1] if '/' in file_path else file_path
                relevant_content.append(f"[Source: {filename}]\n{snippet}\n")
                print(f"Mock: Added relevant content from {filename}")
        
        if relevant_content:
            context = "\n\n---\n\n".join(relevant_content)
            print(f"Mock: Returning context with {len(context)} characters")
        else:
            # If no relevant content found, return a sample of the content
            context_parts = []
            for file_path, content in self.document_contents.items():
                if content and not content.startswith("Error reading"):
                    filename = file_path.split('/')[-1] if '/' in file_path else file_path
                    sample = content[:500] + "..." if len(content) > 500 else content
                    context_parts.append(f"[Source: {filename}]\n{sample}\n")
            
            if context_parts:
                context = "\n\n---\n\n".join(context_parts)
                context = f"No specific relevant content found for query: '{query}'. Here's a sample of available content:\n\n{context}"
            else:
                context = f"No relevant content found for query: '{query}'. Available documents: {list(self.document_contents.keys())}"
            
            print(f"Mock: No relevant content found, returning: {context[:200]}...")
        
        return {
            'context': context,
            'documents': self.documents,
            'scores': [0.8] * len(self.documents)
        }
    
    def _extract_relevant_snippet(self, content, query_terms, max_length=500):
        """Extract a relevant snippet from the content"""
        if not content:
            return ""
            
        content_lower = content.lower()
        
        # Find the position of the first occurrence of any query term
        best_pos = -1
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
        
        if best_pos == -1:
            # If no terms found, return the beginning
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # Extract snippet around the best position
        start = max(0, best_pos - max_length // 2)
        end = min(len(content), start + max_length)
        
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def get_explainable_logs(self, result):
        """Mock explainable logs"""
        return {
            'retrieval_methods': ['mock'],
            'processing_time': 0.1,
            'context_length': len(result.get('context', ''))
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    try:
        # Create upload directory
        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
        
        # Initialize RAG pipeline
        get_rag_pipeline()
        
        print("FusionRAG API started successfully!")
    except Exception as e:
        print(f"Error during startup: {e}")
        # Don't raise the error, just log it

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "FusionRAG API is running!", "version": "1.0.0"}

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint"""
    pipeline = get_rag_pipeline()
    return SystemStatus(
        status="healthy",
        available_retrieval_methods=[method.value for method in RetrievalMethod],
        available_llm_providers=[provider.value for provider in LLMProvider],
        loaded_documents=len(loaded_documents),
        indexed_documents=len(indexed_documents)
    )

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document"""
    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in Config.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {Config.SUPPORTED_EXTENSIONS}"
            )
        
        # Validate file size
        if hasattr(file, 'size') and file.size and file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE} bytes"
            )
        
        # Save file
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load document into pipeline
        pipeline = get_rag_pipeline()
        pipeline.load_documents([file_path])
        loaded_documents.append(file_path)
        
        return DocumentUploadResponse(
            filename=file.filename,
            status="success",
            message="Document uploaded and loaded successfully",
            document_id=str(len(loaded_documents))
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=Dict[str, str])
async def index_documents():
    """Index all loaded documents"""
    try:
        pipeline = get_rag_pipeline()
        pipeline.index_documents()
        global indexed_documents
        indexed_documents = loaded_documents.copy()
        
        return {"message": f"Successfully indexed {len(indexed_documents)} documents"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using RAG"""
    try:
        start_time = time.time()
        
        pipeline = get_rag_pipeline()
        
        # Convert retrieval methods to pipeline format
        retrieval_methods = [method.value for method in request.retrieval_methods]
        
        # Query the pipeline
        result = pipeline.query(
            request.query,
            use_adaptive_retrieval=request.use_adaptive_retrieval
        )
        
        # Get context and generate response
        context = result.get('context', '')
        print(f"API: Context length: {len(context)}")
        print(f"API: Context preview: {context[:200]}...")
        print(f"API: Context is empty: {not context}")
        print(f"API: Context is whitespace only: {context.strip() == '' if context else True}")
        
        # Use the appropriate LLM provider
        llm = LLMIntegration(request.llm_provider.value)
        answer = llm.generate_response(
            request.query,
            context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        processing_time = time.time() - start_time
        
        # Get metrics and logs
        metrics = {
            'context_length': len(context),
            'retrieval_methods_used': retrieval_methods,
            'processing_time': processing_time
        }
        
        retrieval_logs = pipeline.get_explainable_logs(result) if hasattr(pipeline, 'get_explainable_logs') else {}
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            context=context,
            retrieval_methods_used=retrieval_methods,
            metrics=metrics,
            retrieval_logs=retrieval_logs,
            processing_time=processing_time
        )
    
    except Exception as e:
        print(f"API: Error in query_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare", response_model=RAGComparisonResponse)
async def compare_rag_methods(request: RAGComparisonRequest):
    """Compare different RAG methods"""
    try:
        start_time = time.time()
        results = {}
        
        pipeline = get_rag_pipeline()
        
        for method in request.retrieval_methods:
            method_start_time = time.time()
            
            # Query with specific method
            result = pipeline.query(
                request.query,
                use_adaptive_retrieval=False  # Disable adaptive retrieval for comparison
            )
            
            context = result.get('context', '')
            
            # Generate response
            llm = LLMIntegration(request.llm_provider.value)
            answer = llm.generate_response(
                request.query,
                context,
                max_tokens=1000,
                temperature=0.7
            )
            
            method_processing_time = time.time() - method_start_time
            
            results[method.value] = QueryResponse(
                query=request.query,
                answer=answer,
                context=context,
                retrieval_methods_used=[method.value],
                metrics={
                    'context_length': len(context),
                    'processing_time': method_processing_time
                },
                retrieval_logs=pipeline.get_explainable_logs(result) if hasattr(pipeline, 'get_explainable_logs') else {},
                processing_time=method_processing_time
            )
        
        total_processing_time = time.time() - start_time
        
        # Calculate comparison metrics
        comparison_metrics = {
            'total_processing_time': total_processing_time,
            'method_comparison': {
                method.value: {
                    'processing_time': results[method.value].processing_time,
                    'context_length': results[method.value].metrics['context_length']
                }
                for method in request.retrieval_methods
            }
        }
        
        return RAGComparisonResponse(
            query=request.query,
            results=results,
            comparison_metrics=comparison_metrics,
            processing_time=total_processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[str])
async def list_documents():
    """List all loaded documents"""
    return loaded_documents

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document"""
    try:
        if 0 <= document_id < len(loaded_documents):
            document_path = loaded_documents.pop(document_id)
            
            # Remove from pipeline if possible
            pipeline = get_rag_pipeline()
            if hasattr(pipeline, 'remove_document'):
                pipeline.remove_document(document_path)
            
            # Delete file
            if os.path.exists(document_path):
                os.remove(document_path)
            
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True
    )
