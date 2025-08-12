# FusionRAG Webhook API - Hackathon Submission

## 🎯 What We Built

A complete webhook API that meets all hackathon requirements:

### ✅ Hackathon Compliance
- **Endpoint**: `POST /hackrx/run` ✅
- **Authentication**: Bearer token ✅
- **Request Format**: Exact JSON structure ✅
- **Response Format**: Exact JSON structure ✅
- **HTTPS Ready**: Production deployment ready ✅

### 🏗️ Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Hackathon     │───▶│   FusionRAG     │───▶│   Groq API      │
│   Platform      │    │   Pipeline      │    │   (Llama3)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Hybrid        │
│   Webhook       │    │   Retrieval     │
└─────────────────┘    └─────────────────┘
```

## 📁 Files Created

### Core API Files
- `main.py` - Main FastAPI application with webhook endpoint
- `requirements.txt` - All necessary dependencies
- `start_server.py` - Easy startup script with environment checks

### Configuration & Setup
- `env_example.txt` - Environment variables template
- `setup.py` - Automated setup script
- `deploy.py` - Deployment verification script

### Testing & Documentation
- `test_webhook.py` - Comprehensive test suite
- `README.md` - Complete documentation
- `SUMMARY.md` - This file

### Deployment
- `Dockerfile` - Container deployment
- `docker-compose.yml` - Local development with Docker

## 🚀 Quick Start

### 1. Setup (One-time)
```bash
cd backend
python setup.py
```

### 2. Configure API Keys
Edit `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
WEBHOOK_API_KEY=your_webhook_secret_key_here
```

### 3. Start Server
```bash
python start_server.py
```

### 4. Test
```bash
python test_webhook.py
```

## 🔗 API Endpoints

### Main Webhook
```
POST /hackrx/run
Authorization: Bearer <api_key>
Content-Type: application/json

{
    "documents": "https://example.com/document.pdf",
    "questions": ["Question 1", "Question 2"]
}
```

### Health Check
```
GET /health
```

### API Docs
```
GET /docs
```

## 🧪 Testing

The test suite validates:
- ✅ Health check endpoint
- ✅ Authentication (invalid key rejection)
- ✅ Main webhook functionality
- ✅ Response format compliance

## 🌐 Deployment

### Local Development
```bash
python start_server.py
```

### Docker
```bash
docker build -t fusionrag-webhook .
docker run -p 8000:8000 fusionrag-webhook
```

### Production Platforms
- **Heroku**: Use `Procfile` and environment variables
- **Railway**: Direct deployment with environment variables
- **Render**: Web service deployment
- **DigitalOcean App Platform**: Container deployment

## 📋 Hackathon Submission

### Webhook URL
```
https://your-deployed-domain.com/hackrx/run
```

### Description
```
FastAPI + FusionRAG + Groq (Llama3) + Hybrid Retrieval
```

### Features
- 🔍 **Hybrid Retrieval**: BM25 + Vector Search
- 🤖 **Groq LLM**: Fast Llama3-8b responses
- 📄 **Document Processing**: PDF download & parsing
- ⚡ **Async Processing**: High-performance async/await
- 🔐 **Secure**: Bearer token authentication
- 🧪 **Tested**: Comprehensive test suite

## 🎯 Key Advantages

1. **Speed**: Groq API provides ultra-fast LLM responses
2. **Accuracy**: FusionRAG's hybrid retrieval ensures relevant context
3. **Reliability**: Comprehensive error handling and logging
4. **Scalability**: Async processing handles multiple requests
5. **Compliance**: Exact hackathon specification implementation

## 🔧 Customization

### RAG Configuration
Edit `RAG_CONFIG` in `main.py`:
```python
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
```

### Groq Model
Change model in `GroqLLMIntegration`:
```python
self.model = "llama3-8b-8192"  # or "mixtral-8x7b-32768"
```

## 🚨 Important Notes

1. **API Keys**: Never commit `.env` file to version control
2. **HTTPS**: Required for production deployment
3. **Rate Limits**: Groq has rate limits, monitor usage
4. **Response Time**: First request may be slower due to model loading
5. **Memory**: Ensure sufficient RAM for embedding models

## 🎉 Ready for Submission!

Your webhook API is now ready for hackathon submission. The implementation:

- ✅ Meets all technical requirements
- ✅ Uses only Groq API (no ChatGPT)
- ✅ Integrates with your FusionRAG system
- ✅ Includes comprehensive testing
- ✅ Ready for production deployment

**Next Step**: Deploy to your preferred platform and submit the webhook URL!
