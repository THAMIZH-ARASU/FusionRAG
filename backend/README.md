# FusionRAG Webhook API

A FastAPI-based webhook API that integrates with FusionRAG and Groq LLM for document question-answering. This API is designed to meet the hackathon requirements for the `/hackrx/run` endpoint.

## Features

- ✅ **Hackathon Compliant**: Implements the exact `/hackrx/run` endpoint specification
- 🔐 **Secure Authentication**: Bearer token authentication
- 📄 **Document Processing**: Downloads and processes PDF documents from URLs
- 🤖 **Groq LLM Integration**: Uses Groq API for fast, accurate responses
- 🔍 **Advanced RAG**: Leverages FusionRAG's hybrid retrieval system
- ⚡ **Async Processing**: High-performance async/await implementation
- 🧪 **Built-in Testing**: Comprehensive test suite included

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file based on `env_example.txt`:

```bash
cp env_example.txt .env
```

Edit `.env` with your actual values:

```env
# Groq API Configuration
GROQ_API_KEY=your_actual_groq_api_key_here

# Webhook API Security
WEBHOOK_API_KEY=your_webhook_secret_key_here

# Server Configuration
PORT=8000
HOST=0.0.0.0
```

### 3. Start the Server

```bash
python start_server.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### Main Webhook Endpoint

**POST** `/hackrx/run`

**Headers:**
```
Authorization: Bearer <your_webhook_api_key>
Content-Type: application/json
Accept: application/json
```

**Request Body:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment...",
        "There is a waiting period of thirty-six months..."
    ]
}
```

### Health Check

**GET** `/health`

Returns server health status.

### API Documentation

**GET** `/docs`

Interactive API documentation (Swagger UI).

## Testing

### Run the Test Suite

```bash
python test_webhook.py
```

This will test:
- ✅ Health check endpoint
- ✅ Invalid API key rejection
- ✅ Main webhook functionality

### Manual Testing with curl

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer your_webhook_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    ]
  }'
```

## Deployment

### Local Development

```bash
python start_server.py
```

### Production Deployment

1. **Set environment variables** in your deployment platform
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Start the server**: `python main.py`

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## Configuration

### RAG Pipeline Settings

The RAG pipeline is configured in `main.py`:

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

### Groq Model Settings

Default model: `llama3-8b-8192`

You can change this by setting the `GROQ_MODEL` environment variable.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Webhook API   │───▶│   FusionRAG     │───▶│   Groq API      │
│   (FastAPI)     │    │   Pipeline      │    │   (LLM)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Hybrid        │
│   Downloader    │    │   Retrieval     │
└─────────────────┘    └─────────────────┘
```

## Troubleshooting

### Common Issues

1. **"GROQ_API_KEY environment variable is required"**
   - Set your Groq API key in the `.env` file

2. **"Invalid API key"**
   - Make sure you're using the correct `WEBHOOK_API_KEY` in your requests

3. **Document download fails**
   - Check if the URL is accessible
   - Verify the document format is supported (PDF)

4. **Slow response times**
   - The first request may be slower due to model loading
   - Subsequent requests should be faster

### Logs

The API provides detailed logging. Check the console output for:
- Document download status
- RAG pipeline processing
- Groq API calls
- Error details

## Hackathon Submission

For hackathon submission, use your deployed webhook URL:

```
https://your-deployed-app.com/hackrx/run
```

Make sure:
- ✅ HTTPS is enabled
- ✅ API responds within 30 seconds
- ✅ All test cases pass
- ✅ Authentication is working

## Tech Stack

- **FastAPI**: Modern, fast web framework
- **FusionRAG**: Advanced RAG pipeline with hybrid retrieval
- **Groq**: High-performance LLM API
- **aiohttp**: Async HTTP client/server
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## License

This project is part of the FusionRAG system.
