# FusionRAG Quick Start Guide

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Run the setup script
python setup.py
```

This will:
- Check Python version (3.8+ required)
- Install all required packages
- Create necessary directories
- Create .env file template

### 2. Configure API Keys

Edit `fast_api_backend/.env` and add your API keys:

```env
GROQ_API_KEY=your_GROQ_API_KEY_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the Application

```bash
# Start both FastAPI backend and Flask UI
python start_app.py
```

### 4. Access the Application

- **Web Interface**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs
- **API ReDoc**: http://localhost:8000/redoc

## 🌐 Using the Web Interface

### Upload Documents

1. Go to the "Upload Documents" tab
2. Click "Choose File" or drag and drop files
3. Supported formats: PDF, DOCX, TXT, MD
4. Click "Upload Document"
5. Click "Index Documents" to make them searchable

### Query Documents

1. Go to the "Query Documents" tab
2. Enter your question in the query box
3. Select retrieval methods (BM25, Vector DB, Hybrid)
4. Choose LLM provider (OpenAI, Grok, Google)
5. Adjust parameters (max tokens, temperature)
6. Click "Query Documents"

### Compare Methods

1. Go to the "Compare Methods" tab
2. Enter your question
3. Select methods to compare
4. Choose LLM provider
5. Click "Compare Methods"
6. View side-by-side results and metrics

### System Status

1. Go to the "System Status" tab
2. View system health, document counts, and available methods

## 🔧 API Usage

### Upload Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "retrieval_methods": ["hybrid"],
    "llm_provider": "openai",
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

### Compare Methods

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "retrieval_methods": ["bm25", "vector_db", "hybrid"],
    "llm_provider": "openai"
  }'
```

## 🎯 Features

### Document Processing
- ✅ PDF document support
- ✅ Word document support
- ✅ Text file support
- ✅ Markdown support
- ✅ Automatic chunking and indexing

### Retrieval Methods
- ✅ BM25 keyword-based retrieval
- ✅ Vector database semantic search
- ✅ Knowledge graph retrieval
- ✅ Hybrid retrieval (combines multiple methods)

### LLM Integration
- ✅ OpenAI GPT models
- ✅ Grok AI models
- ✅ Google AI models
- ✅ Configurable parameters

### Web Interface
- ✅ Modern, responsive design
- ✅ Real-time document upload
- ✅ Interactive query interface
- ✅ Method comparison tool
- ✅ System status monitoring
- ✅ Metrics visualization

### API Features
- ✅ RESTful API design
- ✅ OpenAPI documentation
- ✅ CORS support
- ✅ Error handling
- ✅ Health checks

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   lsof -i :5000
   
   # Kill the process or change ports in .env
   ```

2. **API keys not working**
   - Check that your API keys are correctly set in `fast_api_backend/.env`
   - Verify the keys are valid and have sufficient credits

3. **Documents not uploading**
   - Check file size (max 50MB)
   - Ensure file format is supported
   - Check upload directory permissions

4. **Slow performance**
   - Reduce chunk size in configuration
   - Use fewer retrieval methods
   - Check system resources

### Getting Help

- Check the logs in the terminal where you started the application
- Visit the API documentation at http://localhost:8000/docs
- Check the system status at http://localhost:5000 (Status tab)

## 📊 Performance Tips

1. **For large documents**: Use smaller chunk sizes (500-800)
2. **For faster queries**: Use single retrieval method instead of hybrid
3. **For better accuracy**: Use adaptive retrieval with multiple iterations
4. **For cost optimization**: Use local models when possible

## 🎉 Next Steps

1. **Upload your documents** and start querying
2. **Experiment with different retrieval methods** to find what works best
3. **Try the comparison feature** to understand method differences
4. **Check the API documentation** for advanced usage
5. **Customize the configuration** for your specific needs

Happy RAG-ing! 🚀
