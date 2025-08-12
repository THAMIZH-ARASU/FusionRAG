#!/usr/bin/env python3
"""
Startup script for FusionRAG Webhook API
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)

# Check required environment variables
required_vars = ['GROQ_API_KEY', 'WEBHOOK_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print("❌ Missing required environment variables:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\nPlease set these variables or create a .env file based on env_example.txt")
    sys.exit(1)

# Import and run the FastAPI app
from main import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting FusionRAG Webhook API on {host}:{port}")
    print(f"📝 API Documentation: http://{host}:{port}/docs")
    print(f"🔗 Webhook Endpoint: http://{host}:{port}/hackrx/run")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
