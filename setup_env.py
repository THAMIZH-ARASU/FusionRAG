#!/usr/bin/env python3
"""
Setup script for FusionRAG environment configuration
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path("fast_api_backend/.env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("🔧 Creating .env file...")
    
    env_content = """# FusionRAG Environment Variables
# Please replace with your actual API keys

# LLM Provider API Keys
GROQ_API_KEY=your_GROQ_API_KEY_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Server settings
HOST=0.0.0.0
PORT=8000

# File upload settings
UPLOAD_DIR=uploads
MAX_FILE_SIZE=52428800

# RAG settings
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
DEFAULT_MAX_CONTEXT_LENGTH=4000
"""
    
    try:
        # Ensure the directory exists
        env_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env file template")
        print("⚠️  Please edit fast_api_backend/.env with your actual API keys")
        print("\n📝 Required API keys:")
        print("   - GROQ_API_KEY: Get from https://console.groq.com/")
        print("   - GOOGLE_API_KEY: Get from https://makersuite.google.com/app/apikey")
        print("   - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FusionRAG Environment Setup")
    print("=" * 40)
    
    if create_env_file():
        print("\n🎉 Setup complete!")
        print("\nNext steps:")
        print("1. Edit fast_api_backend/.env with your actual API keys")
        print("2. Run: python start_app.py")
        print("3. Access the UI at: http://localhost:5000")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
