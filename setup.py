#!/usr/bin/env python3
"""
Setup script for FusionRAG application
This script helps users install dependencies and configure the environment
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    
    requirements_files = [
        "requirements.txt",
        "fast_api_backend/requirements.txt",
        "flask_ui/requirements.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"Installing from {req_file}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
                print(f"✅ Successfully installed dependencies from {req_file}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies from {req_file}: {e}")
                return False
        else:
            print(f"⚠️  {req_file} not found, skipping...")
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path("fast_api_backend/.env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("\n🔧 Creating .env file...")
    env_content = """# FusionRAG Environment Variables
# Please replace with your actual API keys

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
        env_file.parent.mkdir(parents=True, exist_ok=True)
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("⚠️  Please edit fast_api_backend/.env with your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating necessary directories...")
    
    directories = [
        "fast_api_backend/uploads",
        "flask_ui/uploads",
        "documents"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 FusionRAG Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        print("❌ Failed to create .env file")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit fast_api_backend/.env with your actual API keys")
    print("2. Run the application: python start_app.py")
    print("3. Access the web interface at http://localhost:5000")
    print("4. Access the API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main()
