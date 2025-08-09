#!/usr/bin/env python3
"""
Startup script for FusionRAG application
This script starts both the FastAPI backend and Flask UI
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'flask',
        'requests',
        'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path("fast_api_backend/.env")
    if not env_file.exists():
        print("Creating .env file template...")
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
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("Created .env file template. Please edit it with your actual API keys.")
        return False
    
    return True

def start_fastapi_backend():
    """Start the FastAPI backend"""
    print("Starting FastAPI backend...")
    backend_dir = Path("fast_api_backend")
    if not backend_dir.exists():
        print("Error: fast_api_backend directory not found!")
        return False
    
    os.chdir(backend_dir)
    try:
        # Start the FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\nFastAPI backend stopped.")
    except Exception as e:
        print(f"Error starting FastAPI backend: {e}")
        return False
    
    return True

def start_flask_ui():
    """Start the Flask UI"""
    print("Starting Flask UI...")
    ui_dir = Path("flask_ui")
    if not ui_dir.exists():
        print("Error: flask_ui directory not found!")
        return False
    
    os.chdir(ui_dir)
    try:
        # Start the Flask server
        subprocess.run([
            sys.executable, "app.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\nFlask UI stopped.")
    except Exception as e:
        print(f"Error starting Flask UI: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 FusionRAG Application Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        print("\n⚠️  Please configure your .env file with actual API keys before continuing.")
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Get current directory
    current_dir = Path.cwd()
    
    # Start FastAPI backend in a separate thread
    backend_thread = threading.Thread(target=start_fastapi_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    print("Waiting for FastAPI backend to start...")
    time.sleep(3)
    
    # Start Flask UI
    try:
        start_flask_ui()
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
