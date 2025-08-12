#!/usr/bin/env python3
"""
Setup script for FusionRAG Webhook API
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def setup_environment():
    """Set up environment variables"""
    print("🔧 Setting up environment...")
    
    # Check if .env exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists")
        return True
    
    # Create .env from template
    env_example = Path("env_example.txt")
    if env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
        print("⚠️  IMPORTANT: Please edit .env file with your actual API keys:")
        print("   - GROQ_API_KEY: Get from https://console.groq.com/")
        print("   - WEBHOOK_API_KEY: Set to any secure secret key")
        return True
    else:
        print("❌ env_example.txt not found")
        return False

def install_dependencies():
    """Install Python dependencies"""
    return run_command(
        "pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def test_installation():
    """Test if everything is working"""
    print("🧪 Testing installation...")
    
    # Test imports
    try:
        import fastapi
        import uvicorn
        import aiohttp
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test server import
    try:
        from main import app
        print("✅ Server can be imported successfully")
    except Exception as e:
        print(f"❌ Server import failed: {e}")
        return False
    
    return True

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys:")
    print("   - GROQ_API_KEY: Get from https://console.groq.com/")
    print("   - WEBHOOK_API_KEY: Set to any secure secret key")
    
    print("\n2. Start the server:")
    print("   python start_server.py")
    
    print("\n3. Test the API:")
    print("   python test_webhook.py")
    
    print("\n4. View API documentation:")
    print("   http://localhost:8000/docs")
    
    print("\n5. For hackathon submission:")
    print("   Deploy to your preferred platform (Heroku, Railway, etc.)")
    print("   Use the webhook URL: https://your-domain.com/hackrx/run")
    
    print("\n🔗 Useful commands:")
    print("   Start server: python start_server.py")
    print("   Run tests: python test_webhook.py")
    print("   Deploy check: python deploy.py")
    print("   Docker build: docker build -t fusionrag-webhook .")
    print("   Docker run: docker run -p 8000:8000 fusionrag-webhook")

def main():
    """Main setup function"""
    print("🚀 FusionRAG Webhook API Setup")
    print("="*50)
    
    # Check if we're in the right directory
    if Path.cwd().name != "backend":
        print("❌ Please run this script from the backend directory")
        return 1
    
    steps = [
        ("Environment Setup", setup_environment),
        ("Dependency Installation", install_dependencies),
        ("Installation Test", test_installation),
    ]
    
    all_success = True
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_func():
            all_success = False
            break
    
    if all_success:
        show_next_steps()
        return 0
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
