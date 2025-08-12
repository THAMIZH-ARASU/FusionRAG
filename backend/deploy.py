#!/usr/bin/env python3
"""
Deployment script for FusionRAG Webhook API
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'aiohttp', 'pydantic', 
        'python-dotenv', 'sentence-transformers', 'faiss-cpu'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_environment():
    """Check environment variables"""
    required_vars = ['GROQ_API_KEY', 'WEBHOOK_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables or create a .env file")
        return False
    
    print("✅ Environment variables are set")
    return True

def create_env_file():
    """Create .env file from template"""
    env_example = Path("env_example.txt")
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        return True
    
    if env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your actual API keys")
        return True
    else:
        print("❌ env_example.txt not found")
        return False

def test_server():
    """Test if server can start"""
    try:
        # Try to import the app
        from main import app
        print("✅ Server can be imported successfully")
        return True
    except Exception as e:
        print(f"❌ Server import failed: {str(e)}")
        return False

def show_deployment_info():
    """Show deployment information"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INFORMATION")
    print("="*60)
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if we're in the backend directory
    if current_dir.name != "backend":
        print("⚠️  Make sure you're in the backend directory")
    
    # Show environment info
    print(f"🐍 Python executable: {sys.executable}")
    print(f"💻 Platform: {platform.platform()}")
    
    # Show available endpoints
    print("\n🔗 Available endpoints:")
    print("   - POST /hackrx/run (main webhook)")
    print("   - GET  /health (health check)")
    print("   - GET  /docs (API documentation)")
    
    # Show deployment commands
    print("\n📋 Deployment commands:")
    print("   Local: python start_server.py")
    print("   Direct: python main.py")
    print("   Test: python test_webhook.py")
    
    print("\n🌐 For hackathon submission:")
    print("   Webhook URL: https://your-domain.com/hackrx/run")
    print("   Make sure HTTPS is enabled!")

def main():
    """Main deployment check"""
    print("🔍 FusionRAG Webhook API Deployment Check")
    print("="*50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment Setup", create_env_file),
        ("Server Import", test_server),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! Your API is ready for deployment.")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1
    
    show_deployment_info()
    
    # Ask if user wants to start the server
    print("\n" + "="*50)
    response = input("🚀 Do you want to start the server now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting server...")
        try:
            subprocess.run([sys.executable, "start_server.py"], check=True)
        except KeyboardInterrupt:
            print("\n👋 Server stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to start server: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
