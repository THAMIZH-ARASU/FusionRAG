#!/usr/bin/env python3
"""
Test script to verify FusionRAG setup
"""

import os
import sys
import requests
import time
from pathlib import Path

def test_fastapi_backend():
    """Test FastAPI backend"""
    print("Testing FastAPI backend...")
    
    try:
        # Test if backend is running
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI backend is running")
            return True
        else:
            print(f"❌ FastAPI backend returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI backend is not running")
        return False
    except Exception as e:
        print(f"❌ Error testing FastAPI backend: {e}")
        return False

def test_flask_ui():
    """Test Flask UI"""
    print("Testing Flask UI...")
    
    try:
        # Test if UI is running
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ Flask UI is running")
            return True
        else:
            print(f"❌ Flask UI returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Flask UI is not running")
        return False
    except Exception as e:
        print(f"❌ Error testing Flask UI: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("Testing API endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data['status']}")
            print(f"   - Loaded documents: {health_data['loaded_documents']}")
            print(f"   - Indexed documents: {health_data['indexed_documents']}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing API endpoints: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 FusionRAG Setup Test")
    print("=" * 30)
    
    # Test FastAPI backend
    backend_ok = test_fastapi_backend()
    
    # Test Flask UI
    ui_ok = test_flask_ui()
    
    # Test API endpoints
    api_ok = test_api_endpoints()
    
    print("\n" + "=" * 30)
    if backend_ok and ui_ok and api_ok:
        print("🎉 All tests passed! FusionRAG is running correctly.")
        print("\nYou can now:")
        print("- Access the web interface at: http://localhost:5000")
        print("- Access the API documentation at: http://localhost:8000/docs")
        print("- Upload documents and start querying!")
    else:
        print("❌ Some tests failed. Please check the setup.")
        if not backend_ok:
            print("- Make sure the FastAPI backend is running on port 8000")
        if not ui_ok:
            print("- Make sure the Flask UI is running on port 5000")
        if not api_ok:
            print("- Check if the API endpoints are working correctly")

if __name__ == "__main__":
    main()
