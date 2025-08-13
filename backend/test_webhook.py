#!/usr/bin/env python3
"""
Test script for FusionRAG Webhook API
"""

import asyncio
import aiohttp
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)

# Configuration
API_BASE_URL = "https://fusionrag.onrender.com"
API_KEY = os.getenv("WEBHOOK_API_KEY", "your_webhook_api_key")

# Test data based on hackathon requirements
TEST_REQUEST = {
    "documents": "your-document-location",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

async def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check endpoint...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status}")
                return False

async def test_webhook_endpoint():
    """Test the main webhook endpoint"""
    print("\n🔗 Testing webhook endpoint...")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{API_BASE_URL}/hackrx/run",
                headers=headers,
                json=TEST_REQUEST,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            ) as response:
                
                print(f"📊 Response Status: {response.status}")
                print(f"📋 Response Headers: {dict(response.headers)}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Webhook test passed!")
                    print(f"📝 Number of answers: {len(data.get('answers', []))}")
                    
                    # Print first few answers as preview
                    for i, answer in enumerate(data.get('answers', [])[:3]):
                        print(f"\nAnswer {i+1}: {answer[:200]}...")
                    
                    if len(data.get('answers', [])) > 3:
                        print(f"\n... and {len(data.get('answers', [])) - 3} more answers")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Webhook test failed: {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error testing webhook: {str(e)}")
            return False

async def test_invalid_api_key():
    """Test with invalid API key"""
    print("\n🔐 Testing invalid API key...")
    
    headers = {
        "Authorization": "Bearer invalid-key",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_BASE_URL}/hackrx/run",
            headers=headers,
            json=TEST_REQUEST
        ) as response:
            if response.status == 401:
                print("✅ Invalid API key correctly rejected")
                return True
            else:
                print(f"❌ Invalid API key not rejected: {response.status}")
                return False

async def main():
    """Run all tests"""
    print("🧪 Starting FusionRAG Webhook API Tests")
    print("=" * 50)
    
    # Test health check
    health_ok = await test_health_check()
    
    if not health_ok:
        print("❌ Health check failed. Make sure the server is running.")
        return
    
    # Test invalid API key
    await test_invalid_api_key()
    
    # Test main webhook endpoint
    webhook_ok = await test_webhook_endpoint()
    
    print("\n" + "=" * 50)
    if webhook_ok:
        print("🎉 All tests passed! Your webhook API is working correctly.")
        print(f"\n📋 Webhook URL for submission: {API_BASE_URL}/hackrx/run")
    else:
        print("❌ Some tests failed. Please check the server logs.")

if __name__ == "__main__":
    asyncio.run(main())
