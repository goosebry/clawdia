#!/usr/bin/env python3
"""Test Gemini integration for Ella AI."""

import asyncio
import os
from ella.config import Settings
from ella.llm.gemini_client import get_gemini_client

async def test_gemini():
    """Test Gemini API integration."""
    print("🧪 Testing Gemini Integration")
    print("=" * 40)
    
    # Load settings
    settings = Settings()
    
    if not settings.google_api_key or settings.google_api_key == "your_gemini_api_key_here":
        print("❌ GOOGLE_API_KEY not set in .env file")
        print("📋 To fix this:")
        print("   1. Go to: https://aistudio.google.com/app/apikey")
        print("   2. Create a free API key")
        print("   3. Add it to your .env file:")
        print("      GOOGLE_API_KEY=your_actual_key_here")
        return False
    
    try:
        # Test Gemini client
        print("🔧 Creating Gemini client...")
        client = get_gemini_client(settings.google_api_key, settings.gemini_model)
        
        print("💬 Testing chat completion...")
        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Say 'Hello from Gemini!' and tell me what model you are."}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"✅ Gemini Response: {response}")
        print("🎉 Gemini integration working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        print("🔧 Make sure your API key is valid and has credits")
        return False

if __name__ == "__main__":
    asyncio.run(test_gemini())