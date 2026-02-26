from __future__ import annotations
import asyncio
from ella.config import Settings
from ella.llm.gemini_client import get_gemini_client

async def test():
    settings = Settings()
    client = get_gemini_client(settings.google_api_key)
    try:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Say exactly: 'Gemini integration successful!'"}],
            model="gemini-2.5-pro"
        )
        print(f"✅ LLM Response: {response.text}")
    except Exception as e:
        print(f"❌ LLM error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
