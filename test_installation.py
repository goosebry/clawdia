#!/usr/bin/env python3
"""
Simple test to verify Ella AI installation is working
"""

def test_basic_imports():
    """Test that we can import essential modules"""
    print("🧪 Testing basic imports...")
    
    try:
        import ella
        print("✅ ella package imported")
    except ImportError as e:
        print(f"❌ Failed to import ella: {e}")
        return False
    
    try:
        from ella.config import Settings
        print("✅ Settings imported")
    except ImportError as e:
        print(f"❌ Failed to import Settings: {e}")
        return False
    
    try:
        import redis
        import qdrant_client
        import aiomysql
        print("✅ Database clients imported")
    except ImportError as e:
        print(f"❌ Failed to import database clients: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from ella.config import Settings
        settings = Settings()
        print("✅ Configuration loaded")
        print(f"   - Telegram bot token: {'*' * 20}...{settings.telegram_bot_token[-4:]}")
        print(f"   - Gemini model: {settings.gemini_model}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_docker_services():
    """Test Docker service connections"""
    print("\n🧪 Testing Docker service connections...")
    
    # Test Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    
    # Test Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print("✅ Qdrant connection successful")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Ella AI Installation Test")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_docker_services,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your Ella AI installation is ready!")
        print("\n📋 Next steps:")
        print("   1. Start the AI agent: python ella/main.py")
        print("   2. Chat with Ella via Telegram using your bot token")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()