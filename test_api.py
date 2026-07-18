# test_api.py - Updated for the new google.genai SDK
import os
from dotenv import load_dotenv
# Import the new SDK
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"API Key: {api_key[:10]}...{api_key[-5:] if api_key else 'None'}")

if api_key:
    try:
        # 1. Initialize the client with your API key
        client = genai.Client(api_key=api_key)

        # 2. Use the new 'models.generate_content' method
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", # or "gemini-1.5-flash"
            contents="Say hello"
        )

        print("✅ API Key and new SDK work!")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No API key found in .env file")