"""Test script to list available Gemini models."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No API key found in .env file")
    print("Please add: GEMINI_API_KEY=your_key_here")
    exit()

if not api_key.startswith("AIza"):
    print("❌ Invalid API key format. Key should start with 'AIza'")
    exit()

try:
    client = genai.Client(api_key=api_key)
    models = client.models.list()
    
    print("✅ Available Gemini Models:\n")
    for model in models:
        # Check if it's a Gemini model
        if "gemini" in model.name.lower():
            print(f"  - {model.name}")
            # Also check supported methods if available
            if hasattr(model, 'supported_methods'):
                print(f"    Methods: {model.supported_methods}")
    
except Exception as e:
    print(f"❌ Error: {e}")