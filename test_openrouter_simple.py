"""Test OpenRouter with automatic model routing."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ No OpenRouter API key found")
    exit()

print("⏳ Testing OpenRouter API with automatic routing...")

try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AgriDrone"
        },
        json={
            # Use auto-routing instead of specific model
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Say hello in 5 words"}
            ],
            "max_tokens": 20,
            "temperature": 0.5,
            "route": "fallback"  # This enables fallback to other models
        },
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ OpenRouter API works!")
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Model used: {result.get('model', 'Unknown')}")
        print("\n✅ You can now run: streamlit run app.py")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Details: {response.text}")
        
        # Try alternative with paid model as fallback
        print("\n🔄 Trying with a different approach...")
        try:
            response2 = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",  # Try this if available
                    "messages": [
                        {"role": "user", "content": "Say hello"}
                    ],
                    "max_tokens": 10
                },
                timeout=30
            )
            if response2.status_code == 200:
                print("✅ Working with GPT-3.5!")
            else:
                print(f"❌ Still not working: {response2.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")

except Exception as e:
    print(f"❌ Error: {e}")