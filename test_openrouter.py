"""Test OpenRouter API connection with available free models."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ No OpenRouter API key found in .env file")
    print("Please add: OPENROUTER_API_KEY=sk-or-v1-...")
    exit()

# List of confirmed working free models
FREE_MODELS = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free"
]

print("⏳ Testing OpenRouter API with available free models...")

def test_model(model_name):
    """Test a specific model."""
    print(f"\n📌 Testing: {model_name}")
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
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "Say hello in 5 words"}
                ],
                "max_tokens": 20,
                "temperature": 0.5
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS with {model_name}!")
            print(f"Response: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ Failed: {response.status_code} - {response.text[:100]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test each model
working_model = None
for model in FREE_MODELS:
    if test_model(model):
        working_model = model
        break

if working_model:
    print(f"\n✅ Working model found: {working_model}")
    print("\nUpdate your llm_handler.py with this model.")
else:
    print("\n❌ No working free models found. Please check your API key.")
    print("Get a new key from: https://openrouter.ai/keys")