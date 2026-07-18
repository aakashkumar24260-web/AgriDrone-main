"""List all available models on OpenRouter."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ No OpenRouter API key found")
    exit()

try:
    # Get list of all models
    response = requests.get(
        url="https://openrouter.ai/api/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        timeout=30
    )
    
    if response.status_code == 200:
        models = response.json()
        
        print("🔍 Available Models on OpenRouter:\n")
        
        # Filter for free models
        free_models = []
        for model in models.get("data", []):
            model_id = model.get("id", "")
            # Check if it's a free model (usually has ":free" in the ID)
            if ":free" in model_id:
                free_models.append(model_id)
                print(f"  ✅ {model_id}")
            elif "gemini" in model_id.lower() or "llama" in model_id.lower() or "phi" in model_id.lower():
                print(f"  📌 {model_id} (check if free)")
        
        print(f"\n📊 Found {len(free_models)} free models")
        
        if free_models:
            print("\n✅ Try these models in your code:")
            for model in free_models[:5]:
                print(f"  - {model}")
        else:
            print("\n⚠️ No ':free' models found. Trying to find any working models...")
            # Try alternative free models
            test_models = [
                "google/gemini-2.0-flash-lite-preview-02-05",
                "google/gemini-flash-1.5",
                "meta-llama/llama-3.2-3b-instruct"
            ]
            for model in test_models:
                print(f"  Try: {model}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")