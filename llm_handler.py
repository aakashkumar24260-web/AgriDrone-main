"""OpenRouter API handler - Free alternative to OpenAI."""
import os
import requests
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Use the working model from the test
MODEL_NAME = "openai/gpt-5.6-sol"

def initialize_openai():
    """Check if API key is configured."""
    return OPENROUTER_API_KEY is not None and OPENROUTER_API_KEY.startswith("sk-or-v1-")

def initialize_gemini():
    """Alias for initialize_openai for backward compatibility."""
    return initialize_openai()

def generate_report(metrics: Dict, field_name: str, crop_type: str) -> str:
    """Generate agronomist report using OpenRouter."""
    
    if not OPENROUTER_API_KEY:
        return "⚠️ OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file."
    
    healthy_count = metrics.get('healthy', 0)
    early_count = metrics.get('early', 0)
    severe_count = metrics.get('severe', 0)
    total = metrics.get('total', 1)
    
    healthy_pct = (healthy_count / total) * 100
    early_pct = (early_count / total) * 100
    severe_pct = (severe_count / total) * 100
    
    prompt = f"""
You are an agricultural AI assistant. Generate a concise field health report for a farmer.

Field: {field_name} | Crop: {crop_type} | Grid: 25x25

Scan Results:
  Healthy cells  : {healthy_count} ({healthy_pct:.1f}%)
  Early disease  : {early_count}   ({early_pct:.1f}%)
  Severe disease : {severe_count}  ({severe_pct:.1f}%)

Write exactly 4 paragraphs:
  1. Overall field health summary
  2. Disease risk and likely spread pattern
  3. Specific treatment recommendations
  4. Next monitoring schedule

Use plain language. No bullet points. No markdown.
"""
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "AgriDrone"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an agricultural AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"⚠️ API Error ({response.status_code}): {response.text[:200]}"
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def get_spray_advice(metrics: Dict, crop_type: str) -> str:
    """Get spray advice using OpenRouter."""
    
    if not OPENROUTER_API_KEY:
        return "⚠️ OpenRouter API key not found."
    
    affected = metrics.get('early', 0) + metrics.get('severe', 0)
    total = metrics.get('total', 1)
    affected_pct = (affected / total) * 100
    
    prompt = f"""
You are an agricultural AI assistant specializing in crop protection.

Crop: {crop_type}
Affected area: {affected_pct:.1f}% of the field
Early disease: {metrics.get('early', 0)} cells
Severe disease: {metrics.get('severe', 0)} cells

Provide concise spray recommendation: fungicide type, application method, dosage, safety.
"""
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "AgriDrone"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are an agricultural crop protection expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 300
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"⚠️ API Error: {response.text[:200]}"
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"