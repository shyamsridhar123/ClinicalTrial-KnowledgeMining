#!/usr/bin/env python3
"""Simple test to verify GPT-5-mini is working."""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

def test_simple():
    """Test basic GPT-5-mini call without images."""
    print("🔍 Testing GPT-5-mini basic call...")
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    )
    
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT_NAME", "gpt-5-mini"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, GPT-5-mini is working!' and nothing else."}
            ],
            max_completion_tokens=50
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        print(f"✅ Response: {content}")
        print(f"   Tokens: {tokens}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_simple()
    exit(0 if success else 1)
