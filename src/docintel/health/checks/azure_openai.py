"""Azure OpenAI health check."""

import os
from pathlib import Path
from ..base import HealthCheck, HealthStatus


class AzureOpenAIHealthCheck(HealthCheck):
    """Check Azure OpenAI API connectivity."""
    
    name = "Azure OpenAI (GPT-4.1)"
    timeout_seconds = 10.0
    
    async def _perform_check(self):
        # Load from .env explicitly if not in environment
        from dotenv import load_dotenv
        load_dotenv()
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([endpoint, api_key, deployment]):
            missing = []
            if not endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
            if not api_key: missing.append("AZURE_OPENAI_API_KEY")
            if not deployment: missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
            
            return (
                HealthStatus.ERROR,
                f"Missing env vars: {', '.join(missing)}",
                {"missing": missing}
            )
        
        try:
            from openai import AsyncAzureOpenAI
            
            client = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-02-15-preview",
                timeout=self.timeout_seconds
            )
            
            # Minimal test request
            response = await client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            masked_endpoint = endpoint.split("//")[1].split(".")[0] if "//" in endpoint else endpoint
            return (
                HealthStatus.HEALTHY,
                f"Connected to {masked_endpoint}",
                {
                    "endpoint": masked_endpoint + ".openai.azure.com",
                    "deployment": deployment,
                    "model": response.model
                }
            )
        except Exception as e:
            error_msg = str(e)[:100]
            return (
                HealthStatus.ERROR,
                f"Cannot connect: {error_msg}",
                {"error": error_msg}
            )
