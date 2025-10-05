"""
LLM client for routing between GPT-4.1 and GPT-5-mini.
Handles parameter normalization across different model APIs.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import AzureOpenAI

from .model_router import ModelChoice
from docintel.config import AzureOpenAISettings

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any]


class LLMClient:
    """
    Unified client for Azure OpenAI models with automatic parameter normalization.
    
    Handles differences between GPT-4.1 and GPT-5-mini APIs:
    - Parameter naming: max_tokens vs max_completion_tokens
    - Temperature control: supported in GPT-4.1, fixed in GPT-5-mini
    - Multimodal input: both support images
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-08-01-preview",
        gpt41_deployment: Optional[str] = None,
        gpt5_deployment: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY env var)
            endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            api_version: API version to use
            gpt41_deployment: GPT-4.1 deployment name (defaults to AZURE_OPENAI_DEPLOYMENT_NAME)
            gpt5_deployment: GPT-5-mini deployment name (defaults to AZURE_OPENAI_GPT5_DEPLOYMENT_NAME)
        """
        # Load from AzureOpenAISettings if credentials not provided
        if not api_key or not endpoint:
            settings = AzureOpenAISettings()
            self.api_key = api_key or (settings.api_key.get_secret_value() if hasattr(settings.api_key, 'get_secret_value') else settings.api_key)
            self.endpoint = endpoint or str(settings.endpoint)
            self.api_version = api_version
            self.gpt41_deployment = gpt41_deployment or settings.deployment_name
            # Try to get GPT-5 deployment from env, fallback to "gpt-5-mini"
            self.gpt5_deployment = gpt5_deployment or os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT_NAME", "gpt-5-mini")
        else:
            self.api_key = api_key
            self.endpoint = endpoint
            self.api_version = api_version
            self.gpt41_deployment = gpt41_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
            self.gpt5_deployment = gpt5_deployment or os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT_NAME", "gpt-5-mini")
        
        if not self.api_key or not self.endpoint:
            raise ValueError(
                "Azure OpenAI credentials required. Set AZURE_OPENAI_API_KEY "
                "and AZURE_OPENAI_ENDPOINT environment variables."
            )
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )
        
        logger.info(f"LLM client initialized: GPT-4.1={self.gpt41_deployment}, GPT-5-mini={self.gpt5_deployment}")
    
    def query(
        self,
        messages: List[Dict[str, Any]],
        model: ModelChoice,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send query to the appropriate model with parameter normalization.
        
        Args:
            messages: Chat messages in OpenAI format
            model: Which model to use (GPT_4_1 or GPT_5_MINI)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (ignored for GPT-5-mini)
            response_format: Optional response format (e.g., {"type": "json_object"})
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with model output and metadata
        """
        # Select deployment name
        deployment = self._get_deployment_name(model)
        
        # Normalize parameters based on model
        params = self._normalize_parameters(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            **kwargs
        )
        
        logger.info(f"Calling {deployment} with {len(messages)} messages")
        logger.debug(f"Parameters: {params}")
        
        try:
            response = self.client.chat.completions.create(
                model=deployment,
                messages=messages,
                **params
            )
            
            # Extract response
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # DEBUG: Log empty responses
            if not content:
                logger.warning(f"Empty response from {deployment}!")
                logger.warning(f"Full response object: {response}")
                logger.warning(f"Choice: {choice}")
                logger.warning(f"Message: {choice.message}")
            
            result = LLMResponse(
                content=content,
                model=deployment,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=choice.finish_reason,
                metadata={
                    "model_choice": model.value,
                    "deployment_name": deployment,
                    "parameters": params
                }
            )
            
            logger.info(
                f"Response from {deployment}: {result.usage['total_tokens']} tokens "
                f"(finish: {result.finish_reason}), content_length={len(content)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling {deployment}: {e}")
            raise
    
    def query_with_images(
        self,
        text: str,
        images: List[Union[Path, str]],
        model: ModelChoice,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> LLMResponse:
        """
        Query model with text and images (multimodal).
        
        Args:
            text: The query text
            images: List of image paths to encode
            model: Which model to use
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with model output
        """
        # Build multimodal message
        content_parts = [{"type": "text", "text": text}]
        
        for image_path in images:
            encoded_image = self._encode_image(image_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            })
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
        
        logger.info(f"Multimodal query with {len(images)} images")
        
        return self.query(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def _get_deployment_name(self, model: ModelChoice) -> str:
        """Get Azure deployment name for the model."""
        if model == ModelChoice.GPT_4_1:
            return self.gpt41_deployment
        elif model == ModelChoice.GPT_5_MINI:
            return self.gpt5_deployment
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def _normalize_parameters(
        self,
        model: ModelChoice,
        max_tokens: int,
        temperature: Optional[float],
        response_format: Optional[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Normalize parameters based on model requirements.
        
        GPT-4.1:
        - Uses 'max_tokens'
        - Supports 'temperature'
        
        GPT-5-mini (reasoning model):
        - Uses 'max_completion_tokens'
        - Does NOT support 'temperature' (fixed at 1.0)
        """
        params: Dict[str, Any] = {}
        
        if model == ModelChoice.GPT_4_1:
            # GPT-4.1 parameters
            params["max_tokens"] = max_tokens
            
            if temperature is not None:
                params["temperature"] = temperature
            
        elif model == ModelChoice.GPT_5_MINI:
            # GPT-5-mini parameters (reasoning model)
            params["max_completion_tokens"] = max_tokens
            
            # Note: temperature is NOT supported for reasoning models
            if temperature is not None:
                logger.warning(
                    f"Temperature parameter ignored for {model.value} "
                    f"(reasoning models use fixed temperature=1.0)"
                )
        
        # Common parameters
        if response_format:
            params["response_format"] = response_format
        
        # Add any additional kwargs
        params.update(kwargs)
        
        return params
    
    def _encode_image(self, image_path: Union[Path, str]) -> str:
        """
        Encode image as base64 for API submission.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64-encoded image string
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        
        logger.debug(f"Encoded image: {image_path} ({image_path.stat().st_size} bytes)")
        
        return encoded


# Singleton instance
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
