"""
Token counting utilities for accurate context size estimation.

Provides accurate token counting using tiktoken library (OpenAI's tokenizer)
and supports estimation for both text and image tokens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token counting")


class TokenCounter:
    """Token counting for GPT models."""
    
    # Model-specific encodings
    ENCODINGS = {
        "gpt-4.1": "cl100k_base",  # GPT-4 series
        "gpt-5-mini": "cl100k_base",  # GPT-5 series uses same encoding
    }
    
    # Image token estimates (varies by resolution)
    # For 512x512 images: ~765 tokens
    # For 1024x1024 images: ~1500 tokens
    # Conservative estimate: 1000 tokens per image
    IMAGE_TOKENS_ESTIMATE = 1000
    
    def __init__(self, model: str = "gpt-4.1"):
        """
        Initialize token counter for a specific model.
        
        Args:
            model: Model name (gpt-4.1 or gpt-5-mini)
        """
        self.model = model
        self.encoding = None
        
        if TIKTOKEN_AVAILABLE:
            encoding_name = self.ENCODINGS.get(model, "cl100k_base")
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
                logger.debug(f"Initialized tiktoken encoding: {encoding_name}")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")
                self.encoding = None
    
    def count_text_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: The text to tokenize
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self.encoding:
            # Accurate counting with tiktoken
            return len(self.encoding.encode(text))
        else:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4
    
    def count_message_tokens(self, messages: List[dict]) -> int:
        """
        Count tokens in a list of chat messages.
        
        Includes overhead for message formatting:
        - Each message: ~4 tokens overhead
        - Role names: ~1 token each
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Total token count
        """
        if not messages:
            return 0
        
        total = 0
        
        for message in messages:
            # Message overhead (varies by model, ~4 tokens is conservative)
            total += 4
            
            # Role tokens
            if "role" in message:
                total += self.count_text_tokens(message["role"])
            
            # Content tokens
            if "content" in message:
                content = message["content"]
                
                if isinstance(content, str):
                    # Simple text content
                    total += self.count_text_tokens(content)
                
                elif isinstance(content, list):
                    # Multimodal content (text + images)
                    for part in content:
                        if part.get("type") == "text":
                            total += self.count_text_tokens(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            total += self.IMAGE_TOKENS_ESTIMATE
        
        # Add 2 tokens for reply priming
        total += 2
        
        return total
    
    def count_image_tokens(self, image_count: int) -> int:
        """
        Estimate tokens for images.
        
        Args:
            image_count: Number of images
            
        Returns:
            Estimated token count
        """
        return image_count * self.IMAGE_TOKENS_ESTIMATE
    
    def estimate_context_size(
        self,
        query_text: str,
        images: Optional[List[Union[Path, str]]] = None,
        context_docs: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Estimate total context size for a query.
        
        Args:
            query_text: The user's query
            images: Optional list of image paths
            context_docs: Optional list of context documents
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with token breakdown
        """
        images = images or []
        context_docs = context_docs or []
        
        # Count tokens
        query_tokens = self.count_text_tokens(query_text)
        system_tokens = self.count_text_tokens(system_prompt) if system_prompt else 0
        context_tokens = sum(self.count_text_tokens(doc) for doc in context_docs)
        image_tokens = self.count_image_tokens(len(images))
        
        # Message overhead (~10 tokens for formatting)
        overhead = 10
        
        total = query_tokens + system_tokens + context_tokens + image_tokens + overhead
        
        return {
            "total_tokens": total,
            "query_tokens": query_tokens,
            "system_tokens": system_tokens,
            "context_tokens": context_tokens,
            "image_tokens": image_tokens,
            "overhead_tokens": overhead,
            "image_count": len(images),
            "context_doc_count": len(context_docs)
        }
    
    def fits_in_context(
        self,
        estimated_tokens: int,
        model: str,
        output_buffer: int = 50_000
    ) -> bool:
        """
        Check if estimated tokens fit in model's context window.
        
        Args:
            estimated_tokens: Estimated input token count
            model: Model name (gpt-4.1 or gpt-5-mini)
            output_buffer: Tokens to reserve for output
            
        Returns:
            True if fits, False otherwise
        """
        max_context = {
            "gpt-4.1": 1_047_576,
            "gpt-5-mini": 400_000
        }
        
        limit = max_context.get(model, 400_000) - output_buffer
        
        return estimated_tokens <= limit


# Utility functions for quick access

def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Quick function to count tokens in text."""
    counter = TokenCounter(model=model)
    return counter.count_text_tokens(text)


def estimate_query_tokens(
    query_text: str,
    images: Optional[List[Union[Path, str]]] = None,
    context_docs: Optional[List[str]] = None,
    model: str = "gpt-4.1"
) -> dict:
    """Quick function to estimate total context size."""
    counter = TokenCounter(model=model)
    return counter.estimate_context_size(query_text, images, context_docs)


def check_context_fit(
    estimated_tokens: int,
    model: str = "gpt-4.1"
) -> bool:
    """Quick function to check if tokens fit in context."""
    counter = TokenCounter(model=model)
    return counter.fits_in_context(estimated_tokens, model)
