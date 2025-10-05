"""
Query rewriting and expansion for improved semantic search.

This module handles query preprocessing to improve retrieval accuracy,
particularly for short or ambiguous queries that may not produce good
semantic embeddings.

Key features:
- Detects and rewrites "What is X?" pattern queries
- Expands short definitional queries to include multiple phrasings
- Improves BiomedCLIP embedding discrimination for simple queries
"""

import re
from typing import Optional


class QueryRewriter:
    """
    Rewrites and expands user queries to improve semantic search results.
    
    Problem solved:
    - Short queries like "What is niraparib?" produce embeddings with poor
      discrimination (all chunks get similar low scores like 0.663)
    - Longer queries or alternative phrasings like "Define niraparib" work better
    
    Solution:
    - Detect common query patterns (What is, What are, etc.)
    - Expand them to include multiple semantic variations
    - This improves chunk ranking and retrieval quality
    """
    
    # Pattern for "What is/are X?" queries
    WHAT_IS_PATTERN = re.compile(
        r"^what\s+(is|are)\s+(.+?)(?:\?|$)",
        re.IGNORECASE
    )
    
    # Pattern for "How does X work?" queries
    HOW_DOES_PATTERN = re.compile(
        r"^how\s+does\s+(.+?)\s+work(?:\?|$)",
        re.IGNORECASE
    )
    
    def __init__(self, enable_rewriting: bool = True):
        """
        Initialize query rewriter.
        
        Args:
            enable_rewriting: If False, queries pass through unchanged (for debugging)
        """
        self.enable_rewriting = enable_rewriting
    
    def rewrite(self, query: str) -> str:
        """
        Rewrite query to improve semantic search if needed.
        
        Args:
            query: Original user query
            
        Returns:
            Rewritten/expanded query or original if no rewriting needed
            
        Examples:
            >>> rewriter = QueryRewriter()
            >>> rewriter.rewrite("What is niraparib?")
            "Define niraparib. Niraparib mechanism of action. Niraparib description. What is niraparib."
            
            >>> rewriter.rewrite("What are adverse events with niraparib?")
            # No rewriting - query is specific enough
            "What are adverse events with niraparib?"
        """
        if not self.enable_rewriting:
            return query
        
        # Check for "What is X?" pattern
        match = self.WHAT_IS_PATTERN.match(query.strip())
        if match:
            is_are, subject = match.groups()
            
            # Only rewrite if the subject is short (likely a single entity)
            # Don't rewrite complex questions like "What is the relationship between X and Y?"
            if len(subject.split()) <= 3 and not any(
                kw in subject.lower() 
                for kw in ["relationship", "difference", "between", "compared to"]
            ):
                return self._expand_definitional_query(subject, is_are)
        
        # Check for "How does X work?" pattern
        match = self.HOW_DOES_PATTERN.match(query.strip())
        if match:
            subject = match.group(1)
            if len(subject.split()) <= 3:
                return self._expand_mechanism_query(subject)
        
        # No rewriting needed
        return query
    
    def _expand_definitional_query(self, subject: str, is_are: str) -> str:
        """
        Expand "What is X?" to multiple phrasings for better embedding.
        
        Args:
            subject: The entity being asked about (e.g., "niraparib")
            is_are: "is" or "are" from original query
            
        Returns:
            Expanded query with multiple phrasings
        """
        # Build expanded query with multiple semantic variations
        expansions = [
            f"Define {subject}.",
            f"{subject} mechanism of action.",
            f"{subject} description.",
            f"What {is_are} {subject}."  # Include original for context
        ]
        
        return " ".join(expansions)
    
    def _expand_mechanism_query(self, subject: str) -> str:
        """
        Expand "How does X work?" to multiple phrasings.
        
        Args:
            subject: The entity being asked about
            
        Returns:
            Expanded query with multiple phrasings
        """
        expansions = [
            f"{subject} mechanism of action.",
            f"{subject} mode of action.",
            f"How does {subject} work.",
            f"{subject} pharmacology."
        ]
        
        return " ".join(expansions)
    
    def explain_rewrite(self, original: str, rewritten: str) -> Optional[str]:
        """
        Generate explanation if query was rewritten.
        
        Args:
            original: Original query
            rewritten: Rewritten query
            
        Returns:
            Human-readable explanation or None if no rewriting occurred
        """
        if original == rewritten:
            return None
        
        return (
            f"📝 Query expanded for better results:\n"
            f"   Original: '{original}'\n"
            f"   Expanded: '{rewritten}'\n"
            f"   Reason: Short definitional queries benefit from multiple phrasings"
        )


# Convenience function for quick usage
def rewrite_query(query: str, enable: bool = True) -> str:
    """
    Quick function to rewrite a query.
    
    Args:
        query: User query string
        enable: Enable rewriting (set to False for debugging)
        
    Returns:
        Rewritten query
    """
    rewriter = QueryRewriter(enable_rewriting=enable)
    return rewriter.rewrite(query)
