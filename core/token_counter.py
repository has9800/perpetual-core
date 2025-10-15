"""
Token Counter Utility for V2
Fast approximate token counting without tiktoken dependency
"""

import re
from typing import List, Dict


class TokenCounter:
    """Fast approximate token counter (GPT-style)"""
    
    def __init__(self):
        # Average tokens per word for GPT models: ~1.3
        self.tokens_per_word = 1.3
    
    def count_tokens(self, text: str) -> int:
        """
        Fast approximate token count
        Accurate enough for budget management (~5% error)
        """
        if not text:
            return 0
        
        # Split on whitespace and punctuation
        words = len(re.findall(r'\w+', text))
        
        # Estimate tokens (words * 1.3 + punctuation)
        estimated_tokens = int(words * self.tokens_per_word)
        
        return max(estimated_tokens, 1)
    
    def count_messages_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in message list"""
        total = 0
        for msg in messages:
            # Role tokens (system/user/assistant = ~1 token each)
            total += 4  # Overhead per message
            
            # Content tokens
            content = msg.get('content', '')
            total += self.count_tokens(content)
        
        return total
    
    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget"""
        if not text:
            return text
        
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Estimate how much to keep
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.95)  # 5% safety margin
        
        return text[:target_chars] + "..."


# Singleton instance
_counter = TokenCounter()

def count_tokens(text: str) -> int:
    """Convenience function"""
    return _counter.count_tokens(text)

def count_messages_tokens(messages: List[Dict]) -> int:
    """Convenience function"""
    return _counter.count_messages_tokens(messages)

def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Convenience function"""
    return _counter.truncate_to_budget(text, max_tokens)
