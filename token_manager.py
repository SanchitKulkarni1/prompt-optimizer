# token_manager.py
import tiktoken

class TokenManager:
    def __init__(self, gpt_model="gpt-4"):
        self.gpt_encoding = tiktoken.encoding_for_model(gpt_model)
        self.claude_multiplier = 1.3  # estimate (Claude uses ~20-30% more tokens)
    
    def count_tokens(self, text: str, model_type: str) -> int:
        base = len(self.gpt_encoding.encode(text))
        if "claude" in model_type.lower():
            return int(base * self.claude_multiplier)
        return base

    def compression_rate(self, original: str, compressed: str, model_type: str):
        o = self.count_tokens(original, model_type)
        c = self.count_tokens(compressed, model_type)
        return c / o if o else 0
