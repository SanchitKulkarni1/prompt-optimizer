# prompt_optimizer.py
import re
from typing import Optional, List, Dict, Any

from token_manager import TokenManager
from model_adapter import ModelAdapter
from keybert import KeyBERT

class PromptOptimizer:
    def __init__(self, compressor, token_model: str = "gpt-4"):
        """
        compressor: object with compress(text, rate, preserve_keywords) -> str
        """
        self.compressor = compressor
        self.token_manager = TokenManager(gpt_model=token_model)
        self.adapter = ModelAdapter()
        # Initialize KeyBERT
        self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')

    def analyze_prompt(self, prompt_text: str, mode: str = "balanced") -> Dict[str, List[str]]:
            """
            Extracts keywords using KeyBERT.
            Tuning (top_n, diversity) is now controlled by the 'mode'.
            """
            if not prompt_text:
                return {"keywords": []}
    
            text = prompt_text.strip()
    
            # --- NEW LOGIC: Set tuning based on mode ---
            if mode.lower() == "conservative":
                # More keywords = less compression = more conservative
                top_n = 30
                diversity = 0.6
            elif mode.lower() == "aggressive":
                # Fewer keywords = more compression = aggressive
                top_n = 15
                diversity = 0.8
            else: # "balanced"
                top_n = 25
                diversity = 0.7
            # --- END NEW LOGIC ---
    
            keywords_and_scores = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3), 
                stop_words='english',
                use_mmr=True,          
                diversity=diversity,  # Use the new variable
                top_n=top_n           # Use the new variable
            )
    
            # KeyBERT returns tuples of (keyword, score). We just want the keywords.
            keywords = [k[0] for k in keywords_and_scores]
    
            # --- MANUAL ADDITION: Add back any very specific terms
            if "ml" in text.lower() and "ml" not in keywords:
                keywords.append("ML")
    
            known_phrases = ["machine learning", "deep learning", "data science"]
            for p in known_phrases:
                if p in text.lower() and p not in keywords:
                    keywords.append(p)
            
            return {"keywords": keywords}

    def optimize(
        self,
        prompt_text: str,
        target_model: str,
        mode: str = "balanced",  # <-- CHANGED: No more compression_ratio
        preserve_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main optimization pipeline:
        - get semantic keywords (unless user supplied)
        - compress using compressor.compress(...)
        - Bypass logic is still active
        """
        # Get original token count first
        original_token_count = self.token_manager.count_tokens(prompt_text, target_model)

        # Pass the mode to analyze_prompt
        analysis = self.analyze_prompt(prompt_text, mode=mode) # <-- CHANGED
        preserve = preserve_keywords if preserve_keywords else analysis.get("keywords", [])

        compressed_text = self.compressor.compress(
            prompt_text,
            rate=0.5,  # This 'rate' is ignored by SimpleCompressor if keywords are present
            preserve_keywords=preserve
        )
        
        compressed_token_count = self.token_manager.count_tokens(compressed_text, target_model)

        # --- THIS IS THE BYPASS LOGIC ---
        if compressed_token_count >= original_token_count:
            # Compression failed or made no change. Use the original.
            final_text = prompt_text
            final_token_count = original_token_count
            final_system_prompt = "You are a helpful assistant." # Basic system prompt
        else:
            # Compression succeeded! Use the compressed text and keyword prompt.
            final_text = compressed_text
            final_token_count = compressed_token_count
            final_system_prompt = "You are a helpful assistant. Follow the 'Important keywords' line and emphasize them in your output."
        # --- END BYPASS LOGIC ---

        optimized_prompt = {
            "system": final_system_prompt,
            "messages": [{"role": "user", "content": final_text}],
            "flat": final_text
        }

        adapted = self.adapter.adapt(optimized_prompt, target_model)
        compression_rate = final_token_count / original_token_count if original_token_count else 0

        # This dictionary is what 'app.py' will receive
        return {
            "optimized_prompt": optimized_prompt,
            "adapted_payload": adapted,
            "token_count": final_token_count, # This will become 'optTokens'
            "compression_rate": compression_rate
        }