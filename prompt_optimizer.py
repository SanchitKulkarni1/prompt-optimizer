# prompt_optimizer.py
import re
from typing import Optional, List, Dict, Any

from token_manager import TokenManager
from model_adapter import ModelAdapter
from semantic_utils import SemanticHelper
from keybert import KeyBERT

class PromptOptimizer:
    def __init__(self, compressor, token_model: str = "gpt-4"):
        """
        compressor: object with compress(text, rate, preserve_keywords) -> str
        """
        self.compressor = compressor
        self.token_manager = TokenManager(gpt_model=token_model)
        self.adapter = ModelAdapter()
        # semantic helper (small model by default)
        # self.semantic = SemanticHelper(model_name="all-MiniLM-L6-v2")
        self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')

    # def analyze_prompt(self, prompt_text: str) -> Dict[str, List[str]]:
    #     """
    #     Semantic-aware keyword/phrase extractor:
    #      - chooses top sentences by embedding centrality
    #      - extracts frequent tokens from those sentences and normalizes known phrases
    #      - returns prioritized list of phrases/keywords (cap 20)
    #     """
    #     if not prompt_text:
    #         return {"keywords": []}

    #     text = prompt_text.strip()
    #     # 1) get top representative sentences
    #     top_sentences = self.semantic.summarize_sentences(text, top_n=4)

    #     # 2) extract candidate keywords from those sentences
    #     words = []
    #     for s in top_sentences:
    #         # extract alpha-numeric tokens (allow ML, C++)
    #         toks = re.findall(r"\b[a-zA-Z0-9\+\#]{2,}\b", s)
    #         words.extend(toks)

    #     # small stopword set
    #     stopwords = {
    #         # Original
    #         "the", "and", "that", "this", "with", "for", "from", "your", "please",
    #         "would", "could", "should", "a", "an", "to", "in", "of", "on", "it", "is", "are",
    #         "me", "my", "he", "she", "they", "we", "i",
    #         "am", "was", "be", "do", "does", "did", "have", "has", "had",
    #         "what", "where", "when", "why", "how",
    #         "write", "describe", "describing", "ask", "asking", "give", "tell" # Filter out common instruction verbs
    #     }
    #     filtered = [w for w in words if w.lower() not in stopwords and len(w) >= 2]

    #     # frequency ranking
    #     freq = {}
    #     for w in filtered:
    #         lw = w.lower()
    #         freq[lw] = freq.get(lw, 0) + 1
    #     sorted_words = sorted(freq.keys(), key=lambda k: -freq[k])

    #     # detect known phrases
    #     known_phrases = ["machine learning", "deep learning", "data science", "ml", "nlp", "computer vision"]
    #     phrases_found = [p for p in known_phrases if p in text.lower()]

    #     # build final keyword list: phrases first, then frequent tokens
    #     keywords: List[str] = []
    #     for p in phrases_found:
    #         keywords.append("ML" if p == "ml" else p)
    #     for w in sorted_words:
    #         if w.upper() == "ML" and "ML" in keywords:
    #             continue
    #         if w not in keywords:
    #             keywords.append(w)
    #         if len(keywords) >= 20:
    #             break

    #     # expand/normalize keywords using SemanticHelper (small expansion)
    #     expanded = self.semantic.expand_keywords(keywords, top_k=20)
    #     # ensure uniqueness and preserve order
    #     seen = set()
    #     final = []
    #     for k in expanded + keywords:
    #         key_norm = k.strip()
    #         if not key_norm:
    #             continue
    #         if key_norm.lower() not in seen:
    #             final.append(key_norm)
    #             seen.add(key_norm.lower())
    #         if len(final) >= 20:
    #             break

    #     return {"keywords": final[:20]}

    def analyze_prompt(self, prompt_text: str) -> Dict[str, List[str]]:
            """
            Extracts keywords using KeyBERT.
            This finds keywords/phrases most similar to the full prompt text.
            """
            if not prompt_text:
                return {"keywords": []}

            text = prompt_text.strip()

            # --- REPLACEMENT KEYWORD LOGIC ---
            # keyphrase_ngram_range: look for single words or 2-word phrases
            # stop_words='english': use a built-in stopword list
            # top_n=10: get the top 10 most relevant phrases

            keywords_and_scores = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3), 
                stop_words='english', 
                use_mmr=True,
                diversity=0.7,
                top_n=25
                
            )

            # KeyBERT returns tuples of (keyword, score). We just want the keywords.
            keywords = [k[0] for k in keywords_and_scores]

            # --- MANUAL ADDITION: Add back any very specific terms
            # KeyBERT might miss acronyms. We can add them back.
            if "ml" in text.lower() and "ml" not in keywords:
                keywords.append("ML")

            # Your old code was good at finding known phrases
            known_phrases = ["machine learning", "deep learning", "data science"]
            for p in known_phrases:
                if p in text.lower() and p not in keywords:
                    keywords.append(p)
            # --- END REPLACEMENT LOGIC ---

            return {"keywords": keywords}

# prompt_optimizer.py

    def optimize(
        self,
        prompt_text: str,
        target_model: str,
        compression_ratio: float = 0.5,
        preserve_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main optimization pipeline:
        - get semantic keywords
        - compress using compressor.compress(...)
        - produce canonical optimized_prompt with clearer system instruction
        - *** NEW: Bypass if compression fails (makes prompt longer) ***
        """
        # Get original token count first
        original_token_count = self.token_manager.count_tokens(prompt_text, target_model)

        analysis = self.analyze_prompt(prompt_text)
        preserve = preserve_keywords if preserve_keywords else analysis.get("keywords", [])

        compressed_text = self.compressor.compress(
            prompt_text,
            rate=compression_ratio,
            preserve_keywords=preserve
        )
        
        compressed_token_count = self.token_manager.count_tokens(compressed_text, target_model)

        # --- THIS IS THE NEW BYPASS LOGIC ---
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

        return {
            "optimized_prompt": optimized_prompt,
            "adapted_payload": adapted,
            "token_count": final_token_count,
            "compression_rate": compression_rate
            # 'cached' key is missing, but app.py adds it
        }