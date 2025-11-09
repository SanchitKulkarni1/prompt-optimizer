# semantic_utils.py
from sentence_transformers import SentenceTransformer, util
import re
from typing import List

class SemanticHelper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Small, fast model good for semantic similarity and clustering.
        Swap model_name for higher fidelity models if you have RAM/CPU.
        """
        self.model = SentenceTransformer(model_name)

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Return cosine similarity (0..1) between two texts.
        """
        if not text_a or not text_b:
            return 0.0
        emb_a = self.model.encode(text_a, convert_to_tensor=True)
        emb_b = self.model.encode(text_b, convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b)
        return float(sim.item())

    def summarize_sentences(self, text: str, top_n: int = 5) -> List[str]:
        """
        Return top_n representative sentences from the text based on embedding centrality.
        Uses simple regex sentence splitting (no nltk).
        """
        if not text or not text.strip():
            return []

        # crude sentence split (handles ., ?, !)
        sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
        if not sentences:
            return []

        if len(sentences) <= top_n:
            return sentences

        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        mean_emb = embeddings.mean(dim=0, keepdim=True)
        sims = util.cos_sim(mean_emb, embeddings)[0]
        top_idx = sims.argsort(descending=True)[:top_n]
        top = [sentences[i] for i in top_idx]
        return top

    def expand_keywords(self, keywords: List[str], top_k: int = 5) -> List[str]:
        """
        Given short keywords, produce small semantically-related expansions by encoding
        and finding nearest tokens within the same keyword list. This is lightweight — primarily
        returns normalized keywords back. Placeholder for a future more complex implementation.
        """
        # For now just normalize casing and dedupe
        out = []
        for k in keywords:
            t = k.strip()
            if not t:
                continue
            # normalize common abbreviations
            if t.lower() == "ml":
                out.append("ML")
                out.append("machine learning")
            else:
                out.append(t)
        # dedupe preserving order
        seen = set()
        final = []
        for x in out:
            if x.lower() not in seen:
                final.append(x)
                seen.add(x.lower())
            if len(final) >= top_k:
                break
        return final
