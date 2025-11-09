# model_adapter.py
import re
from typing import Dict, Any, List

KEYWORD_LINE_RE = re.compile(r"(?mi)^\s*(Important keywords:)\s*(.+)$")

class ModelAdapter:
    def _extract_keyword_line(self, optimized_prompt: Dict[str, Any]) -> List[str]:
        """
        Search for an "Important keywords: ..." line inside optimized_prompt.flat
        or inside the first user message; return list of keywords (trimmed).
        """
        text_candidates = []
        if optimized_prompt.get("flat"):
            text_candidates.append(optimized_prompt["flat"])
        # also look inside messages (join to find if flat missing)
        for m in optimized_prompt.get("messages", []):
            if isinstance(m, dict) and m.get("content"):
                text_candidates.append(m["content"])

        for txt in text_candidates:
            match = KEYWORD_LINE_RE.search(txt)
            if match:
                kws = match.group(2)
                # split by comma or semicolon, strip whitespace
                parts = re.split(r"[,;]\s*", kws)
                return [p.strip() for p in parts if p.strip()]
        return []

    def _ensure_in_system(self, system: str, keywords: List[str]) -> str:
        """
        Ensure a clear directive containing the keywords is present in the system message.
        If system already includes an Important keywords line, don't duplicate.
        """
        if not keywords:
            return system or ""
        if system and KEYWORD_LINE_RE.search(system):
            return system  # already present
        kw_line = "Important keywords: " + ", ".join(keywords)
        # Prefer concise directive first, then existing system text
        if system:
            return f"{kw_line}\n\n{system}"
        return kw_line

    def adapt(self, optimized_prompt: dict, target_model: str) -> Dict[str, Any]:
        """
        optimized_prompt: {"system": "...", "messages": [{"role":"user","content":"..."}], "flat": "..." }
        This adapter will:
         - extract the Important keywords line (if any)
         - promote it into the system message (or XML wrapper for Claude)
         - return provider-specific payload shape
        """
        t = target_model.lower()
        # find keywords from flat/messages
        keywords = self._extract_keyword_line(optimized_prompt)

        # canonical system message (if present)
        orig_system = optimized_prompt.get("system", "") or ""

        # GPT family: messages[] with a system message first
        if "gpt" in t:
            system_msg = self._ensure_in_system(orig_system, keywords)
            messages = []
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            # append user/assistant messages from optimized_prompt
            messages.extend(optimized_prompt.get("messages", []))
            return {"messages": messages}

        # Claude family: promote keywords into system and wrap in <context> XML
        if "claude" in t or "anthropic" in t:
            system_msg = self._ensure_in_system(orig_system, keywords)
            # wrap in a <context> tag if not present
            if "<context>" not in system_msg.lower():
                system_msg = f"<context>{system_msg}</context>"
            payload = {
                "system": system_msg,
                "messages": optimized_prompt.get("messages", [])
            }
            return payload

        # Default / other providers: return a concatenated input string ensuring keywords are visible
        # Build an explicit input that starts with the kw line (if present) then the flat prompt.
        flat = optimized_prompt.get("flat") or ""
        if keywords:
            kw_line = "Important keywords: " + ", ".join(keywords)
            # avoid duplicating if flat already starts with kw_line
            if flat.strip().lower().startswith("important keywords:"):
                input_text = flat
            else:
                input_text = f"{kw_line}\n\n{flat}"
        else:
            input_text = flat or " ".join(m.get("content", "") for m in optimized_prompt.get("messages", []))
        return {"input": input_text}
