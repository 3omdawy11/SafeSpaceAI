"""
src/utils/memory_summarizer.py
=================================
Summarizes the last few conversation turns into a short rolling memory
note via Groq LLM, so ConversationManager can augment its session context
with an abstractive summary instead of just accumulated keyword lists.

Usage
-----
    from src.utils.memory_summarizer import MemorySummarizer
    summarizer = MemorySummarizer()
    summary = summarizer.summarize([
        {"role": "user", "content": "I've been anxious about work"},
        {"role": "assistant", "content": "That sounds difficult..."},
        {"role": "user", "content": "It's mostly the deadlines"},
    ])
    # "User is anxious about work, mainly due to looming deadlines."
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize the following snippet of a mental health support \
conversation in ONE short sentence (max 30 words). Focus on what the user is \
feeling and dealing with — no preamble, no quotes, just the sentence.

{transcript}

Summary:"""


class MemorySummarizer:
    """
    Parameters
    ----------
    api_key : Groq API key (defaults to GROQ_API_KEY_SUMMARIZER env var,
              falling back to GROQ_API_KEY if unset).
    model   : Groq model name.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 8.0,
    ) -> None:
        self._api_key = (
            api_key
            or os.getenv("GROQ_API_KEY_SUMMARIZER", "")
            or os.getenv("GROQ_API_KEY", "")
        )
        self._model = model
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key, timeout=self._timeout)
        return self._client

    def _fallback(self, turns: list[dict]) -> str:
        """Naive concatenation, used when no API key is set or the call fails."""
        parts = [f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns]
        joined = " | ".join(parts)
        return joined[:200]

    def summarize(self, turns: list[dict]) -> str:
        """
        Summarize a small window of turns (each {"role": ..., "content": ...}).
        Returns an empty string if turns is empty.
        """
        if not turns:
            return ""

        if not self._api_key:
            return self._fallback(turns)

        transcript = "\n".join(
            f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns
        )
        prompt = _SUMMARY_PROMPT.format(transcript=transcript)

        try:
            resp = self._get_client().chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.2,
            )
            summary = (resp.choices[0].message.content or "").strip()
            return summary or self._fallback(turns)
        except Exception as exc:
            logger.warning(f"MemorySummarizer failed: {exc}")
            return self._fallback(turns)

    def __repr__(self) -> str:
        mode = "llm" if self._api_key else "fallback"
        return f"MemorySummarizer(model='{self._model}', mode={mode})"
