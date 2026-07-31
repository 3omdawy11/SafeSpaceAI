"""
src/language_detector/llm_language_detector.py
=================================================
LLM-backed drop-in replacement for LanguageDetector (the TF-IDF + LR model).
Same public contract — detect(text) — so it can be swapped in via
LANGUAGE_CLASSIFIER_BACKEND=llm without touching callers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ISO_CODE_RE = re.compile(r"^[a-z]{2}$")

_LANGUAGE_PROMPT = """You are a language identification engine.
Identify the ISO 639-1 language code (e.g. "en", "es", "ar", "fr", "zh") of the
user's message.

Return ONLY valid JSON, no explanation, no markdown, exactly this schema:
{{"language": "two-letter ISO 639-1 code, lowercase", "confidence": 0.0 to 1.0}}

Message: "{text}"
JSON:"""


class LLMLanguageDetector:
    """
    Zero-shot language detection via Groq LLM.

    Parameters
    ----------
    api_key : Groq API key (defaults to GROQ_API_KEY_LANGUAGE_CLASSIFIER env var,
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
            or os.getenv("GROQ_API_KEY_LANGUAGE_CLASSIFIER", "")
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

    def _detect_via_llm(self, text: str) -> dict[str, Any]:
        prompt = _LANGUAGE_PROMPT.format(text=text)
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)

        lang = str(data.get("language", "")).strip().lower()
        if not _ISO_CODE_RE.match(lang):
            lang = "en"

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "language": lang,
            "confidence": confidence,
            "method": "llm",
            "all_scores": None,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, text: str) -> dict[str, Any]:
        """
        Detect the language of *text*.

        Returns
        -------
        {"language": "en", "confidence": 0.95, "method": "llm", "all_scores": None}
        """
        text = (text or "").strip()
        if not text:
            return {"language": "en", "confidence": 0.0, "method": "llm_empty", "all_scores": None}

        if not self._api_key:
            logger.warning("LLMLanguageDetector: no API key set — defaulting to 'en'")
            return {"language": "en", "confidence": 0.0, "method": "llm_no_key", "all_scores": None}

        try:
            return self._detect_via_llm(text)
        except Exception as exc:
            logger.warning(f"LLMLanguageDetector failed: {exc}")
            return {"language": "en", "confidence": 0.0, "method": "llm_error_fallback", "all_scores": None}

    def detect_batch(self, texts: "list[str]") -> "list[dict[str, Any]]":
        return [self.detect(t) for t in texts]

    def __repr__(self) -> str:
        mode = "ready" if self._api_key else "no-api-key"
        return f"LLMLanguageDetector(model='{self._model}', status={mode})"
