"""
src/emotion_classifier/llm_emotion_classifier.py
===================================================
LLM-backed drop-in replacement for EmotionClassifier (the trained BiLSTM
model). Same public contract — classify(text) / predict(text) — so it can
be swapped in via EMOTION_CLASSIFIER_BACKEND=llm without touching callers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Mirrors EMOTION_LABELS in emotion_classifier.py — duplicated (not imported)
# so picking the "llm" backend doesn't force-load torch/transformers.
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

_EMOTION_PROMPT = """You are an emotion classifier for a mental health support chatbot.
Classify the dominant emotion expressed in the user's message.

Return ONLY valid JSON, no explanation, no markdown, exactly this schema:
{{"emotion": "one of: sadness, joy, love, anger, fear, surprise", "confidence": 0.0 to 1.0}}

User message: "{text}"
JSON:"""


class LLMEmotionClassifier:
    """
    Zero-shot emotion classification via Groq LLM.

    Parameters
    ----------
    api_key : Groq API key (defaults to GROQ_API_KEY_EMOTION_CLASSIFIER env var,
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
            or os.getenv("GROQ_API_KEY_EMOTION_CLASSIFIER", "")
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

    def _classify_via_llm(self, text: str) -> dict:
        prompt = _EMOTION_PROMPT.format(text=text)
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)

        emotion = str(data.get("emotion", "")).strip().lower()
        if emotion not in EMOTION_LABELS:
            emotion = "unknown"

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        all_scores = {lbl: 0.0 for lbl in EMOTION_LABELS}
        if emotion in all_scores:
            all_scores[emotion] = confidence

        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1])),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str) -> dict:
        """
        Classify a single text.

        Returns
        -------
        {"emotion": "sadness", "confidence": 0.87, "all_scores": {...}}
        """
        if not isinstance(text, str) or not text.strip():
            return {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}

        if not self._api_key:
            logger.warning("LLMEmotionClassifier: no API key set — returning unknown")
            return {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}

        try:
            return self._classify_via_llm(text)
        except Exception as exc:
            logger.warning(f"LLMEmotionClassifier failed: {exc}")
            return {"emotion": "unknown", "confidence": 0.0, "all_scores": {}}

    def predict(self, text: str) -> dict:
        """Alias to keep consistent with the FastAPI endpoint expectations."""
        return self.classify(text)

    def classify_batch(self, texts: list[str]) -> list[dict]:
        return [self.classify(t) for t in texts]

    @property
    def labels(self) -> list[str]:
        return EMOTION_LABELS

    def __repr__(self) -> str:
        mode = "ready" if self._api_key else "no-api-key"
        return f"LLMEmotionClassifier(model='{self._model}', status={mode})"
