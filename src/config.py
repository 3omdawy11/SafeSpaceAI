"""
src/config.py
==============
Central place to resolve runtime backend choices from environment variables.

Both the FastAPI startup (src/api/app.py) and the legacy Orchestrator
(src/pipeline/orchestrator.py) build their classifiers through the factory
functions here, so a single env var flips the behaviour everywhere instead
of being checked in two different places.

Env vars
--------
EMOTION_CLASSIFIER_BACKEND    : "model" (default) | "llm"
LANGUAGE_CLASSIFIER_BACKEND   : "model" (default) | "llm"
MEMORY_SUMMARIZATION_ENABLED  : "true" (default) | "false"

GROQ_API_KEY_EMOTION_CLASSIFIER  : key used when EMOTION_CLASSIFIER_BACKEND=llm
GROQ_API_KEY_LANGUAGE_CLASSIFIER : key used when LANGUAGE_CLASSIFIER_BACKEND=llm
GROQ_API_KEY_SUMMARIZER          : key used for conversation memory summarization
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# Every other module in this codebase hardcodes this model name for Groq
# calls (translator, NER, intent classifier) — reuse it here for consistency.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "")


def emotion_classifier_backend() -> str:
    return os.getenv("EMOTION_CLASSIFIER_BACKEND", "model").strip().lower()


def language_classifier_backend() -> str:
    return os.getenv("LANGUAGE_CLASSIFIER_BACKEND", "model").strip().lower()


def memory_summarization_enabled() -> bool:
    return _flag("MEMORY_SUMMARIZATION_ENABLED", True)


def build_emotion_classifier():
    """Return an EmotionClassifier-compatible instance per EMOTION_CLASSIFIER_BACKEND."""
    if emotion_classifier_backend() == "llm":
        from src.emotion_classifier.llm_emotion_classifier import LLMEmotionClassifier
        return LLMEmotionClassifier(api_key=os.getenv("GROQ_API_KEY_EMOTION_CLASSIFIER", ""))

    from src.emotion_classifier.emotion_classifier import EmotionClassifier
    return EmotionClassifier()


def build_language_detector():
    """Return a LanguageDetector-compatible instance per LANGUAGE_CLASSIFIER_BACKEND."""
    if language_classifier_backend() == "llm":
        from src.language_detector.llm_language_detector import LLMLanguageDetector
        return LLMLanguageDetector(api_key=os.getenv("GROQ_API_KEY_LANGUAGE_CLASSIFIER", ""))

    from src.language_detector.language_detector import LanguageDetector
    return LanguageDetector()


def build_memory_summarizer():
    """Return a MemorySummarizer instance, or None if summarization is disabled."""
    if not memory_summarization_enabled():
        return None

    from src.utils.memory_summarizer import MemorySummarizer
    return MemorySummarizer(api_key=os.getenv("GROQ_API_KEY_SUMMARIZER", ""))
