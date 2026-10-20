# SafeSpace AI — Mental Health Support Chatbot (Backend)

The FastAPI backend for **SafeSpace AI**, a multilingual mental health support chatbot. It detects the user's language, emotion and intent, screens every message for crisis signals, and answers mental health questions with retrieval-augmented generation: a Groq-hosted LLM (`llama-3.3-70b-versatile`) writes the reply, grounded in a Qdrant knowledge base built from professional counseling Q&A and mental health books (PDFs).

The backend runs as a Docker app (see `Dockerfile`) and serves the SafeSpace AI frontend.

**⚠️ This is experimental.** Not a replacement for actual therapy or crisis intervention. Always escalate real emergencies.

---

## Quick Start

```bash
# 1. Install dependencies (+ the project itself, so `src.*` imports resolve)
pip install -r requirements.txt
pip install -e .

# 2. Prepare data (loads datasets, chunks PDFs)
python scripts/prepare_data.py

# 3. Train language detection + emotion classifier
python scripts/train_language_detector.py
python scripts/train_emotion_classifier.py   # BiLSTM, run on Kaggle GPU

# 4. Setup RAG (indexes knowledge base to Qdrant)
python scripts/setup_rag_pipeline.py

# 5. Start the API
uvicorn src.api.app:app --reload

# 6. Smoke-test the core models
python main.py
```

Open `http://localhost:8000/docs` for interactive API testing.

---

## What It Does

```
User: "I'm anxious about work presentations"
     ↓
[Language detection] → English (no translation needed)
[Emotion + Intent]   → fear  |  asking_mental_health_question
[Safety check]       → no crisis signals → continue
[NER extraction]     → symptoms: ["anxiety"], triggers: ["presentations", "work"]
[Query optimization] → rewrite for better retrieval
[HyDE]               → LLM drafts a hypothetical counsellor answer, used as the search query
[Semantic search]    → top-4 chunks from Qdrant (all-MiniLM-L6-v2 embeddings)
[RAG generation]     → Groq LLM replies using the chunks, emotion and session context
     ↓
Bot: "I understand presentation anxiety is really challenging. 
      Here are evidence-based approaches: breathing techniques, 
      visualization, gradual exposure..."
```

---

## Key Features

**Core NLP Modules**
- **Language detection**: TF-IDF + Logistic Regression over 20 languages (ar, bg, de, el, en, es, fr, hi, it, ja, nl, pl, pt, ru, sw, th, tr, ur, vi, zh); input is translated to English and replies are translated back
- **Emotion classification**: custom BiLSTM trained on Kaggle GPU; 6 labels (anger, fear, joy, love, sadness, surprise)
- **Intent classification**: zero-shot with the Groq LLM; 5 intents (`greeting`, `goodbye`, `gratitude`, `asking_mental_health_question`, `out_of_scope`) plus a crisis flag
- **NER**: extracts symptoms, triggers, duration and severity, accumulated across the session
- Optional LLM backends for language and emotion classification (see Environment)

**Safety**
- Every message passes a crisis guardrail: the intent classifier's crisis flag plus a tiered keyword/emotion safety check (`src/pipeline/safety_checks.py`)
- High-severity crises skip RAG and get an immediate de-escalation reply with crisis resources
- Medium-severity signals still get a normal answer, with helpline resources appended

**Retrieval & Generation**
- Query optimization + HyDE before retrieval, to bridge casual user wording and clinical knowledge-base language
- Semantic search in Qdrant (HNSW, cosine, all-MiniLM-L6-v2)
- Replies are conditioned on retrieved chunks, detected emotion and session context

**Conversation Memory**
- Per-process conversation history with optional LLM memory summarization
- Saying goodbye returns a short summary of the session

---

## How It Works (System Overview)

Step numbers match the comments in `src/api/routes/chat.py`.

```
User message
    │
    ▼
1. Language detection ──── non-English? ──► 2. Translate to English
    │                                              │
    ◄──────────────────────────────────────────────┘
    ▼
3. Emotion classification  ║  Intent classification      (run in parallel)
    │
    ▼
4. Safety check (keywords + emotion) + intent crisis flag
    │
    ├── crisis (high severity) ──► 5. De-escalation reply + crisis resources ──┐
    │                                                                          │
    ▼  intent?                                                                 │
    ├── greeting / gratitude / out_of_scope ──► canned reply ──────────────────┤
    ├── goodbye ──► farewell + session summary ────────────────────────────────┤
    │                                                                          │
    └── asking_mental_health_question                                          │
          │                                                                    │
          ▼                                                                    │
        6. NER  ║  Query optimization          (parallel)                      │
          ▼                                                                    │
        7. HyDE: LLM drafts a hypothetical counsellor answer                   │
          ▼                                                                    │
        8. Semantic search in Qdrant (top-4 chunks)                            │
          ▼                                                                    │
        9. LLM generation (chunks + emotion + session context)                 │
           + helpline resources if medium-severity signals                     │
          │                                                                    │
          ▼                                                                    │
       10. Update conversation history / NER context ◄────────────────────────┘
          ▼
       11. Translate reply back to the user's language (if needed)
          ▼
       Response + metadata (language, emotion, intent, sources, is_crisis, context)
```

---

## Installation & Setup

### Environment

```bash
# .env file (required)
GROQ_API_KEY=your_groq_key_here
QDRANT_URL=https://your-instance.qdrant.io
QDRANT_API_KEY=your_qdrant_key

# Optional (see src/config.py)
EMOTION_CLASSIFIER_BACKEND=model        # "model" (BiLSTM, default) | "llm"
LANGUAGE_CLASSIFIER_BACKEND=model       # "model" (TF-IDF, default) | "llm"
MEMORY_SUMMARIZATION_ENABLED=true
GROQ_API_KEY_EMOTION_CLASSIFIER=...     # used when EMOTION_CLASSIFIER_BACKEND=llm
GROQ_API_KEY_LANGUAGE_CLASSIFIER=...    # used when LANGUAGE_CLASSIFIER_BACKEND=llm
GROQ_API_KEY_SUMMARIZER=...             # used for memory summarization
```

Get keys:
- **Groq**: https://console.groq.com/ (free tier available)
- **Qdrant**: https://cloud.qdrant.io/ (free tier, 1GB limit)
- **W&B**: `wandb login` locally, or set `WANDB_API_KEY` in Kaggle secrets when training on Kaggle.

### Data & Models

Datasets auto-download from HuggingFace:
- Language identification (90k samples)
- Emotion detection (6k Twitter messages)
- Mental health Q&A (17k professional counseling pairs)
- Our PDFs from `data/raw/mental_health_books/`

**Note on emotion classifier**: Training was on Kaggle GPU for speed (`config.py` detects Kaggle and adjusts paths). The trained weights, tokenizer and config are committed under `models/emotion_classifier/` via Git LFS.

---

## API Usage

All routes are mounted under `/v1`.

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel anxious about work"}'
```

Response:
```json
{
  "response": "I understand work-related anxiety is really common...",
  "metadata": {
    "language": "en",
    "emotion": "fear",
    "intent": "asking_mental_health_question",
    "confidence_scores": {"language": 0.98, "emotion": 0.87, "intent": 0.95},
    "sources": [
      {"chunk_text": "anxiety management techniques...", "source": "...", "section": "...", "confidence": 0.82}
    ],
    "is_crisis": false,
    "context_accumulated": {
      "symptoms": ["anxiety"], "triggers": ["work"], "duration": "", "severity": ""
    }
  }
}
```

### Health

```bash
curl http://localhost:8000/v1/health          # liveness
curl http://localhost:8000/v1/health/models   # runs a sample through every loaded model
```

---

## Limitations

- **Not a therapist**: Responses are AI-generated from training data, not from licensed professionals. Always encourage users to seek real help.
- **Knowledge base dependent**: Quality depends on your PDFs and the counseling dataset. Garbage in = garbage out.
- **English-biased pipeline**: Non-English input is translated to English for retrieval and generation, then translated back, so quality depends on the translation step.
- **No user persistence**: This is a proof-of-concept. Conversation history lives in memory in a single shared per-process manager; it is lost on restart and not separated per user.
- **API rate limits**: Using free Groq tier (~30 req/min). Would need paid plan for production.

---

## Project Structure

```
mental_health_chatbot/
├── data/                             # Git-ignored; created by prepare_data.py
│   ├── raw/mental_health_books/      # Your PDFs
│   ├── processed/                    # emotion_{train,val,test}.csv, knowledge_base_combined.json
│   └── splits/                       # train/ val/ test/
├── models/
│   ├── language_detection/           # TF-IDF + LogisticRegression (model.pkl, vectorizer.pkl)
│   └── emotion_classifier/           # BiLSTM weights, tokenizer, config (Git LFS)
├── configs/
│   └── qdrant_config.yaml            # Qdrant connection, collection, HNSW, search settings
├── src/
│   ├── api/                          # FastAPI app
│   │   ├── app.py                    # App, CORS, model loading at startup
│   │   ├── dependencies.py
│   │   ├── schemas.py                # Request/response models
│   │   └── routes/                   # chat.py (/v1/chat), health.py (/v1/health)
│   ├── config.py                     # Env-driven backend selection (model vs LLM)
│   ├── language_detector/            # TF-IDF detector + LLM detector
│   ├── language_translator/          # Groq-based translation
│   ├── emotion_classifier/           # BiLSTM model, training, eval + LLM classifier
│   ├── intent_classifier/            # Groq zero-shot intent + crisis flag
│   ├── ner_extractor/                # Symptoms / triggers / duration / severity
│   ├── rag_engine/
│   ├── pipeline/                     # Legacy orchestrator + safety checks
│   ├── prompts/                      # Prompt templates
│   └── utils/                        # Embedder, vector DB, hybrid search, query optimizer,
│                                     # conversation manager, memory summarizer, preprocessing
├── scripts/
│   ├── prepare_data.py
│   ├── train_language_detector.py
│   ├── train_emotion_classifier.py
│   ├── train_emotion_classifier_encoder.py
│   ├── evaluate_emotion_classifier.py
│   ├── setup_rag_pipeline.py
│   └── test_intent_classifier.py
├── notebooks/                        # Data loading, language detection, emotion classifier
├── requirements/                     # base.txt, dev.txt
├── config.py                         # Training paths/hyperparameters (Kaggle-aware)
├── main.py                           # Local smoke test of intent/language/emotion models
├── Dockerfile                        # HF Spaces image (uvicorn on port 7860)
├── pyproject.toml
└── requirements.txt
```

---

## Testing

```bash
# Smoke-test intent, language and emotion models
python main.py

# Intent classifier only
python scripts/test_intent_classifier.py

# Interactive testing
uvicorn src.api.app:app --reload
# Then visit http://localhost:8000/docs, or GET /v1/health/models
```

The smoke test covers greetings, mental health questions, non-English input, out-of-scope questions, gratitude and crisis messages.

---

## Next Steps

If you want to extend this:

- **Better knowledge base**: Add more mental health PDFs or scrape legitimate resources
- **Evaluation metrics**: Measure response quality, user satisfaction, crisis detection accuracy
- **Fine-tuning the LLM**: Use Groq or another provider to fine-tune on your specific use case
- **Deployment**: Already ships as a Docker image for Hugging Face Spaces; could move to Kubernetes or serverless

---

## Notes

- Code assumes Python 3.10+
- GPU training on Kaggle (free), local inference on CPU is fine for a chatbot
- Uses semantic search (all-MiniLM-L6-v2) for embeddings—small, fast, good quality
- Qdrant's HNSW indexing is approximate but accurate enough for this use case

---

## License & Disclaimer

This is an educational project. Don't use it as actual mental health treatment. Always direct users in crisis to real resources:
- **US**: 988 Suicide & Crisis Lifeline
- **International**: findahelpline.com

---

## Tech stack

**Serving:** FastAPI · Uvicorn · Docker (Hugging Face Spaces)  
**LLM & retrieval:** Groq (Llama 3.3 70B) · Qdrant · sentence-transformers (all-MiniLM-L6-v2) · rank_bm25 · Docling (PDF parsing) · spaCy  
**ML & training:** PyTorch · scikit-learn · Hugging Face Datasets/Transformers · pandas · matplotlib · seaborn · Weights & Biases

