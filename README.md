---
title: SafeSpace AI Backend
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Mental Health Support Chatbot with RAG

A chatbot trained on professional mental health counseling data and our own knowledge base (PDFs). Uses retrieval-augmented generation to provide grounded, empathetic responses to mental health queries.

**⚠️ This is experimental.** Not a replacement for actual therapy or crisis intervention. Always escalate real emergencies.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (loads datasets, chunks PDFs)
python scripts/00_prepare_data.py

# 3. Train language detection + emotion classifier
python scripts/01_train_language_detector.py
# (emotion classifier trains on Kaggle GPU - see docs/SETUP.md)

# 4. Setup RAG (indexes knowledge base to Qdrant)
python scripts/03_setup_rag.py

# 5. Start the API
uvicorn app.main:app --reload

# 6. Test it
python scripts/04_test_pipeline.py
```

Open `http://localhost:8000/docs` for interactive API testing.

---

## What It Does

```
User: "I'm anxious about work presentations"
     ↓
[Language detection] → English
[Emotion classifier] → Fear/Anxiety
[Intent detection] → asking_mental_health_question
[NER extraction] → symptoms: ["anxiety"], triggers: ["presentations", "work"]
[Query optimization] → rewrite for better retrieval
[Hybrid search] → BM25 keywords + semantic similarity
[RAG generation] → Groq LLM generates response using retrieved chunks
     ↓
Bot: "I understand presentation anxiety is really challenging. 
      Here are evidence-based approaches: breathing techniques, 
      visualization, gradual exposure..."
```

---

## Key Features

**Core NLP Modules**
- Multi-language detection (20 languages via TF-IDF)
- Emotion classification (DistilBERT fine-tuned on Kaggle GPU)
- Intent routing (zero-shot with Groq LLM)

---

## How It Works (System Overview)

```
User Query
    ↓
┌─────────────────────────┐
│  Language Detection     │ ← Detects input language
└────────┬────────────────┘
         ↓
    [Is English?]
         │
      ┌──┴──┐
      ↓     ↓
     Yes    No
      │     │
      │     ↓
      │  ┌──────────────────────┐
      │  │ Translate to English │
      │  └──────────┬───────────┘
      │             ↓
      └─────────────┘ (Translated Query)
                ↓
      ┌─────────────────────────┐
      │  Intent Classification  │ ← Routes the request
      └────────┬────────────────┘
               ↓
          [Intent type?]
               │
          ┌────┼────┬────┬─────┐
          ↓    ↓    ↓    ↓     ↓
       Greeting Goodbye Gratitude Mental_Health Out_of_Scope
             │                │
             ↓                ↓
        Direct Response   RAG Pipeline
                              │
                      ┌───────┼───────┐
                      ↓       ↓       ↓
                     NER  Query-Rewrite Hybrid-Search
                      │       │       │
                      └───────┴───┬───┘
                              ↓
                      Retrieved Chunks
                              │
                      ┌───────┴────────┐
                      ↓                ↓
                     HyDE         LLM Generation
                      │                │
                      └────────┬───────┘
                              ↓
                      Final Response (English)
                      + Sources
                      + Context
                              │
                    [Was input non-English?]
                              │
                          ┌───┴────┐
                          ↓        ↓
                         Yes       No
                          │        │
                          ↓        │
                   ┌──────────────────────┐
                   │ Translate Response   │
                   │ Back to Original     │
                   │ Language             │
                   └──────────┬───────────┘
                              ↓
                      Final Response
                      + Sources
                      + Context
                      (in Original Language)
```

---

## Installation & Setup

### Environment

```bash
# .env file (required)
GROQ_API_KEY=your_groq_key_here
QDRANT_URL=https://your-instance.qdrant.io
QDRANT_API_KEY=your_qdrant_key
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

**Note on emotion classifier**: Fine-tuning was on Kaggle GPU for speed. See `docs/SETUP.md` for instructions to export the model locally.

---

## API Usage

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel anxious about work"}'
```

Response:
```json
{
  "response": "I understand work-related anxiety is really common...",
  "emotion": "fear",
  "language": "en",
  "intent": "asking_mental_health_question",
  "confidence_scores": {
    "language": 0.98,
    "emotion": 0.87,
    "intent": 0.95
  },
  "sources": [
    {"text": "anxiety management techniques...", "source": "mental_health_dataset"},
    {"text": "breathing exercises for stress...", "source": "mental_health_books.pdf"}
  ]
}
```

### Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "session_1", "helpful": true, "comment": "great advice"}'
```

---

## Limitations

- **Not a therapist**: Responses are AI-generated from training data, not from licensed professionals. Always encourage users to seek real help.
- **Knowledge base dependent**: Quality depends on your PDFs and the counseling dataset. Garbage in = garbage out.
- **English-biased training**: Language detection works across 20 languages, but responses are English-only (for now).
- **No real user data**: This is a proof-of-concept. No real conversation history, no user persistence.
- **API rate limits**: Using free Groq tier (~30 req/min). Would need paid plan for production.

---

## Project Structure

```
mental-health-chatbot/
├── data/
│   ├── raw/                          # Downloads + your PDFs
│   ├── processed/                    # Cleaned data + chunks
│   └── splits/                       # Train/val/test
├── models/
│   ├── language_detection/           # TF-IDF + LogisticRegression
│   ├── emotion_classifier/           # DistilBERT (trained on Kaggle)
│   └── rag_components/               # Config files
├── src/
│   ├── modules/                      # Language, emotion, intent, RAG classes
│   ├── utils/                        # Preprocessing, embeddings, vector DB, NER
│   └── pipeline/                     # Orchestrator + safety checks
├── app/
│   └── main.py                       # FastAPI server
├── scripts/
│   ├── 00_prepare_data.py
│   ├── 01_train_language_detector.py
│   ├── 03_setup_rag.py
│   └── 04_test_pipeline.py
├── notebooks/
│   └── *.ipynb                       # Training + exploration
└── docs/
    ├── SETUP.md                      # Detailed setup
    ├── API_DOCUMENTATION.md
    └── ARCHITECTURE.md
```

---

## Testing

```bash
# Run full pipeline test
python scripts/04_test_pipeline.py

# Interactive testing
uvicorn app.main:app --reload
# Then visit http://localhost:8000/docs
```

Test queries cover:
- All 5 intent types (greeting, mental health Q, out of scope, etc.)
- Multi-language input
- Crisis detection
- Multi-turn conversation

---

## Next Steps

If you want to extend this:

- **Better knowledge base**: Add more mental health PDFs or scrape legitimate resources
- **Evaluation metrics**: Measure response quality, user satisfaction, crisis detection accuracy
- **Fine-tuning the LLM**: Use Groq or another provider to fine-tune on your specific use case
- **Deployment**: Docker + Kubernetes or serverless (AWS Lambda, etc.)

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

**Questions?** Check `docs/SETUP.md` for more detailed walkthroughs, or `docs/ARCHITECTURE.md` for technical deep dives.

## Tech stack

Python · PyTorch · Hugging Face Transformers · pandas · scikit-learn · matplotlib · seaborn · Weights & Biases

