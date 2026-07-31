import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.intent_classifier.intent_classifier import IntentClassifier
from src.language_translator import Translator
from src.ner_extractor import NERExtractor
from src.utils import QueryOptimizer, Embedder
from src import config

from src.api.routes import chat, health

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading NLP models into memory...")
    print(f"  Emotion classifier backend  : {config.emotion_classifier_backend()}")
    print(f"  Language classifier backend : {config.language_classifier_backend()}")
    app.state.lang_detector      = config.build_language_detector()
    app.state.emotion_classifier = config.build_emotion_classifier()
    app.state.intent_classifier  = IntentClassifier()
    app.state.translator         = Translator(groq_api_key=os.environ["GROQ_API_KEY"])
    app.state.ner_extractor      = NERExtractor()
    app.state.query_optimizer    = QueryOptimizer()
    app.state.embedder            = Embedder(groq_api_key=os.environ["GROQ_API_KEY"])
    print("All models loaded ✓")

    yield

    print("Shutting down...")
    del app.state.lang_detector
    del app.state.emotion_classifier
    del app.state.intent_classifier
    del app.state.translator
    del app.state.ner_extractor
    del app.state.query_optimizer
    del app.state.embedder

app = FastAPI(
    title="Mental Health Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://moemad07-safespace-ai-frontend.hf.space",
        "https://duospace-safespace-ai-frontend.hf.space",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router,   prefix="/v1", tags=["Chat"])
app.include_router(health.router, prefix="/v1", tags=["System Health"])

