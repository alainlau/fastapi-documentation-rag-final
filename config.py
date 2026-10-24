"""Single source of truth for ingestion, retrieval, generation, and evaluation."""
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "corpus_dir": "data/corpus",
    "raw_docs_dir": "data/raw/docs",
    "chroma_dir": "chroma_db",
    "collection_name": "fastapi_docs",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 256,
    "chunk_overlap": 60,
    "max_embedding_tokens": 240,
    "similarity_metric": "cosine",
    "similarity_threshold": 0.35,
    "retrieval_mode": "similarity",  # first-stage: similarity or mmr
    "top_k": 12,                 # first-stage candidates, unchanged from Week 2
    "fetch_k": 20,               # only used with MMR
    "mmr_lambda": 0.6,
    "use_reranker": True,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "context_k": 4,              # evidence chunks passed to the generator
    "evaluation_k": 4,
    "judge_provider": "gemini",
    "judge_model": os.getenv("JUDGE_MODEL", "gemini-3.6-flash"),
    "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "refusal_sentence": "I cannot answer this question based on the provided documentation.",
}
