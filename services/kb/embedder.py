"""Sentence-transformers embedder (Slice 6).

all-mpnet-base-v2 produces 768-dimensional normalized vectors.
Model is loaded once and reused across calls (singleton via module-level state).
"""
import threading
from typing import List

_model = None
_lock = threading.Lock()

MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIM = 768


def embed(text: str) -> List[float]:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
    return _model.encode(text, convert_to_numpy=True).tolist()
