# embeddings.py
# Local sentence-transformers embedding pipeline producing 384-dim vectors for the semantic cache.
import asyncio
import re
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and return the singleton SentenceTransformer (CPU)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def normalize_text(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip())


def encode_prompt(prompt: str) -> np.ndarray:
    """Return an L2-normalized float32 embedding of shape (384,)."""
    vector = get_embedding_model().encode(
        normalize_text(prompt),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vector.astype(np.float32, copy=False)


async def async_encode_prompt(prompt: str) -> np.ndarray:
    """Non-blocking wrapper: runs encoding in a worker thread."""
    return await asyncio.to_thread(encode_prompt, prompt)
