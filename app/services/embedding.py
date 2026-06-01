from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def generate_embedding(text: str) -> list[float]:
    model = _load_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single model forward pass.

    Significantly faster than calling generate_embedding() in a loop because
    sentence-transformers processes the list as mini-batches internally.
    Use this for all bulk ingestion paths.

    Args:
        texts: List of strings to embed.  Empty list returns empty list.

    Returns:
        List of float vectors in the same order as the input texts.
    """
    if not texts:
        return []
    model = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]
