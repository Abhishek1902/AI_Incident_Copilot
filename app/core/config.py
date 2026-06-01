from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Database ---
    DATABASE_URL: str

    # --- Embedding model (sentence-transformers) ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    TOP_K_RESULTS: int = 5

    # --- OpenAI (RAG) ---
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    # Separate model for the evaluator so you can use a cheaper/faster model
    # for judge calls without affecting answer quality.
    EVAL_MODEL: str = "gpt-4o-mini"

    # --- Retrieval & reranking ---
    # Initial candidates pulled from the vector index before reranking.
    RERANK_TOP_K: int = 20
    # Final documents passed to the LLM after the reranker trims the candidate list.
    FINAL_TOP_K: int = 5
    # Minimum cross-encoder logit to include an incident hit in the LLM prompt.
    # ms-marco logits range ~-10 (irrelevant) to ~+10 (highly relevant).
    RERANK_THRESHOLD: float = -2.0
    # Maximum total characters of document content allowed in a single prompt.
    # Keeps prompts within model context limits and reduces cost.
    MAX_CONTEXT_CHARS: int = 2000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
