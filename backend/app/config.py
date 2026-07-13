"""Configurazione centralizzata: legge .env via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "gemini"            # "gemini" | "ollama"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Modelli fastembed (multilingue IT+EN)
    dense_model: str = "intfloat/multilingual-e5-small"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    # Qdrant embedded
    qdrant_path: str = "./qdrant_data"
    collection: str = "documind"

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_n_rerank: int = 5
    use_hyde: bool = False

    # Chunking (word-based ≈ token)
    chunk_words: int = 350
    chunk_overlap: int = 60


settings = Settings()
