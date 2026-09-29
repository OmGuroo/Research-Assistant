from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "Multi-Agent Research Assistant"
    app_env: str = "development"
    debug: bool = True
    
    # Gemini Configuration
    gemini_api_key: Optional[str] = None
    gemini_model_name: str = "gemini-3.5-flash"
    gemini_temperature: float = 0.0
    
    # Active LLM Provider (groq or gemini)
    llm_provider: str = "groq"
    
    # Groq Configuration
    groq_api_key: Optional[str] = None
    groq_model_name: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.0
    
    # Tavily Configuration
    tavily_api_key: Optional[str] = None
    tavily_search_depth: str = "basic"
    tavily_max_results: int = 5
    tavily_include_answer: bool = False
    tavily_include_raw_content: bool = False
    
    # RAG Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chroma_persist_dir: str = "./data/chroma"
    
    # Adaptive Graph Configuration
    max_research_iterations: int = 2
    
    # We will load these from the .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
