from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongodb_uri: str
    mongodb_db: str = "cdss"

    jwt_secret: str
    jwt_alg: str = "HS256"
    jwt_expire_minutes: int = 1440

    # Groq (OpenAI-compatible)
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    # Frontend-only Gemini key may be present in the shared .env.
    # Accept it here so pydantic does not treat it as an unexpected extra.
    gemini_api_key: str | None = None

    # Back-compat: if a user already set GROK_* in env, accept it too.
    # Prefer GROQ_* when both are present.
    grok_api_key: str | None = None
    grok_base_url: str = "https://api.x.ai/v1"
    grok_model: str = "grok-2-latest"

    upload_dir: str = "./uploads"
    chroma_dir: str = "./chroma"

    cors_allow_origins: list[str] = ["http://localhost:5173"]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Back-compat resolution
if not settings.groq_api_key and settings.grok_api_key:
    settings.groq_api_key = settings.grok_api_key
    settings.groq_base_url = settings.grok_base_url
    settings.groq_model = settings.grok_model
