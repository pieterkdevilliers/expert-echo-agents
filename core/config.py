from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Setting config for main app
    """
    PROJECT_NAME: str = "ExpertEcho Agent Service"
    API_V1_PREFIX: str = "/api/v1"
    ENV: str = "development"
    OPENAI_API_KEY: str
    AI_API_KEY: str
    dev_key: str
    ENVIRONMENT: str
    CHAT_MODEL_NAME:str
    CHROMA_AUTH_TOKEN_TRANSPORT_HEADER: str
    CHROMA_ENDPOINT: str
    CHROMA_SERVER_AUTHN_CREDENTIALS: str
    CHROMA_SERVER_AUTHN_PROVIDER: str
    CHROMA_DEV_MODE: bool
    OPENAI_CHAT_MODEL: str
    LOGFIRE_ENABLED: bool

    class Config:
        """
        env config
        """
        env_file = ".env"

    headers = {
    'X-Chroma-Token': CHROMA_SERVER_AUTHN_CREDENTIALS,
    'Content-Type': 'application/json'
}


settings = Settings()
