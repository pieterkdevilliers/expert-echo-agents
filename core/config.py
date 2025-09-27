from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Setting config for main app
    """
    PROJECT_NAME: str = "ExpertEcho Agent Service"
    API_V1_PREFIX: str = "/api/v1"
    ENV: str = "development"
    DATABASE_URL: str
    AWS_SES_REGION: str
    AWS_ACCESS_KEY: str
    AWS_SECRET_KEY: str
    AWS_SES_VERIFIED_MAIL: str
    OPENAI_API_KEY: str
    AI_API_KEY: str
    dev_key: str

    class Config:
        """
        env config
        """
        env_file = ".env"


settings = Settings()
