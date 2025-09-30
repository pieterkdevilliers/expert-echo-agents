import os
from fastapi import Header, HTTPException
from core.config import settings
from dotenv import load_dotenv


load_dotenv()

ENVIRONMENT = os.environ.get('ENVIRONMENT')
if ENVIRONMENT == 'development':
    API_KEY = settings.dev_key

else: API_KEY = settings.AI_API_KEY


def verify_api_key(x_api_key: str = Header(...)):
    """
    Validate the received API Key for all API Calls
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


