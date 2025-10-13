from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Query(BaseModel):
    """
    Basic Query Structure
    """
    query: str
    prompt: str
    visitor_email: str
    visitor_uuid: str
    account_unique_id: str
    chat_history: List[Dict[str, Any]]
    relevance_score: float
    k_value: int
    sources_returned: int
    temperature: float
    chat_session_id: str
    scoreapp_report_text: Dict
    user_products_prompt: str

class UserQuery(BaseModel):
    """
    User Query Structure - for re-phrasing
    """
    query: str


# Define dependencies structure
class AgentDeps(BaseModel):
    query: str
    account_unique_id: str
    visitor_email: str
    chat_history: Optional[list] = None
    prompt_text: Optional[str] = None
    temperature: float = 0.2
    k_value: int = 7
    relevance_score: float = 0.7
    sources_returned: int = 3
    scoreapp_report_text: Dict[str, Any] = {}
    user_products_prompt: str = ""


class QueryRequest(BaseModel):
    query: str
    account_unique_id: str
    visitor_email: str
    chat_history: Optional[list] = None
    prompt_text: Optional[str] = None
    temperature: float = 0.2
    k_value: int = 7
    relevance_score: float = 0.7
    sources_returned: int = 3
    scoreapp_report_text: Dict[str, Any] = {}
    user_products_prompt: str = ""
    stream: bool = True  # Enable streaming by default