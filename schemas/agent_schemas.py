from pydantic import BaseModel


class Query(BaseModel):
    """
    Basic Query Structure
    """
    query: str
    instructions: str
